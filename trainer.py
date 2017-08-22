import dynet as dy
import json, numpy
import utils, decode, eval

class TrainingProgress(object):
    '''
    Object used to store, serialize and deserialize pure python variables that change during training and should be preserved in order to properly restart the training process
    '''
    def load_from_json(self, file_name):
        self.__dict__.update(json.load(open(file_name, 'r')))

    def save_to_json(self, file_name):
        json.dump(self.__dict__, open(file_name, 'w'), indent=2)

    def __init__(self):
        self.bad_counter = 0
        self.anneal_restarts_done = 0
        self.uidx = 0                   # update
        self.eidx = 0                   # epoch
        self.estop = False
        self.hist_points = []
        self.hist_scores = []            # the bigger the better
        self.best_scores = -12345678
        self.best_point = -1

class Trainer(object):
    TP_TAIL = ".progress.json"
    TR_TAIL = ".trainer"
    ANNEAL_DECAY = 0.5
    CURR_PREFIX = "zcurr."
    BEST_PREFIX = "zbest."

    def __init__(self, opts, model):
        self.opts = opts
        self._tp = TrainingProgress()
        self._mm = model
        self.trainer = None
        self.validator = {"ll":self._validate_ll, "bleu":self._validate_bleu}[self.opts["valid_metric"]]
        self._set_trainer(True)

    def _set_trainer(self, renew):
        cur_lr = self.opts["lrate"] * (Trainer.ANNEAL_DECAY ** self._tp.anneal_restarts_done)
        if self.trainer is None:
            self.trainer = {"sgd": dy.SimpleSGDTrainer(self._mm.model, self.opts["lrate"]),
                            "momentum": dy.MomentumSGDTrainer(self._mm.model, self.opts["lrate"], self.opts["moment"]),
                            "adam": dy.AdamTrainer(self._mm.model, self.opts["lrate"])
                            }[self.opts["trainer_type"]]
        if renew:
            self.trainer.restart()
        self.trainer.learning_rate = cur_lr
        self.trainer.set_clip_threshold(self.opts["clip_c"])
        utils.printing("Set trainer %s with lr %s." % (self.opts["trainer_type"], cur_lr), func="info")

    # load and save
    def load(self, basename):
        # load model
        self._mm.load(basename)
        utils.printing("Reload model from %s." % basename, func="io")
        # load progress
        if self.opts["reload_training_progress"]:
            tp_name = basename + Trainer.TP_TAIL
            tr_name = basename + Trainer.TR_TAIL
            utils.printing("Reload trainer from %s and %s." % (tp_name, tr_name), func="io")
            self._tp.load_from_json(tp_name)
            if self._finished():
                utils.fatal('Training is already complete. Disable reloading of training progress (--no_reload_training_progress)'
                            'or remove or modify progress file (%s) to train anyway.')
            # set learning rate
            self._set_trainer(True)
            pass    # TODO, load-shadows

    def save(self, basename):
        # save model
        self._mm.save(basename)
        # save progress
        tp_name = basename + Trainer.TP_TAIL
        tr_name = basename + Trainer.TR_TAIL
        utils.printing("Save trainer to %s and %s." % (tp_name, tr_name), func="io")
        self._tp.save_to_json(tp_name)
        pass    # TODO, save-shadows

    # helpers
    def _finished(self):
        return self._tp.estop or self._tp.eidx >= self.opts["max_epochs"] \
                or self._tp.uidx >= self.opts["max_updates"]

    def _update(self):
        self.trainer.update()
        self._tp.uidx += 1

    def _validate_ll(self, dev_iter):
        # log likelihood
        one_recorder = utils.OnceRecorder("VALID-LL")
        for xs, ys, tk_x, tk_t in dev_iter:
            loss = self._mm.fb(xs, ys, False)
            one_recorder.record(xs, ys, loss, 0)
        one_recorder.report()
        return -1 * (one_recorder.loss / one_recorder.words)

    def _validate_bleu(self, dev_iter):
        # bleu score
        decode.decode(dev_iter, self._mm, self._mm.target_dict, self.opts, self.opts["dev-output"])
        s = eval.evaluate(self.opts["output"], self.opts["test"][1], self.opts["eval_metric"])
        raise s

    def _validate(self, dev_iter, name=None):
        # validate and log in the stats
        ss = ".e%s-u%s" % (self._tp.eidx, self._tp.uidx) if name is None else name
        with utils.Timer(name="Valid %s" % ss, print_date=True):
            score = self.validator(dev_iter)
            utils.printing("Validating %s for %s: score is %s." % (self.opts["valid_metric"], ss, score), func="info")
            # checkpoint - write current & best
            self.save(Trainer.CURR_PREFIX+self.opts["model"])
            if not self.opts["overwrite"]:
                self.save(self.opts["model"]+ss)
            ttp = self._tp
            ttp.hist_points.append(ss)
            ttp.hist_scores.append(score)
            if score > ttp.best_scores:
                ttp.bad_counter = 0     # reset
                ttp.best_scores = score
                ttp.best_point = len(ttp.hist_scores)-1
                self.save(Trainer.BEST_PREFIX+self.opts["model"])
            else:
                # anneal and early update
                ttp.bad_counter += 1
                if ttp.bad_counter >= self.opts["patience"]:
                    ttp.bad_counter = 0
                    if ttp.anneal_restarts_done < self.opts["anneal_restarts"]:
                        utils.printing("Patience up, annealing for %s." % (self._tp.anneal_restarts_done+1), func="info")
                        if self.opts["anneal_reload_best"]:
                            self.load(Trainer.BEST_PREFIX+self.opts["model"])   # load best
                        self._tp.anneal_restarts_done += 1                      # new tp now maybe
                        self._set_trainer(self.opts["anneal_renew_trainer"])
                    else:
                        utils.printing("Sorry, Early Update !!", func="warn")
                        ttp.estop = True

    # main training
    def train(self, train_iter, dev_iter):
        one_recorder = utils.OnceRecorder("CHECK")
        while not self._finished():     # epochs
            utils.printing("", func="info")
            with utils.Timer(name="Iter %s" % self._tp.eidx, print_date=True) as et:
                iter_recorder = utils.OnceRecorder("ITER-%s"%self._tp.eidx)
                for xs, ys, tk_x, tk_t in train_iter:
                    if numpy.random.random() < self.opts["rand_skip"]:     # introduce certain randomness
                        continue
                    # check length
                    if len(xs) and len(xs[0]) and len(xs[0][0]) != self.opts["factors"]:
                        utils.fatal("Mismatch for factors %s != %s" % (self.opts["factors"], len(xs[0][0])))
                    # training for one batch
                    loss = self._mm.fb(xs, ys, True)
                    self._update()
                    one_recorder.record(xs, ys, loss, 1)
                    iter_recorder.record(xs, ys, loss, 1)
                    if self.opts["debug"]:
                        mem0, mem1 = utils.get_statm()
                        utils.DEBUG("[%s/%s] after fb(%s):%s/%s" % (mem0, mem1, len(xs), max([len(i) for i in xs]), max([len(i) for i in ys])))
                    # time to validate and save best model ??
                    if self._tp.uidx % self.opts["valid_freq"] == 0:
                        one_recorder.report()
                        self._validate(dev_iter)
                        one_recorder.reset()
                        if self._finished():
                            break
                iter_recorder.report()
                self._validate(dev_iter, name=".e%s"%self._tp.eidx)
                self._tp.eidx += 1
