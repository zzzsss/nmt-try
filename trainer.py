import dynet as dy
import json, numpy, os, subprocess
import utils

class TrainingProgress(object):
    '''
    Object used to store, serialize and deserialize pure python variables that change during training and should be preserved in order to properly restart the training process
    '''
    def load_from_json(self, file_name):
        self.__dict__.update(json.load(open(file_name, 'rb')))

    def save_to_json(self, file_name):
        json.dump(self.__dict__, open(file_name, 'wb'), indent=2)

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
    TR_TAIL = ".??"     # TODO
    ANNEAL_DECAY = 0.5
    CURR_PREFIX = "zcurr."
    BEST_PREFIX = "zbest."

    def __init__(self, opts, model):
        self.opts = opts
        self._tp = TrainingProgress()
        self._mm = model
        self.trainer = None
        self.validator = {"ll":self._validate_ll, "bleu":self._validate_bleu}[self.opts["validMetric"]]
        self._set_trainer(True)

    def _set_trainer(self, renew):
        cur_lr = self.opts["lrate"] * (Trainer.ANNEAL_DECAY ** self._tp.anneal_restarts_done)
        if renew:
            self.trainer = {"sgd": dy.SimpleSGDTrainer(self._mm.model, self.opts["lrate"]),
                            "momentum": dy.MomentumSGDTrainer(self._mm.model, self.opts["lrate"], self.opts["moment"]),
                            "adam": dy.AdamTrainer(self._mm.model, self.opts["lrate"])
                            }[self.opts["trainer_type"]]
        utils.printing("Set trainer %s with lr %s." % (self.opts["trainer_type"], cur_lr), func="info")
        self.trainer.learning_rate = cur_lr
        self.trainer.set_clip_threshold(self.opts["clip_c"])

    # load and save, TODO currently cannot load/save trainer (param-shadows)
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
            # TODO: real reload

    def save(self, basename):
        # save model
        self._mm.save(basename)
        # save progress
        tp_name = basename + Trainer.TP_TAIL
        tr_name = basename + Trainer.TR_TAIL
        utils.printing("Save model and trainer to %s, %s and %s." % (basename, tp_name, tr_name), func="io")
        self._tp.save_to_json(tp_name)

    # helpers
    def _finished(self):
        return self._tp.estop or self._tp.eidx >= self.opts["max_epochs"] \
                or self._tp.uidx >= self.opts["max_updates"]

    def _update(self):
        self.trainer.update()
        self._tp.uidx += 1

    def _validate_ll(self, dev_iter):
        # log likelihood
        one_loss = 0.
        one_sents = 0
        for xs, ys in dev_iter:
            one_sents += len(xs)
            loss = self._mm.fb(xs, ys, False)
            one_loss += loss
        return -1 * (one_loss / one_sents)

    def _validate_bleu(self, dev_iter):
        # TODO
        raise NotImplementedError("TODO")

    def _validate(self, dev_iter):
        # validate and log in the stats
        ss = ".e%s-u%s" % (self._tp.eidx, self._tp.uidx)
        with utils.Timer(name="Valid %s" % ss, print_date=True):
            score = self.validator(dev_iter)
            utils.printing("Validating %s for %s: score is %s." % (self.opts["validMetric"], ss, score), func="info")
            # checkpoint - write current & best
            self.save(Trainer.CURR_PREFIX+self.opts["model"])
            if not self.opts["overwrite"]:
                self.save(self.opts["model"]+ss)
            ttp = self._tp
            ttp.hist_points.append(ss)
            ttp.hist_scores.append(score)
            if score > ttp.best_scores:
                ttp.best_scores = score
                ttp.best_point = len(ttp.hist_scores)-1
                self.save(Trainer.BEST_PREFIX+self.opts["model"])
            else:
                # anneal and early update
                ttp.bad_counter += 1
                if ttp.bad_counter >= self.opts["patience"]:
                    ttp.bad_counter = 0
                    if ttp.anneal_restarts_done < self.opts["anneal_restarts"]:
                        if self.opts["anneal_reload_best"]:
                            self.load(Trainer.BEST_PREFIX+self.opts["model"])   # load best
                        self._tp.anneal_restarts_done += 1                      # new tp now maybe
                        self._set_trainer(self.opts["anneal_renew_trainer"])
                    else:
                        utils.printing("Sorry, Early Update !!", func="warn")
                        ttp.estop = True

    # main training
    def train(self, train_iter, dev_iter):
        one_loss = 0.
        one_sents = one_updates = 0
        one_timer = utils.Timer()
        while not self._finished():     # epochs
            with utils.Timer(name="Iter %s" % self._tp.eidx, print_date=True) as et:
                n_samples = 0
                for xs, ys in train_iter:
                    if numpy.random.random() < self.opts["rand_skip"]:     # introduce certain randomness
                        continue
                    # check length
                    if len(xs) and len(xs[0]) and len(xs[0][0]) != self.opts["factors"]:
                        utils.fatal("Mismatch for factors %s != %s" % (self.opts["factors"], len(xs[0][0])))
                    n_samples += len(xs)
                    # training for one batch
                    if True:
                        mem = '?'
                        # p = subprocess.Popen(["cat", "/proc/self/statm"], stdout=subprocess.PIPE)
                        # rss = [l for l in p.stdout][0].split()
                        # p = subprocess.Popen("ps -ef -o pid,rss | grep %s" % os.getpid(), shell=True, stdout=subprocess.PIPE)
                        # rss = [l for l in p.stdout][0].split()
                        # utils.DEBUG("list is %s" % rss)
                        # mem = int(rss[1])*4/1024
                        utils.DEBUG("[%s MB] before fb:%s/%s" % (mem, max([len(i) for i in xs]), max([len(i) for i in ys])))
                    # loss = 0.
                    loss = self._mm.fb(xs, ys, True)
                    self._update()
                    one_updates += 1
                    one_loss += loss
                    one_sents += len(xs)
                    # time to validate and save best model ??
                    if self._tp.uidx % self.opts["validFreq"] == 0:
                        one_time = one_timer.get_time()
                        utils.printing("At this checkpoint, %s(time)/%s(updates)/%s(sents)/%s(loss-per)/%s(time-per)"
                                       % (one_time, one_updates, one_sents, one_loss/one_sents, one_time/one_sents), func="info")
                        self._validate(dev_iter)
                        one_loss = 0.
                        one_sents = one_updates = 0
                        one_timer = utils.Timer()
                        if self._finished():
                            break
                utils.printing("This iter train for %s within %s." % (n_samples, et.get_time()), func="info")
                self._tp.eidx += 1
