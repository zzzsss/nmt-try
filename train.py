import args, utils, model, trainer
from data import Dict, TextIterator
import os, random, numpy, sys

random.seed(12345)
numpy.random.seed(12345)
# --dynet-seed 12345

def main(opts):
    # 0. check options
    args.check_options(opts)
    # 1. obtain dictionaries
    source_corpus, target_corpus = opts["train"]
    source_dicts = []
    target_dict = None
    if not opts["rebuild_dicts"]:
        try:
            if len(opts["dicts_final"]) == opts["factors"]+1:     # directly read dicts
                source_dicts = [Dict.read(f) for f in opts["dicts_final"][:-1]]
                target_dict = Dict.read(opts["dicts_final"][-1])
            elif len(opts["dicts_raw"]) == opts["factors"]+1:     # cut from raw dicts
                source_dicts = [Dict(vname=f, thres=opts["dicts_thres"]) for f in opts["dicts_raw"][:-1]]
                target_dict = Dict(vname=opts["dicts_raw"][-1], thres=opts["dicts_thres"])
        except:
            utils.printing("Read dictionaries fail %s//%s, rebuild them." % (opts["dicts_final"], opts["dicts_raw"]), func="warn")
    if target_dict is None:
        # rebuild the dictionaries from corpus
        assert opts["factors"] == 1     # other factors from outside dictionaries
        source_dicts = [Dict(fname=source_corpus, thres=opts["dicts_thres"])]
        target_dict = Dict(fname=target_corpus, thres=opts["dicts_thres"])
        # save dictionaries
        try:
            for d, name in zip(source_dicts, opts["dicts_final"]):
                d.write(name)
            target_dict.write(opts["dicts_final"][-1])
        except:
            utils.printing("Write dictionaries fail: %s, skip this step." % opts["dicts_final"], func="warn")
    # 2. corpus iterator
    train_iter = TextIterator(source_corpus, target_corpus, source_dicts, target_dict,
                              batch_size=opts["batch_size"], maxlen=opts["max_len"], use_factor=(opts["factors"]>1))
    dev_iter = TextIterator(opts["dev"][0], opts["dev"][1], source_dicts, target_dict,
                              batch_size=opts["valid_batch_size"], maxlen=None, use_factor=(opts["factors"]>1),
                              skip_empty=False, shuffle_each_epoch=False, sort_by_length=False)
    # 3. about model & trainer
    mm = model.NMTModel(opts, source_dicts, target_dict)
    tt = trainer.Trainer(opts, mm)  # trainer + training_progress
    if opts["reload"] and os.path.exists(opts["reload_model_name"]):
        tt.load(opts["reload_model_name"])
    # 4. training
    tt.train(train_iter, dev_iter)
    utils.printing("=== Training ok!! ===", func="info")

if __name__ == '__main__':
    utils.printing("cmd: %s" % ' '.join(sys.argv))
    opts = args.init("train")
    main(vars(opts))
