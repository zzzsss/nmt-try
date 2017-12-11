import os
import random

from . import args, model, utils
import numpy
from .data import Dict, TextIterator

from . import trainer

random.seed(12345)
numpy.random.seed(12345)
# --dynet-seed 12345

def main0(opts):
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
    train_iter = TextIterator(source_corpus, target_corpus, source_dicts, target_dict, sort_type=opts["training_sort_type"],
                              batch_size=opts["batch_size"], maxlen=opts["max_len"], use_factor=(opts["factors"]>1),
                              shuffle_each_epoch=opts["shuffle_training_data"])
    # special restoring for test/dev-iter
    dev_iter = TextIterator(opts["dev"][0], opts["dev"][1], source_dicts, target_dict,
                              batch_size=opts["valid_batch_width"], maxlen=None, use_factor=(opts["factors"]>1),
                              skip_empty=False, shuffle_each_epoch=False, sort_type="src", is_dt=True, onelen=50)
    # 3. about model & trainer
    mm = model.NMTModel(opts, source_dicts, target_dict)
    tt = trainer.Trainer(opts, mm)  # trainer + training_progress
    if opts["reload"] and os.path.exists(opts["reload_model_name"]):
        tt.load(opts["reload_model_name"], opts["reload_training_progress"])
    # 4. training
    tt.train(train_iter, dev_iter)
    utils.printing("=== Training ok!! ===", func="info")

def main():
    opts = args.init("train")
    utils.init_print()
    main0(opts)

if __name__ == '__main__':
    main()
