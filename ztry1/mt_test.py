import sys
from . import mt_args, mt_search, mt_eval
from zl import utils
from zl.data import Vocab, get_arranger
from .mt_mt import s2sModel, mt_decode

def main():
    # init
    opts = mt_args.init("test")
    # 1. data
    source_dict, target_dict = Vocab.read(opts["dicts"][0]), Vocab.read(opts["dicts"][1])
    # -- here no need for test[1], but for convenience ...
    test_iter = get_arranger(opts["test"], [source_dict, target_dict], shuffling_corpus=False, shuflling_buckets=False, sort_prior=[0], batch_size=opts["test_batch_size"], maxibatch_size=-1, max_len=utils.Constants.MAX_V, min_len=0, one_len=opts["max_len"])
    # 2. model
    mm = []
    for mn in opts["models"]:
        x = s2sModel(opts, source_dict, target_dict)     # rebuild from opts, thus use the same opts when testing
        x.load(mn)
        mm.append(x)
    if len(mm) == 0:
        utils.zlog("No models specified, must be testing mode?", func="warn")
        mm.append(s2sModel(opts, source_dict, target_dict))      # no loading, only for testing
    # 3. decode
    utils.zlog("=== Start to decode ===", func="info")
    with utils.Timer(tag="Decoding", print_date=True):
        mt_decode(test_iter, mm, target_dict, opts, opts["output"])
    utils.zlog("=== End decoding, write to %s ===" % opts["output"], func="info")
    mt_eval.evaluate(opts["output"], opts["test"][1], opts["eval_metric"])

if __name__ == '__main__':
    main()
