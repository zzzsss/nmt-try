# re-ranking with k-best list or possibly with gold ones
# also some kind of analysis from here
from . import mt_args, mt_eval, mt_mt
from zl.data import Vocab, get_arranger_simple
from zl import utils

def main():
    # init
    opts = mt_args.init("rerank")
    # special readings from args for re-ranking mode
    # only accept spaced (multi-mode) nbest files for target & non-multi for golds
    # 1. data (only accepting nbest files)
    source_dict, target_dict = Vocab.read(opts["dicts"][0]), Vocab.read(opts["dicts"][1])
    dicts = [source_dict] + [target_dict for _ in opts["test"][1:]]
    # -- no sorting
    test_iter = get_arranger_simple(opts["test"], dicts, multis=True, batch_size=["test_batch_size"])
    gold_iter = get_arranger_simple(opts["gold"], [target_dict for _ in opts["gold"]], multis=True, batch_size=["test_batch_size"])
    utils.zcheck_matched_length(test_iter, gold_iter)
    # 2. model
    mm = []
    try:
        for mn in opts["models"]:
            x = mt_mt.s2sModel(opts, source_dict, target_dict, None)     # rebuild from opts, thus use the same opts when testing
            x.load(mn)
            mm.append(x)
    except:
        pass
    # 3. analysis
    if len(mm) == 0:
        utils.zlog("No models specified, only analysing!", func="warn")
    # 4. rerank
    else:
        utils.zlog("=== Start to rerank ===", func="info")
        pass

if __name__ == '__main__':
    main()
