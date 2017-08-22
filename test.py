import args, utils, model, decode, eval
from data import Dict, TextIterator
import sys

def main(opts):
    # 0. check options
    args.check_options(opts)
    # 1. datas
    source_dicts = [Dict.read(f) for f in opts["dicts_final"][:-1]]
    target_dict = Dict.read(opts["dicts_final"][-1])
    # -- here no need for test[1], but for convenience ...
    test_iter = None
    if not opts["loop"]:
        test_iter = TextIterator(opts["test"][0], opts["test"][1], source_dicts, target_dict,
                              batch_size=opts["valid_batch_size"], maxlen=None, use_factor=(opts["factors"]>1),
                              skip_empty=False, shuffle_each_epoch=False, sort_by_length=False)
    # 2. model
    mm = []
    for mn in opts["models"]:
        x = model.NMTModel(opts, source_dicts, target_dict)     # rebuild from opts, thus use the same opts when testing
        x.load(mn)
        mm.append(x)
    if len(mm) == 0:
        utils.printing("No models specified, must be testing mode?", func="warn")
        mm.append(model.NMTModel(opts, source_dicts, target_dict))      # no loading, only for testing
    # 3. decode
    utils.printing("=== Start to decode ===", func="info")
    if not opts["loop"]:
        with utils.Timer(name="Decoding", print_date=True):
            decode.decode(test_iter, mm, target_dict, opts, opts["output"])
        utils.printing("=== End decoding, write to %s ===" % opts["output"], func="info")
        eval.evaluate(opts["output"], opts["test"][1], opts["eval_metric"])
    else:
        while True:
            utils.printing("Enter the src to translate:")
            line = sys.stdin.readline()
            if len(line)==0:
                break
            sss = Dict.w2i(source_dicts, line.split(), (opts["factors"]>1))
            rs = decode.search([sss], mm, opts, opts["decode_way"], opts["decode_batched"])
            strs = Dict.i2w(target_dict, rs[0])
            utils.printing(" ".join(strs), func="none", out=sys.stdout)

if __name__ == '__main__':
    opts = args.init("test")
    main(vars(opts))
