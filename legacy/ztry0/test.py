import sys

from . import args, decode, eval, model, utils
from .data import Dict, TextIterator


def main0(opts):
    # 1. datas
    source_dicts = [Dict.read(f) for f in opts["dicts_final"][:-1]]
    target_dict = Dict.read(opts["dicts_final"][-1])
    # -- here no need for test[1], but for convenience ...
    test_iter = []
    if opts["decode_type"] != "loop":
        # special restoring for test/dev-iter (here big maxibatch-size)
        test_iter = TextIterator(opts["test"][0], opts["test"][1], source_dicts, target_dict,
                              batch_size=opts["test_batch_size"], maxlen=None, use_factor=(opts["factors"]>1),
                              skip_empty=False, shuffle_each_epoch=False, sort_type="src", is_dt=True, onelen=50)
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
    if opts["decode_type"] == "decode":
        utils.printing("=== Start to decode ===", func="info")
        with utils.Timer(name="Decoding", print_date=True):
            decode.decode(test_iter, mm, target_dict, opts, opts["output"])
        utils.printing("=== End decoding, write to %s ===" % opts["output"], func="info")
        eval.evaluate(opts["output"], opts["test"][1], opts["eval_metric"])
    elif opts["decode_type"] == "decode_gold":
        utils.printing("=== Start to decode (with gold in reranking) ===", func="info")
        pgfile = opts["output"]+".gold"
        with utils.Timer(name="Decoding_Gold", print_date=True):
            decode.decode_gold(test_iter, mm, target_dict, opts, opts["output"], pgfile)
        utils.printing("=== End decoding, write to %s and %s ===" % (opts["output"], pgfile), func="info")
        # todo(warn), restoring here, that is to say: using bpe-ref for this test
        eval.evaluate(opts["output"], opts["test"][1], opts["eval_metric"], True)
        eval.evaluate(pgfile, opts["test"][1], opts["eval_metric"], True)
    elif opts["decode_type"] == "loop":
        while True:
            utils.printing("Enter the src to translate:")
            line = sys.stdin.readline()
            if len(line)==0:
                break
            sss = Dict.w2i(source_dicts, line.split(), (opts["factors"]>1))
            rs = decode.search([sss], mm, opts, opts["decode_way"], opts["decode_batched"])
            best_seq = rs[0][0].get_path("last_action")
            strs = Dict.i2w(target_dict, best_seq)
            utils.printing(" ".join(strs), func="none", out=sys.stdout)
    elif opts["decode_type"].startswith("test"):
        one_recorder = utils.OnceRecorder("Test-LL")
        ff = {"test1":mm[0].fb, "test2":mm[0].fb2}[opts["decode_type"]]
        utils.printing("Testing log likelihood, only using the first model; running %s." % str(ff))
        for xs, ys, tk_x, tk_t in test_iter:
            loss = ff(xs, ys, False)
            if type(loss) == list:  # for fb2
                loss = sum(loss)
            one_recorder.record(xs, ys, loss, 0)
        one_recorder.report()
    else:
        pass

def main():
    opts = args.init("test")
    utils.init_print()
    main0(opts)

if __name__ == '__main__':
    main()