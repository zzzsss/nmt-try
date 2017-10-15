from zl import utils, data

def _test_utils():
    log_file = "test.log"
    utils.init(log_file)
    utils.zlog("this starts the test", func="debug")
    with utils.Timer("test", print_date=True):
        utils.zcheck(100==1+99, "math", "fatal")
        utils.zcheck(100==1, "math2", "warn")
    with utils.Timer("test", print_date=True, accumulated=True):
        for _ in range(100):
            z = utils.Random.binomial(1, 0.1, 100, "test")
        utils.zlog(z)
    with utils.Timer("test", print_date=True, accumulated=True):
        utils.zcheck(100==1+99, "math", "fatal")
        for _ in range(100):
            z = utils.Random.binomial(1, 0.1, 100, "test")
        utils.zcheck(100==1, "math2", "warn")
    utils.zlog(utils.Task.get_accu(), func="info")
    with open(log_file) as f:
        w = [i for i in utils.ZStream.stream_on_file(f)]
    utils.zlog(w)
    utils.zforce(utils.zcheck_ff_iter, [1,2,3], lambda x: x>0, "what")

def _test_data_iter():
    # get vocab
    files = ["utils.py", "utils.py"]
    vv = [data.Vocab(fname=f, fthres=1+i) for i, f in enumerate(files)]
    # get data
    aa = data.get_arranger(files, vv, True, True, [0,1], 4, 5, 20, 2, 15)
    for ds in aa.arrange_batches():
        utils.zlog(ds)

if __name__ == '__main__':
    _test_utils()
    _test_data_iter()
