# some useful functions
import sys, gzip, platform, subprocess, os, time
import numpy as np

# Part 0: loggings
def zopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding="utf-8")
    else:
        return open(filename, mode, encoding="utf-8")

def zlog(s, func="plain", flush=True):
    Logger._instance._log(str(s), func, flush)

class Logger(object):
    _instance = None
    _logger_heads = {
        "plain":"-- ", "time":"## ", "io":"== ", "info":"** ", "score":"%% ",
        "warn":"!! ", "fatal":"KI ", "debug":"DE ", "none":"**INVALID-CODE**"
    }
    @staticmethod
    def _get_ch(func):  # get code & head
        if func not in Logger._logger_heads:
            func = "none"
        return func, Logger._logger_heads[func]
    MAGIC_CODE = "sth_magic_that_cannot_be_conflicted"

    @staticmethod
    def init(files):
        s = "%s-%s.log" % (platform.uname().node, '-'.join(time.ctime().split()[-4:]))
        files = [f if f!=Logger.MAGIC_CODE else s for f in files]
        ff = dict((f, True) for f in files)
        lf = dict((l, True) for l in Logger._logger_heads)
        Logger._instance = Logger(ff, lf)
        zlog("START!!", func="plain")

    # =====
    def __init__(self, file_filters, func_filters):
        self.file_filters = file_filters
        self.func_filters = func_filters
        self.fds = {}
        # the managing of open files (except outside handlers like stdio) is by this one
        for f in self.file_filters:
            if isinstance(f, str):
                self.fds[f] = zopen(f, mode="w")
            else:
                self.fds[f] = f

    def __del__(self):
        for f in self.file_filters:
            if isinstance(f, str):
                self.fds[f].close()

    def _log(self, s, func, flush):
        func, head = Logger._get_ch(func)
        if self.func_filters[func]:
            ss = head + s
            for f in self.fds:
                if self.file_filters[f]:
                    print(ss, file=self.fds[f], flush=flush)

    # todo (register or filter files & codes)

# Part 1: checkers
def zcheck(ff, ss, func):
    if Checker._checker_enabled:
        Checker._instance._check(ff, ss, func)

def zforced_check(ff, ss):
    Checker._instance._check(ff, ss, "forced")

def zfatal():
    raise RuntimeError()

# should be used when debugging or only fatal ones, comment out if real usage
class Checker(object):
    _instance = None
    _checker_filters = {"warn": True, "fatal": True}
    _checker_handlers = {"warn": (lambda: 0), "fatal": (lambda: zfatal())}
    _checker_enabled = True  # todo(warn) a better way to disable is to comment out

    @staticmethod
    def init(enabled):
        Checker._checker_enabled = enabled
        Checker._instance = Checker(Checker._checker_filters, Checker._checker_handlers)

    # =====
    def __init__(self, func_filters, func_handlers):
        self.func_filters = func_filters
        self.func_handlers = func_handlers

    def _check(self, form, ss, func):
        if self._checker_filters[func]:
            if not form:
                zlog(ss, func=func)
                self.func_handlers[func]()

    def _forced_check(self, form, ss):
        if not form:
            zlog(ss, func="fatal")
            zfatal()

    # todo (manage filters and recordings)

# Part 2: info
def get_statm():
    with zopen("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4//1024) + "MiB"
    try:
        p = subprocess.Popen("nvidia-smi | grep -E '%s.*MiB'" % os.getpid(), shell=True, stdout=subprocess.PIPE)
        line = p.stdout.readlines()
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

# keep times and other stuffs
class Task(object):
    _accu_info = {}  # accumulated info

    @staticmethod
    def get_accu(x=None):
        if x is None:
            return Task._accu_info
        else:
            return Task._accu_info[x]

    def __init__(self, tag, accumulated):
        self.tag = tag
        self.accumulated = accumulated

    def init_state(self):
        raise NotImplementedError()

    def begin(self):
        raise NotImplementedError()

    def end(self, s=None):
        raise NotImplementedError()

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.accumulated:
            if self.tag not in Task._accu_info:
                Task._accu_info[self.tag] = self.init_state()
            Task._accu_info[self.tag] = self.end(Task._accu_info[self.tag])
        else:
            self.end()

class Timer(Task):
    START = 0.

    @staticmethod
    def init():
        Timer.START = time.time()

    @staticmethod
    def systime():
        return time.time()-Timer.START

    def __init__(self, tag, info="", accumulated=False, print_date=False, quiet=False):
        super(Timer, self).__init__(tag, accumulated)
        self.print_date = print_date
        self.quiet = quiet
        self.info = info
        self.accu = 0.   # accumulated time
        self.paused = False
        self.start = None

    def pause(self):
        if not self.paused:
            cur = Timer.systime()
            self.accu += cur - self.start
            self.start = cur
            self.paused = True

    def resume(self):
        if not self.paused:
            zlog("Timer should be paused to be resumed.", func="warn")
        else:
            self.start = Timer.systime()
            self.paused = False

    def get_time(self):
        self.pause()
        self.resume()
        return self.accu

    def init_state(self):
        return 0.

    def begin(self):
        self.start = Timer.systime()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            zlog("Start timer %s: %s at %.3f. (%s)" % (self.tag, self.info, self.start, cur_date), func="time")

    def end(self, s=None):
        self.pause()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            zlog("End timer %s: %s at %.3f, the period is %.3f seconds. (%s)" % (self.tag, self.info, Timer.systime(), self.accu, cur_date), func="time")
        # accumulate
        if s is not None:
            return s+self.accu
        else:
            return None

# Part 3: randomness (all from numpy)
class Random(object):
    _seeds = {}

    @staticmethod
    def get_generator(task):
        if task not in Random._seeds:
            one = 1
            for t in task:
                one = one * ord(t) // (2**31)
            Random._seeds[task] = np.random.RandomState(one)
        return Random._seeds[task]

    @staticmethod
    def init():
        np.random.seed(12345)

    @staticmethod
    def _function(task, *argv):
        rg = Random.get_generator(task)
        return getattr(rg, task)(*argv)

    @staticmethod
    def shuffle(xs):
        Random._function("shuffle", xs)

    @staticmethod
    def binomial(n, p, size):
        return Random._function("binomial", n, p, size)

    @staticmethod
    def ortho_weight(ndim):
        W = Random._function("randn", ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(np.float)

# Calling once at start, init them all
def init(extra_file=Logger.MAGIC_CODE):
    Logger.init([extra_file, sys.stderr])
    Checker.init(True)
    Timer.init()
    Random.init()

# ========================================================== #
# outside perspective: init, zlog, zcheck, Random, Timer
def _test():
    init("test.log")
    zlog("this starts the test", func="debug")
    with Timer("test", print_date=True):
        zcheck(100==1+99, "math", "fatal")
        zcheck(100==1, "math2", "warn")
    with Timer("test", print_date=True, accumulated=True):
        for _ in range(100):
            z = Random.binomial(1, 0.1, 100)
        zlog(z)
    with Timer("test", print_date=True, accumulated=True):
        zcheck(100==1+99, "math", "fatal")
        for _ in range(100):
            z = Random.binomial(1, 0.1, 100)
        zcheck(100==1, "math2", "warn")
    zlog(Task.get_accu(), func="info")

if __name__ == '__main__':
    _test()
