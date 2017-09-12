# some useful functions
import time, sys, os, subprocess, random, json, platform

# tools
from tools import shuffle, get_final_vocab, get_origin_vocab, zfopen
from tools import printing as helper_print

# print and log
class Logger(object):
    MAGIC_CODE = "sth_magic_that_cannot_be_conflicted"
    printing_log_file = None

    @staticmethod
    def start_log(s):
        Logger.end_log()
        if s == Logger.MAGIC_CODE:
            s = "%s-%s.log" % (platform.uname().node, '-'.join(time.ctime().split()[-2:]))
        Logger.printing_log_file = zfopen(s, "w")
        printing("Start logging at %s" % Logger.printing_log_file)

    @staticmethod
    def end_log():
        if Logger.printing_log_file is not None:
            Logger.printing_log_file.close()

def printing(s, func="plain", out=sys.stderr):
    helper_print(s, func, out)
    if Logger.printing_log_file is not None:
        helper_print(s, func, Logger.printing_log_file)

def init_print():
    printing("*cmd: %s" % ' '.join(sys.argv))
    printing("*platform: %s" % ' '.join(platform.uname()))

def DEBUG(s):
    printing(s, func="debug")

def DEBUG_check(b):
    if not b:
        fatal("assert %s failed." % b)

def fatal(s):
    printing(s, func="fatal")
    printing("================= FATAL, exit =================", func="none")
    # sys.exit()
    raise s

def get_statm():
    with zfopen("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4//1024) + "MiB"
    try:
        p = subprocess.Popen("nvidia-smi | grep -E '%s.*MiB'" % os.getpid(), shell=True, stdout=subprocess.PIPE)
        line = p.stdout.readlines()
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

class Timer(object):
    NAMED = {}
    def __init__(self, name=None, cname=None, print_date=False, quiet=False, info=""):
        self.name = name
        self.cname = cname
        self.print_date = print_date
        self.quiet = quiet
        self.info = info
        self.accu = 0.   # accumulated time
        self.paused = False
        self.start = time.time()

    def pause(self):
        if not self.paused:
            cur = time.time()
            self.accu += cur - self.start
            self.start = cur
            self.paused = True

    def resume(self):
        if not self.paused:
            printing("Timer should be paused to be resumed.", func="warn")
        else:
            self.start = time.time()
            self.paused = False

    def get_time(self):
        self.pause()
        self.resume()
        return self.accu

    def end(self):
        self.pause()
        if self.cname is not None:
            if self.cname not in Timer.NAMED:
                Timer.NAMED[self.cname] = 0
            Timer.NAMED[self.cname] += self.accu

    def __enter__(self):
        cur_date = time.ctime() if self.print_date and not self.quiet else ""
        printing("Start timer %s: %s at %s. (%s)" % (self.name, self.info, time.time(), cur_date), func="time") if not self.quiet else None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        cur_date = time.ctime() if self.print_date and not self.quiet else ""
        printing("End timer %s at %s, the period is %s seconds. (%s)" % (self.name, time.time(), self.accu, cur_date), func="time") if not self.quiet else None
        return False

    @staticmethod
    def show_accu(x=""):
        if x == "":
            print(Timer.NAMED)
        else:
            print("%s: %s"%(x,Timer.NAMED[x]))

    @staticmethod
    def del_accu(x=""):
        if x == "":
            Timer.NAMED = {}
        else:
            Timer.NAMED[x] = 0.

class Accu(object):
    NAMED = {}
    METHODS = {}
    INITS = {}
    @staticmethod
    def show_nums(x=""):
        if x == "":
            print(Accu.NAMED)
        else:
            print("%s: %s"%(x,Accu.NAMED[x]))

    @staticmethod
    def del_nums(x=""):
        if x == "":
            Accu.NAMED = {}
        else:
            Accu.NAMED[x] = Accu.INITS[x]()

    @staticmethod
    def get_num(x=""):
        return Accu.NAMED[x]

    @staticmethod
    def add_num(x, f, init, func, inplace=False):
        if x not in Accu.NAMED:
            Accu.NAMED[x] = init()
            Accu.INITS[x] = init
            Accu.METHODS[x] = func
        if inplace:
            Accu.METHODS[x](Accu.NAMED[x], f)
        else:
            Accu.NAMED[x] = Accu.METHODS[x](Accu.NAMED[x], f)

class OnceRecorder(object):
    def __init__(self, name):
        self.name = name
        self.loss = 0.
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = Timer()

    def record(self, src, trg, loss, update):
        self.loss += loss
        self.sents += len(src)
        self.words += sum([len(x) for x in src])     # for src
        self.updates += update

    def reset(self):
        self.loss = 0.
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = Timer()

    def get(self, k):
        return {"loss_per_word":self.loss / self.words}[k]

    # const, only reporting, could be called many times
    def report(self, head=""):
        one_time = self.timer.get_time()
        loss_per_sentence = self.loss / self.sents
        loss_per_word = self.loss / self.words
        sent_per_second = float(self.sents) / one_time
        word_per_second = float(self.words) / one_time
        printing(head + "Recoder <%s>, %s(time)/%s(updates)/%s(sents)/%s(words)/%s(sl-loss)/%s(w-loss)/%s(s-sec)/%s(w-sec)" % (self.name, one_time, self.updates, self.sents, self.words, loss_per_sentence, loss_per_word, sent_per_second, word_per_second), func="info")
