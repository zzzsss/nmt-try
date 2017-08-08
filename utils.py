# some useful functions
import time, sys, os, subprocess, random, json

# tools
from tools import shuffle, get_final_vocab, get_origin_vocab

# printing functions
from tools import printing

def DEBUG(s):
    printing(s, func="debug")

def fatal(s):
    printing(s, func="dead")
    printing("================= FATAL, exit =================", func="none")
    sys.exit()

def get_statm():
    with open("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4/1024) + "MiB"
    try:
        p = subprocess.Popen("nvidia-smi | grep -E '%s.*MiB'" % os.getpid(), shell=True, stdout=subprocess.PIPE)
        line = p.stdout.readlines()
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

class Timer:
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

class Accu:
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
