# some useful functions
import time, sys, os, tempfile, random, json

# printing functions
_printing_heads = {
    "plain":"-- ", "time":"## ", "io":"== ", "info":"** ",
    "warn":"!! ", "fatal":"KI ", "debug":"DE ", "none":""
}
def printing(s, func="plain"):
    print(_printing_heads[func]+s)

def DEBUG(s):
    printing(s, func="debug")

def fatal(s):
    printing(s, func="dead")
    printing("================= FATAL, exit =================", func="none")
    sys.exit()

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

# tools
def shuffle(files):
    with open(files[0]) as f:
        lines = [[i.strip()] for i in f]
    for ff in files[1:]:
        with open(ff) as f:
            for i, li in enumerate(f):
                lines[i].append(li.strip())
    random.shuffle(lines)
    # write
    for ii, ff in enumerate(files):
        path, filename = os.path.split(os.path.realpath(ff))
        with open(filename+'.shuf', 'w') as f:
            for l in lines:
                f.write(l[ii]+"\n")
    # read
    fds = []
    for ff in files:
        path, filename = os.path.split(os.path.realpath(ff))
        fds.append(open(filename+'.shuf', 'r'))
    return fds

def get_origin_vocab(f):
    word_freqs = {}
    # read
    for line in f:
        words_in = line.strip().split()
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1
    # sort
    words = [w for w in word_freqs]
    words = sorted(words, key=lambda x: word_freqs[x], reverse=True)
    # write
    v = {}
    for ii, ww in enumerate(words):
        v[ww] = {"rank":ii, "freq":word_freqs[ww]}
    return v

def get_final_vocab(v, thres):
    # filter
    MAGIC_THRES = 100
    d = {"<non>": 0}
    # filtering function
    ff = lambda x: True
    if thres > MAGIC_THRES:
        ff = lambda x: x["rank"] < thres
    elif thres <= MAGIC_THRES:
        ff = lambda x: x["freq"] >= thres
    for k in v:
        if ff(v[k]):
            d[k] = len(d)
    printing("Build Dictionary: Cut from %s to %s." % (len(v), len(d)-1))
    # special
    for s in ["<eos>", "<pad>", "<unk>"]:
        d[s] = len(d)
    printing("Build Dictionary: Finish %s." % (len(d)))
    return d

# utils with cmd
def main():
    # cmd: python *.py raw // python *.py cut <thres> // python *.py shuffle
    if sys.argv[1] == shuffle:
        shuffle(sys.argv[1:])
    else:
        worddict = get_origin_vocab(sys.stdin)
        if sys.argv[1] == "raw":
            v = worddict
        elif sys.argv[1] == "cut":
            v = get_final_vocab(worddict, int(sys.argv[2]))
        else:
            v = []
        json.dump(v, sys.stdout, ensure_ascii=False)
        return

if __name__ == '__main__':
    main()
