import time, sys, os, subprocess, random, json

_printing_heads = {
    "plain":"-- ", "time":"## ", "io":"== ", "info":"** ",
    "warn":"!! ", "fatal":"KI ", "debug":"DE ", "none":""
}
def printing(s, func="plain"):
    print(_printing_heads[func]+s)

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
