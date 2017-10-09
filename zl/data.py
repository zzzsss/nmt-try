# data iterators and dictionaries

from . import utils
import json
import numpy as np

# The class of vocabulary or dictionary: conventions: 0:zeros, 1->wordceil:words, then:special tokens
class Vocab(object):
    _special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>", ]    # PLUS <non>: 0

    @staticmethod
    def _build_vocab(f, rthres, fthres, start_idx):
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
        # write with filters
        v = {}
        cur_idx = start_idx
        for ii, ww in enumerate(words):
            rank, freq = ii, word_freqs[ww]
            if rank <= rthres and freq >= fthres:
                v[ww] = cur_idx
                cur_idx += 1
        return v, words

    @staticmethod
    def _getd_from_file(fname, rthres, fthres):
        with utils.Timer(tag="vocab", info="read vocabs from corpus"):
            with utils.zopen(fname) as f:
                d, words = Vocab._build_vocab(f, rthres, fthres, 1)
            for one in Vocab._special_tokens:
                d[one] = len(d)
            utils.zlog("Build Dictionary: (origin=%s, final=%s, special=%s)." % (len(words), len(d), len(Vocab._special_tokens)+1))
        return d

    def __init__(self, d=None, fname=None, rthres=100000, fthres=1):
        self.d = {}
        if d is not None:
            self.d = d
        else:
            self.d = Vocab._getd_from_file(fname, rthres, fthres)
        # reverse vocab
        self.v = ["" for _ in range(len(self.d))]
        for k in self.d:
            self.v[self.d[k]] = k

    def write(self, wf):
        with utils.zopen(wf, 'w') as f:
            f.write(json.dumps(self.d, ensure_ascii=False, indent=2))
            utils.zlog("-- Write Dictionary to %s: Finish %s." % (wf, len(self.d)), func="io")

    @staticmethod
    def read(rf):
        with utils.zopen(rf) as f:
            df = json.loads(f.read())
            utils.zlog("-- Read Dictionary from %s: Finish %s." % (rf, len(df)), func="io")
            return Vocab(d=df)

    # queries
    def get_wordceil(self):
        return self.d["<unk>"]+1      # excluding special tokens except unk

    @property
    def eos(self):
        return self.d["<eos>"]

    @property
    def pad(self):
        return self.d["<pad>"]

    @property
    def unk(self):
        return self.d["<unk>"]

    @property
    def bos(self):
        return self.d["<bos>"]

    def getw(self, index):
        return self.v[index]

    def __getitem__(self, item):
        assert type(item) == str
        if item in self.d:
            return self.d[item]
        else:
            return self.d["<unk>"]

    def __len__(self):
        return len(self.d)

    # words <=> indexes (be aware of lists)
    @staticmethod
    def w2i(dicts, ss, use_factor):
        if type(dicts) == list:
            tmp = []
            for w in ss:
                if use_factor:
                    w = [dicts[i][f] for (i,f) in enumerate(w.split('|'))]
                else:
                    w = [dicts[0][w]]
                tmp.append(w)
            tmp.append([d.eos for d in dicts] if use_factor else [dicts[0].eos])  # add eos
        else:
            tmp = [dicts[w] for w in ss]
            tmp.append(dicts.eos)
        return tmp

    @staticmethod
    def i2w(dicts, ii):
        return [dicts._getw(i) for i in ii[:-1]]    # no eos


# some helpers
