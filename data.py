# some routines for preparing data and dictionaries

import json, numpy
import utils

class Dict:
    def __init__(self, d=None, fname=None, vname=None, thres=None):
        # load the json file, if thres<10: min-len, else: max-size
        self.d = {}
        if d is not None:
            self.d = d
        else:
            # obtain original dictionary df
            if vname is not None:
                df = json.loads(vname)
            else:
                with utils.Timer(name="Dictionary", info="read vocabs from corpus"):
                    with utils.zfopen(fname) as f:
                        df = utils.get_origin_vocab(f)
            # filtering function
            self.d = utils.get_final_vocab(df, thres)
        # for debugging
        self.v = ["" for _ in range(len(self.d))]
        for k in self.d:
            self.v[self.d[k]] = k

    def get_num_words(self):
        return self.d["<eos>"]      # excluding special tokens

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
    def start(self):
        return self.d["<go!!>"]

    def write(self, wf):
        with utils.zfopen(wf, 'w') as f:
            f.write(json.dumps(self.d, ensure_ascii=False, indent=2))
            utils.printing("-- Write Dictionary to %s: Finish %s." % (wf, len(self.d)), func="io")

    @staticmethod
    def read(rf):
        with utils.zfopen(rf) as f:
            df = json.loads(f.read())
            utils.printing("-- Read Dictionary from %s: Finish %s." % (rf, len(df)), func="io")
            return Dict(d=df)

    def _getw(self, index):
        # assert type(index) == int
        return self.v[index]

    def __getitem__(self, item):
        assert type(item) == str
        if item in self.d:
            return self.d[item]
        else:
            return self.d["<unk>"]

    def __len__(self):
        return len(self.d)

    # words <=> indexes
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

# ======================= (data_iterator from nematus) ===================== #

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, source_dicts, target_dict, batch_size=None, maxlen=None, use_factor=False,
                 skip_empty=True, is_dt=False, shuffle_each_epoch=True, sort_type="non", maxibatch_size=20, onelen=80):
        # data
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            with utils.Timer(name="Shuffle", info="shuffle the bilingual corpus"):
                self.source, self.target = utils.shuffle([self.source_orig, self.target_orig])
        else:
            self.source = utils.zfopen(source, 'r')
            self.target = utils.zfopen(target, 'r')
        self.source_dicts = source_dicts
        self.target_dict = target_dict
        # options
        self.batch_size = batch_size
        self.init_batch_size = batch_size
        self.maxlen = maxlen if maxlen is not None else 100000        # ignore sentences above this
        self.onelen = onelen    # single batch out if >= onelen
        self.skip_empty = skip_empty
        self.use_factor = use_factor
        self.shuffle = shuffle_each_epoch
        self.sort_type = sort_type
        self.source_buffer = []
        self.target_buffer = []
        self.maxibatch_size = maxibatch_size
        # self.sort_ksize = sort_ksize
        self.is_dt = is_dt      # is dev/test ? => sort all if sorting and special treating for long sentences
        self.len_sort_indexes = []
        self.long_back = None
        self.num_sents = None

    @property
    def k(self):
        # return self.sort_ksize
        return self.batch_size * self.maxibatch_size

    # information about sort-by-lengths
    @staticmethod
    def _SBL_combine(a, b):
        MAXLEN = 1000   # this should enough
        UNIT = 4
        return MAXLEN * (a//UNIT) + b
    SBL_TYPES = {
        "src-trg": [True, lambda s,t: TextIterator._SBL_combine(s,t)],
        "trg-src": [True, lambda s,t: TextIterator._SBL_combine(t,s)],
        "src": [True, lambda s,t:s],
        "trg": [True, lambda s,t:t],
        "plus": [True, lambda s,t:s+t],
        "times": [True, lambda s,t:s*t],
        "non": [False, None]
    }
    def SBL_need_sort(self):
        return TextIterator.SBL_TYPES[self.sort_type][0]
    def SBL_sort_buffer_index(self):
        ff = TextIterator.SBL_TYPES[self.sort_type][1]
        tlen = numpy.array([ff(len(s), len(t)) for s,t in zip(self.source_buffer, self.target_buffer)])
        tidx = tlen.argsort()
        tidx = [i for i in reversed(tidx)]
        return tidx     # tidx is huge=>small, but reading is popping from the back

    def restore_sort_by_length(self, s):
        # restore the ordering in-place
        assert len(s) == len(self.len_sort_indexes)     # no discarding of 0 and long sentences
        assert not self.shuffle
        if not self.SBL_need_sort():
            return
        r = [None for _ in self.len_sort_indexes]
        for ind, sent in zip(reversed(self.len_sort_indexes), s):
            r[ind] = sent
        return r

    def __iter__(self):
        return self

    def __len__(self):
        # return num of sentences
        if self.num_sents is None:
            self.num_sents = 0
            for i in self:
                self.num_sents += len([x for x in i][0])
        return self.num_sents

    # only at the ending status
    def bsize(self, bs=None):
        if bs is None:
            return self.batch_size
        else:
            self.batch_size = int(bs)
            return self.batch_size

    def reset(self):
        if self.shuffle:
            with utils.Timer(name="Shuffle", info="shuffle the bilingual corpus"):
                try:
                    [fd.close() for fd in (self.source, self.target)]
                except:
                    utils.printing("closing error.", func="warn")
                self.source, self.target = utils.shuffle([self.source_orig, self.target_orig])
        else:
            self.source.seek(0)
            self.target.seek(0)

    def __next__(self):
        source, target, tokens_src, tokens_trg = [], [], [], []
        # fill buffer, if it's empty
        # utils.DEBUG("hh")
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        if len(self.source_buffer) == 0 or len(self.source_buffer) < self.batch_size:
            for ss in self.source:
                # utils.DEBUG(ss)
                ss = ss.split()
                tt = self.target.readline().split()
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):  # empty
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:      # >max-len
                    continue
                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if not self.is_dt and len(self.source_buffer) == self.k:                   # full
                    break
            # utils.DEBUG("here")
            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.reset()
                raise StopIteration
            # sort by target buffer
            if self.SBL_need_sort():
                tidx = self.SBL_sort_buffer_index()
                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                if self.is_dt:
                    self.len_sort_indexes = tidx
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
        # actual work here
        while True:
            # from backs
            if self.long_back is not None:
                tmp = self.long_back
                self.long_back = None
                return tmp
            # read from source file and map to word index
            try:
                ss = self.source_buffer.pop()
                tokens_src.append(ss)
            except IndexError:
                break
            sss = Dict.w2i(self.source_dicts, ss, self.use_factor)
            # read from source file and map to word index
            tt = self.target_buffer.pop()
            tokens_trg.append(tt)
            ttt = Dict.w2i(self.target_dict, tt, False)
            source.append(sss)
            target.append(ttt)
            # special treating with long sentences (mainly for dev and test, thus only check src)
            if len(ss) > self.onelen:
                self.long_back = tuple([x.pop()] for x in (source, target, tokens_src, tokens_trg))
                if len(source) == 0:
                    continue
                else:
                    break
            if len(source) >= self.batch_size or len(target) >= self.batch_size:
                break
        if len(source) == 0:
            self.reset()
            raise StopIteration
        return (source, target, tokens_src, tokens_trg)
