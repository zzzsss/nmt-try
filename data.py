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
            f.write(json.dumps(self.d, ensure_ascii=False))
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
                 skip_empty=True, shuffle_each_epoch=True, sort_by_length=True, maxibatch_size=20):
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
        self.maxlen = maxlen if maxlen is not None else 100000        # ignore sentences above this
        self.skip_empty = skip_empty
        self.use_factor = use_factor
        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length
        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        #self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

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
        source = []
        target = []
        tokens_src, tokens_trg = [], []
        # fill buffer, if it's empty
        # utils.DEBUG("hh")
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        if len(self.source_buffer) == 0:
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
                if len(self.source_buffer) == self.k:                   # full
                    break
            # utils.DEBUG("here")
            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                #self.end_of_data = False
                self.reset()
                raise StopIteration
            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                tidx = [i for i in reversed(tidx)]
                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
        # actual work here
        while True:
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
            if len(source) >= self.batch_size or len(target) >= self.batch_size:
                break
        return source, target, tokens_src, tokens_trg
