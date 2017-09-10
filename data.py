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
                 skip_empty=True, shuffle_each_epoch=True, sort_type="non", maxibatch_size=20):
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
        self.sort_type = sort_type
        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        #self.end_of_data = False
        self.num_batches = None
        # indexes by sorting by length
        self.len_sort_indexes_cache = []
        self.len_sort_indexes = []

    # information about sort-by-lengths
    SBL_TYPES = {
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
        return tidx

    def restore_sort_by_length(self, s):
        # restore the ordering in-place
        if self.shuffle or not self.SBL_need_sort():
            return
        cur_start = 0
        for tidx in self.len_sort_indexes:
            new_s = [None for _ in range(len(tidx))]
            for i, idx in enumerate(reversed(tidx)):
                new_s[idx] = s[cur_start+i]
            s[cur_start:cur_start+len(tidx)] = new_s
            cur_start += len(tidx)
        assert cur_start == len(s), "Wrong length of results"

    def __iter__(self):
        return self

    def __len__(self):
        if self.num_batches is None:
            self.num_batches = sum([1 for _ in self])
        return self.num_batches

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
        # sorted_indexes
        self.len_sort_indexes = self.len_sort_indexes_cache
        self.len_sort_indexes_cache = []

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
            if self.SBL_need_sort():
                tidx = self.SBL_sort_buffer_index()
                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                if not self.shuffle:    # no meaning who shuffling (training) #TODO: bad dependencies
                    self.len_sort_indexes_cache.append(tidx)
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
