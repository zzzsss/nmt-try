# some routines for preparing data and dictionaries

import json, gzip, numpy
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
                    with open(fname) as f:
                        df = utils.get_origin_vocab(f)
            # filtering function
            self.d = utils.get_final_vocab(df, thres)

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

    def write(self, wf):
        with open(wf, 'w') as f:
            f.write(json.dumps(self.d, ensure_ascii=False))
            utils.printing("-- Write Dictionary to %s: Finish %s." % (wf, len(self.d)), func="io")

    @staticmethod
    def read(rf):
        with open(rf) as f:
            df = json.loads(f)
            utils.printing("-- Read Dictionary from %s: Finish %s." % (rf, len(df)), func="io")
            return Dict(d=df)

    def __getitem__(self, item):
        if item in self.d:
            return self.d[item]
        else:
            return self.d["<unk>"]

    def __len__(self):
        return len(self.d)

# ======================= (data_iterator from nematus) ===================== #

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, source_dicts, target_dict, batch_size=None, maxlen=None, use_factor=False,
                 skip_empty=True, shuffle_each_epoch=False, sort_by_length=True, maxibatch_size=20):
        # data
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            with utils.Timer(name="Shuffle", info="shuffle the bilingual corpus"):
                self.source, self.target = utils.shuffle([self.source_orig, self.target_orig])
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = source_dicts
        self.target_dict = target_dict
        # options
        self.batch_size = batch_size
        self.maxlen = maxlen
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

    def next(self):
        source = []
        target = []
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
            except IndexError:
                break
            tmp = []
            for w in ss:
                if self.use_factor:
                    w = [self.source_dicts[i][f] for (i,f) in enumerate(w.split('|'))]
                else:
                    w = [self.source_dicts[0][w]]
                tmp.append(w)
            tmp.append([d.eos for d in self.source_dicts] if self.use_factor else [self.source_dicts[0].eos])  # add eos
            ss = tmp
            # read from source file and map to word index
            tt = self.target_buffer.pop()
            tt = [self.target_dict[w] for w in tt]
            tt.append(self.target_dict.eos)     # add eos
            source.append(ss)
            target.append(tt)
            if len(source) >= self.batch_size or len(target) >= self.batch_size:
                break
        return source, target
