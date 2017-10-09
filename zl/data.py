# data iterators and dictionaries

from . import utils
import json, os, sys
import numpy as np

# ========== Vocab ========== #
# The class of vocabulary or dictionary: conventions: 0:zeros, 1->wordceil:words, then:special tokens
class Vocab(object):
    SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>", ]    # PLUS <non>: 0
    NON_TOKEN = "<non>"

    @staticmethod
    def _build_vocab(stream, rthres, fthres, specials):
        word_freqs = {}
        # read
        for w in stream:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1
        # sort
        words = [w for w in word_freqs]
        words = sorted(words, key=lambda x: word_freqs[x], reverse=True)
        # write with filters
        v = {Vocab.NON_TOKEN: 0}    # todo(warn) hard-coded
        cur_idx = 1
        for ii, ww in enumerate(words):
            rank, freq = ii, word_freqs[ww]
            if rank <= rthres and freq >= fthres:
                v[ww] = cur_idx
                cur_idx += 1
        # add specials
        for one in specials:
            v[one] = len(v)
        utils.zlog("Build Dictionary: (origin=%s, final=%s, special=%s as %s)." % (len(words), len(v), len(specials)+1, specials))
        return v, words

    def __init__(self, d=None, s=None, fname=None, rthres=100000, fthres=1, specials=None):
        # three possible sources: d=direct-dict, s=iter(str), f=tokenized-file
        if specials is None:    # todo(warn) default is like this
            specials = Vocab.SPECIAL_TOKENS
        self.d = {}
        if d is not None:
            self.d = d
            # insure specials are in here
            utils.zforce(utils.zcheck_ff_iter, specials, lambda x: x in self.d, "not included specials", "warn")
            utils.zforce(utils.zcheck_ff, self.d[Vocab.NON_TOKEN], lambda x: x==0, "unequal to 0", "warn")
        elif s is not None:
            with utils.Timer(tag="vocab", info="build vocabs from stream."):
                self.d, _ = Vocab._build_vocab(s, rthres, fthres, specials)
        elif fname is not None:
            with utils.Timer(tag="vocab", info="build vocabs from corpus %s" % fname):
                with utils.zopen(fname) as f:
                    self.d, _ = Vocab._build_vocab(utils.ZStream.stream_on_file(f), rthres, fthres, specials)
        else:
            utils.zfatal("No way to init Vocab.")
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
        utils.zcheck_type(item, str)
        if item in self.d:
            return self.d[item]
        else:
            return self.d["<unk>"]

    def __len__(self):
        return len(self.d)

    # words <=> indexes (be aware of lists)
    @staticmethod
    def w2i(dicts, ss, add_eos=True, use_factor=False, factor_split='|'):
        # Usage: list(Vocab), list(str) => list(list(int))[use_factor] / list(int)[else]
        if not isinstance(dicts, list):
            dicts = [dicts]
        utils.zcheck_ff_iter(dicts, lambda x: isinstance(x, Vocab), "Wrong type")
        # lookup
        tmp = []
        for w in ss:
            if use_factor:
                idx = [dicts[i][f] for (i,f) in enumerate(w.split(factor_split))]
            else:
                idx = dicts[0][w]
            tmp.append(idx)
        if add_eos:
            tmp.append([d.eos for d in dicts] if use_factor else dicts[0].eos)  # add eos
        return tmp

    @staticmethod
    def i2w(dicts, ii, rm_eos=True, factor_split='|'):
        # Usage: list(Vocab), list(int)/list(list(int)) => list(str)
        if not isinstance(dicts, list):
            dicts = [dicts]
        utils.zcheck_ff_iter(dicts, lambda x: isinstance(x, Vocab), "Wrong type")
        tmp = []
        # get real list
        real_ii = ii
        if len(ii)>0 and rm_eos and ii[-1][0]==dicts[0].eos:
            real_ii = ii[:-1]
        # transform each token
        for one in real_ii:
            if not isinstance(one, list):
                one = [one]
            utils.zcheck_matched_length(one, dicts)
            tmp.append(factor_split.join([v.getw(idx) for v, idx in zip(dicts, one)]))
        return tmp

# ========== Instance and Reader ========== #
# handling the batching of instances, also filtering, sorting, recording, etc.
class BatchArranger(object):
    def __init__(self, streamer, batch_size, maxibatch_size, outliers, single_outlier, sorting_keyer, tracking_order, shuffling):
        self.streamer = streamer
        self.outliers = [] if outliers is None else outliers
        self.single_outliers = (lambda x: False) if single_outlier is None else single_outlier
        self.sorting_keyer = sorting_keyer  # default(None) no sorting
        self.tracking_order = tracking_order
        self.tracking_list = None
        self.batch_size = batch_size
        # todo(notice): if <=0 then read all at one time and possibly sort all
        self.maxibatch_size = maxibatch_size if maxibatch_size>0 else sys.maxsize
        self.shuffling = shuffling  # shuffle inside the maxi-batches?

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.streamer)

    @property
    def k(self):
        # return self.sort_ksize
        return self.batch_size * self.maxibatch_size

    # getter and checker for batch_size (might have unknown effects if changed in the middle)
    def bsize(self, bs=None):
        if bs is None:
            return self.batch_size
        else:
            self.batch_size = int(bs)
            return self.batch_size

    # the iterating generator
    def arrange_batches(self):
        # read streams of batches of items, return list-of-items
        buffer = []
        if self.tracking_order:
            self.tracking_list = []
        inst_stream = self.streamer.stream()
        while True:
            # read into buffer
            for idx, one in enumerate(inst_stream):
                # filters out (like short or long instances)
                filtered_flag = False
                for tmp_filter in self.outliers:
                    if tmp_filter(one):
                        filtered_flag = True
                        break
                if filtered_flag:
                    continue
                # need to generate single instance
                if self.single_outliers(one):
                    if self.tracking_order:
                        self.tracking_list.append(idx)
                    yield [one]
                # adding in buffer with index
                buffer.append((idx, one))
                if len(buffer) == self.k:  # must reach equal because +=1 each time
                    break
            # time for yielding
            if len(buffer) > 0:
                # sorting
                sorted_buffer = buffer
                if self.sorting_keyer is not None:
                    sorted_buffer = sorted(buffer, key=(lambda x: self.sorting_keyer(x[1])))
                # prepare buckets
                buckets = [sorted_buffer[_s:_s+self.batch_size] for _s in range(0, len(sorted_buffer), self.batch_size)]
                # another shuffle?
                if self.shuffling:
                    utils.Random.shuffle(buckets, "data_bucket")
                # final yielding
                for oneb in buckets:
                    if self.tracking_order:
                        self.tracking_list += [_one[0] for _one in oneb]
                    yield [_one[1] for _one in oneb]
                buffer = []
            else:
                break

    def restore_order(self, x):
        # rearrange back according to self.tracking_list (caused by sorting and shuffling)
        utils.zforce(utils.zcheck, self.tracking_order, "Tracking function has not been opened")
        utils.zforce(utils.zcheck_matched_length, self.tracking_list, x)
        ret = [None for _ in x]
        for idx, one in zip(self.tracking_list, x):
            utils.zforce(utils.zcheck_type, ret[idx], type(None), "Wrong tracking list, internal error!!")
            ret[idx] = one
        return ret

# single instance, batched-version should be handled when modeling since it will be quite different for different models
class Instance(object):
    pass

# read and return Instance
class InstanceReader(object):
    def stream(self):
        raise NotImplementedError()

# ========== Specified Instance and DataIter for simple seq2seq texts ========== #
class TextInstance(Instance):
    # multiple sequences of indexes
    def __init__(self, words, idxes):
        utils.zcheck_matched_length(words, idxes)
        self.words = words
        self.idxes = idxes

    def __repr__(self):
        return "||".join([" ".join(one) for one in self.words])

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        return self.idxes[item]

    def get_lens(self):
        return [len(ww) for ww in self.words]

    def get_origin(self, i):
        return self.words[i]

class TextInstanceRangeOutlier(object):
    # detect outlier, return True if outlier detected
    def __init__(self, a=None, b=None):
        self.a = -1*sys.maxsize if a is None else a
        self.b = sys.maxsize if b is None else b

    def __call__(self, inst):  # True if any not in [a, b)
        return any(not (x >= self.a and x < self.b) for x in inst.get_lens())

class TextInstanceLengthSorter(object):
    # information about sort-by-lengths
    def __init__(self, prior):
        self.prior = prior

    def __call__(self, inst):
        # return the key for one inst
        MAXLEN, UNIT = 1000, 4  # todo(warn), magic number
        acc = 0
        lens = inst.get_lens()
        for i in reversed(self.prior):
            acc = acc * MAXLEN + lens[i] // UNIT
        return acc

# read from files
class TextFileReader(InstanceReader):
    def __init__(self, files, vocabs, shuffling):
        utils.zforce(utils.zcheck_matched_length, files, vocabs)
        self.files = files
        self.vocabs = vocabs
        self.shuffling = shuffling
        self.num_insts = -1

    @staticmethod
    def shuffle_corpus(files):
        # global shuffle, creating tmp files on current dir
        with utils.zopen(files[0]) as f:
            lines = [[i.strip()] for i in f]
        for ff in files[1:]:
            with utils.zopen(ff) as f:
                for i, li in enumerate(f):
                    lines[i].append(li.strip())
        utils.Random.shuffle(lines, "corpus")
        # write
        filenames_shuf = []
        for ii, ff in enumerate(files):
            path, filename = os.path.split(os.path.realpath(ff))
            filenames_shuf.append(filename+'.%d.shuf'%ii)   # avoid identical file-names
            with utils.zopen(filenames_shuf[-1], 'w') as f:
                for l in lines:
                    f.write(l[ii]+"\n")
        # read
        fds = [utils.zopen(_f) for _f in filenames_shuf]
        return fds

    def __len__(self):
        # return num of instances (use cached value)
        if self.num_insts < 0:
            self.num_insts = sum(len(i) for i in self)
        return self.num_insts

    def stream(self):
        if self.shuffling:
            with utils.Timer(tag="shuffle", info="shuffle the file corpus"):
                fds = TextFileReader.shuffle_corpus(self.files)
        else:
            fds = [utils.zopen(one) for one in self.files]
        # read them and yield --- checking length
        for idx, ss in enumerate(fds[0]):
            self.num_insts = max(self.num_insts, idx)
            # read them
            insts = [ss]
            if len(fds) > 1:
                for ffd in fds[1:]:
                    line = ffd.readline()
                    utils.zforce(utils.zcheck_ff, line, lambda x: len(x)>0, "EOF (unmatched) %s" % ffd, "warn")
                    insts.append(line)
            # split and lookup (this one with no factors)
            words = [[x for x in one.strip().split()] for one in insts]
            idxes = [Vocab.w2i(vv, ws, add_eos=True, use_factor=False) for vv, ws in zip(self.vocabs, words)]
            yield TextInstance(words, idxes)
        # close
        # check unmatches
        if len(fds) > 1:
            for ffd in fds[1:]:
                line = ffd.readline()
                utils.zforce(utils.zcheck_ff, line, lambda x: len(x)==0, "EOF (unmatched) %s" % ffd, "warn")
        for ffd in fds:
            ffd.close()

# one call for convenience
def get_arranger(files, vocabs, shuffling_corpus, shuflling_buckets, sort_prior, batch_size, maxibatch_size, max_len, min_len, one_len):
    streamer = TextFileReader(files, vocabs, shuffling_corpus)
    tracking_order = True if maxibatch_size<=0 else False   # todo(warn): -1 for dev/test
    arranger = BatchArranger(streamer=streamer, batch_size=batch_size, maxibatch_size=maxibatch_size, outliers=[TextInstanceRangeOutlier(min_len, max_len)], single_outlier=TextInstanceRangeOutlier(min_len, one_len), sorting_keyer=TextInstanceLengthSorter(sort_prior), tracking_order=tracking_order,shuffling=shuflling_buckets)
    return arranger
