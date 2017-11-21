# to output the results
from zl.search import State, SearchGraph
from zl import data, utils
import numpy as np

class Outputter(object):
    def __init__(self, opts):
        self.opts = opts
        self.unk_rep = opts["decode_replace_unk"]
        self.count = 0

    def transform_src(self, w):
        if all(str.isalnum(z) or z=="." for z in w):
            return w
        else:
            return "<unkown>"

    def format(self, states, target_dict, kbest, verbose):
        ret = ""
        if not kbest:
            states = [states[0]]
        if verbose:
            ret += "# znumber%s" % self.count + "\n"
            # ret += states[0].sg.show_graph(target_dict, False) + "\n"
        self.count += 1
        for s in states:
            # list of states including eos
            paths = s.get_path()
            utils.zcheck(paths[-1].action_code==target_dict.eos, "Not ended with EOS!")
            for one in paths[:-1]:
                tmp_code = one.action_code
                unk_replace = False
                if not self.unk_rep or tmp_code != target_dict.unk:
                    ret += target_dict.getw(one.action_code)
                else:
                    # todo: replacing unk with bilingual dictionary
                    unk_replace = True
                    xwords = one.get("attention_src")
                    xidx = np.argmax(one.get("attention_weights"))
                    if xidx >= len(xwords):
                        utils.zcheck(False, "attention out of range", func="warn")
                        rrw = "<out-of-range>"
                    else:
                        rrw = xwords[xidx]
                    ret += self.transform_src(rrw)
                if verbose:
                    if unk_replace:
                        ret += "<UNK>"
                    ret += ("(%.3f)" % one.action_score())
                ret += " "
            # last one
            if verbose:
                one = paths[-1]
                ret += target_dict.getw(one.action_code)
                ret += ("(%.3f)" % one.action_score())
                ret += " -> (OVERALL:%.3f,%.3f,%d)" % (one.score_partial, one.score_final, one.length)
            ret += "\n"
        if kbest:
            ret += "\n"
        return ret
