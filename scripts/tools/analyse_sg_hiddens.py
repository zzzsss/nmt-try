#!/bin/python3

# extract hiddens from dumped sg and dimension reduction for them

import sys, pickle, json
from sklearn import manifold

def tsne(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    e = tsne.fit_transform(X)
    return e

printing = lambda x: print(x, flush=True)

def main():
    HID_NAME = "hidv"
    TSNE_NAME = "hidv_tsne"
    fname = sys.argv[1]
    batch_size = int(sys.argv[2])
    #
    with open(fname, "rb") as f:
        states_all = pickle.load(f)
    cur_idx = 0
    insts_num = len(states_all)
    printing("ALL: Processing %d insts with bsize %d." % (insts_num, batch_size))
    while cur_idx < insts_num:
        next_idx = min(insts_num, cur_idx+batch_size)
        cur_states = states_all[cur_idx:next_idx]
        states_count = sum(len(one[1]) for one in cur_states)
        printing("One: Processing %d/%d insts of %s states." % (next_idx, insts_num, states_count))
        full_x = []
        for states_set in cur_states:
            for s in states_set[1]:
                full_x.append(s[HID_NAME])
        tsne_x = tsne(full_x)
        # assign
        assign_idx = 0
        for states_set in cur_states:
            for s in states_set[1]:
                s[TSNE_NAME] = tsne_x[assign_idx]
                assign_idx += 1
        cur_idx = next_idx
    with open(fname+".tsne", "wb") as f:
        pickle.dump(states_all, f)
    with open(fname+".json", "w") as f:
        for states_set in states_all:
            states = []
            for s in states_set[1]:
                states.append({"path":s["path"], TSNE_NAME:s[TSNE_NAME].tolist(), "state":s["state"]})
            f.write(json.dumps((states_set[0], states))+"\n")

if __name__ == '__main__':
    main()

# python3 ../../znmt/scripts/tools/analyse_sg_hiddens.py ? 1
# for f in *.hid; do echo processing $f; python3 ../../znmt/scripts/tools/analyse_sg_hiddens.py $f 1; done
