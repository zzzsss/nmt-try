import sys, json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_PLOT = True
except:
    HAS_PLOT = False

def read_from_json(fname, idx):
    TSNE_NAME = "hidv_tsne"
    with open(fname, "r") as f:
        thems = [json.loads(line) for line in f]
    print("#Load for %d insts." % len(thems))
    inst = thems[idx]
    states = inst[1]
    print("#with #%d states, str is:" % len(states) + inst[0])
    states.sort(key=lambda x:len(x["path"]))
    x, y = [one[TSNE_NAME][0] for one in states], [one[TSNE_NAME][1] for one in states]
    props = [s["state"] for s in states]
    texts = [" ".join(s["path"]) for s in states]
    return x,y,props,texts

# def read_from_txt(fname, idx, rangea, rangeb):
def read_from_txt(fname, idx, maxn):
    x,y,props,texts = [],[],[],[]
    with open(fname, "r", encoding="utf8") as f:
        # id, x, y, txt
        for line in f:
            try:
                thems = line.split()
                sent_len = len(thems[4:])
                # if sent_len < rangea or sent_len > rangeb:
                #     continue
                if len(x) >= maxn:
                    break
                x.append(float(thems[1]))
                y.append(float(thems[2]))
                props.append(thems[3])
                texts.append(" ".join(thems[4:]))
            except:
                pass
    return x,y,props,texts

def main():
    fname = sys.argv[1]
    idx = int(sys.argv[2])
    # range [a,b]
    if idx<0:
        # rangea, rangeb = int(sys.argv[3]), int(sys.argv[4])
        # data = read_from_txt(fname, idx, rangea, rangeb)
        data = read_from_txt(fname, idx, int(sys.argv[3]))
    else:
        data = read_from_json(fname, idx)
    show(data)

def show(data):
    # list of [a, b, props, texts]
    x,y,props,texts = data
    if HAS_PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        plt.axis([-15, 15, -10, 10])
        colors = ['b' if p!='GOLD' else 'r' for p in props]
        # plt.scatter(x, y, c=colors)
        plt.scatter(y, x, c=colors)
    i = 0
    for a,b,p,t in zip(x,y,props,texts):
        print("%d %.3f %.3f %s %s" % (i, a, b, p, t))
        # if HAS_PLOT:
        #     plt.annotate(str(i), (a,b))
        i += 1
    if HAS_PLOT:
        ax.add_patch(
            patches.Rectangle((-0.5,5.3), 1.6, 1.6, fill=False)
        )
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

# python3 ../../znmt/scripts/tools/draw_sg_hiddens.py ? #idx 1 1000
