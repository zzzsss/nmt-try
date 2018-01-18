import sys, json

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except:
    HAS_PLOT = False

def read_from_json(fname, idx):
    TSNE_NAME = "hidv_tsne"
    with open(fname, "r") as f:
        thems = [json.loads(line) for line in f]
    print("Load for %d insts." % len(thems))
    inst = thems[idx]
    states = inst[1]
    print(inst[0]+"with %d states." % len(states))
    states.sort(key=lambda x:len(x["path"]))
    x, y = [one[TSNE_NAME][0] for one in states], [one[TSNE_NAME][1] for one in states]
    props = [s.state() for s in states]
    texts = [" ".join(s["path"]) for s in states]
    return x,y,props,texts

def read_from_txt(fname, idx):
    x,y,props,texts = [],[],[],[]
    with open(fname, "r") as f:
        # id, x, y, txt
        for line in f:
            try:
                thems = line.split()
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
    if idx<0:
        data = read_from_txt(fname, idx)
    else:
        data = read_from_json(fname, idx)
    show(data)

def show(data):
    # list of [a, b, props, texts]
    x,y,props,texts = data
    if HAS_PLOT:
        # plt.axis([0, 20, 0, 20])
        colors = ['b' if p!='GOLD' else 'r' for p in props]
        plt.scatter(x, y, c=colors)
    i = 0
    for a,b,p,t in zip(x,y,props,texts):
        print("%d %.3f %.3f %s %s" % (i, a, b, p, t))
        if HAS_PLOT:
            plt.annotate(str(i), (a,b))
        i += 1
    if HAS_PLOT:
        plt.show()

if __name__ == '__main__':
    main()

# python3 ../../znmt/scripts/tools/draw_sg_hiddens.py ? #idx
