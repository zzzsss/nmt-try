import sys
import matplotlib.pyplot as plt

def draw(X, Y):
    assert len(X)==len(Y)
    for x,y in zip(X, Y):
        print("%s %s" % (x,y))
    plt.scatter(X, Y)
    plt.show()

def read():
    X = []
    Y = []
    temp = [-1, []]   # [max-shards, [(ns, time)]]
    #
    def _fin():
        if temp[0]>0:
            best_idx = min(temp[1], key=lambda x: x[1])[0]
            X.append(temp[0])
            Y.append(best_idx)
    #
    for line in sys.stdin:
        if "impl_tf" in line and "shard" in line:
            _fin()
            ms = int(line.split()[-1])
            temp = [ms, []]
        elif line.startswith("topk-tf"):
            fs = line.split()
            ns = int(fs[0].split('-')[-1])
            tt = float(fs[-3])
            temp[1].append((ns, tt))
    _fin()
    return X, Y

def read2():
    X, Y = [], []
    for line in sys.stdin:
        fs = line.split()
        x, y = float(fs[0]), float(fs[1])
        X.append(x)
        Y.append(y)
    return X, Y

def main():
    X,Y = read()
    draw(X,Y)

main()
