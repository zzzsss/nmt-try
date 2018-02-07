# split and eval

import sys, os
with open(sys.argv[1]) as fd:
    lines = [l for l in fd]

TMP_NAME = "tmp.txt"
CMD = "perl ../../znmt/scripts/multi-bleu.perl ../../en_de_data_z5/%(dataname)s.tok.de <tmp.txt"

start = 0
for i, name in [(3003, "data2014"), (2169, "data2015"), (2999, "data2016")]:
    with open(TMP_NAME, "w") as fd:
        for l in lines[start:start+i]:
            fd.write(l)
    print(name, i)
    os.system(CMD % {"dataname": name})
    start += i
