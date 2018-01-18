#!/bin/python

# calculate TER based on tercom

import os, sys

def main():
    #
    TER_HOME = "%s/mt/tools/tercom-0.7.25" % os.environ["ZZ"]
    TER_TMP_HYP = "%s/tmp/hyp.txt" % TER_HOME
    TER_TMP_REF = "%s/tmp/ref.txt" % TER_HOME
    #
    origin_hyp = sys.argv[1]
    origin_refs = sys.argv[2:]
    #
    with open(TER_TMP_HYP, "w") as outf:
        with open(origin_hyp) as inf:
            for i, s in enumerate(inf):
                outf.write(s.strip() + "(ID%d)"%i + "\n")
    with open(TER_TMP_REF, "w") as outf:
        for one_ref in origin_refs:
            with open(one_ref) as inf:
                for i, s in enumerate(inf):
                    outf.write(s.strip() + "(ID%d)"%i + "\n")
    #
    os.system("java -jar %s/tercom.7.25.jar -h %s -r %s" % (TER_HOME, TER_TMP_HYP, TER_TMP_REF))

if __name__ == '__main__':
    main()
