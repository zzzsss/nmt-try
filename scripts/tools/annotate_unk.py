#!/bin/python3

# annotate the texts with <unk> explicitly noted
import sys, json

def main():
    d = json.loads(open(sys.argv[1]).read())
    for line in sys.stdin:
        fs = line.split()
        for i in range(len(fs)):
            w = fs[i]
            if w not in d:
                fs[i] += "<unk>"
        print(" ".join(fs))

if __name__ == '__main__':
    main()
