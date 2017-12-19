#!/bin/python3

# cmd: python extract_n.py <numbers> <files>
import sys

def ee(nums, files):
    print("Extracting %s from %s." % (nums, files))
    ss = []
    for f in files:
        with open(f) as fd:
            str_all = fd.read()
            items = str_all.split("\n\n")
            if len(items) <= 1:
                items = str_all.split("\n")
            ss.append(items)
    for num in nums:
        print("Extracting for number %s" % num)
        for s in ss:
            print(s[num])
        print("\n")

def main():
    nums = []
    files = []
    for one in sys.argv[1:]:
        try:
            x = int(one)
            nums.append(x)
        except:
            files.append(one)
    ee(nums, files)

if __name__ == '__main__':
    main()
