import dynet as dy
import sys
i = 1
flag = False
try:
    x = int(sys.argv[1])
    i = x
    flag = True
except:
    pass
if flag:    # specific mode
    STEP = 10
    ss = (128*i//STEP,1024,)
    print("size is %s" % (ss,))
    while True:
        dy.renew_cg()
        x = dy.zeros(ss)
        for i in range(STEP):
            for j in range(1024):
                x += 2.0 * x
        y = x.value()
else:       # auto mode
    ss = (256*i,1024,1024)
    print("size is %s" % (ss,))
    x = dy.zeros(ss)
    y = x.value()
    print("what is next\n")
    input("1")
    input("2")
    input("3")
print("finish\n")

# to run specifically
# PYTHONPATH=$DY_ZROOT/gbuild/python python3 ../znmt/run/zhold.py 1 --dynet-mem 4 --dynet-devices GPU:?
# to kill
# ps -elf | grep zhold.py | sed -r 's/[]+/ /g' | tee /proc/self/fd/2 | cut -f 4 -d ' ' | xargs kill -9
