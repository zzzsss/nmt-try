from dynet import *
import sys

if sys.argv[1] == "die":
    die = True
else:
    die = False
print("Die?: %s" % die)

# simple forever loop for the memory checking
m = Model()
px = m.add_parameters((10,10))
if die:
    x = parameter(px)
    y = x*x
    while True:
        cg_checkpoint()
        for i in range(10000):
            z = y+y
        print(z.value())
        cg_revert()
else:
    while True:
        renew_cg()
        x = parameter(px)
        y = x*x
        for i in range(10000):
            z = y+y
        print(z.value())
