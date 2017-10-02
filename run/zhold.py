import dynet as dy
import sys
i = 1
try:
    x = int(sys.argv[1])
    i = x
except:
    pass
ss = (256*i,1024,1024)
print("size is %s" % (ss,))
x = dy.zeros(ss)
y = x.value()
print("what is next\n")
input("1")
input("2")
input("3")

# to kill
# ps -elf | grep zhold.py | sed -r 's/[]+/ /g' | tee /proc/self/fd/2 | cut -f 4 -d ' ' | xargs kill -9
