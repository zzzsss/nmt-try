import dynet as dy

x = dy.Model()
# f = open("_t.cpp")
# rss = f.readline().split()
# f.close()
with open("/proc/self/statm") as f:
    rss = int(f.readline().split()[1])
print(rss)
