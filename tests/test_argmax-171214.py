import sys
import dynet as dy
import numpy as np

# todo(wait to see)
whichx = int(sys.argv[1])
on_gpu = whichx

INPUT_SIZE = 500
BATCH_SIZE = 80
VOCAB_SIZE = 50000
STEP = 100

m = dy.Model()
trainer = dy.SimpleSGDTrainer(m)

pW = m.add_parameters((VOCAB_SIZE, INPUT_SIZE))
pb = m.add_parameters(VOCAB_SIZE)

W = dy.parameter(pW)
b = dy.parameter(pb)
input = dy.ones((INPUT_SIZE,), batch_size=BATCH_SIZE)
golds = [i for i in range(BATCH_SIZE)]

for s in range(STEP):
    output = dy.affine_transform([b, W, input])
    gold_expr = dy.pick_batch(output, golds)
    if on_gpu:
        max_expr = dy.max_dim(output)
    else:
        output_v = output.vec_value()
        max_idxs = [int(np.argmax(output_v[i*VOCAB_SIZE:(i+1)*VOCAB_SIZE])) for i in range(BATCH_SIZE)]
        max_expr = dy.pick_batch(output, max_idxs)
    one_loss = max_expr - gold_expr
    loss = dy.sum_batches(one_loss) / BATCH_SIZE
    loss.forward()
    loss.backward()

# ==================
import dynet as dy
import numpy as np
VOCAB_SIZE = 1000
BATCH_SIZE = 20
K = 5
v = [float(np.random.randint(0, 10000)) for i in range(VOCAB_SIZE*BATCH_SIZE)]
t = dy.inputVector(v)
t2 = dy.reshape(t, (VOCAB_SIZE,), BATCH_SIZE)
t2_t = t2.tensor_value()
p = t2_t.max_and_argmax(0, K)

for i in range(BATCH_SIZE):
    print(np.argsort(v[i*VOCAB_SIZE:(i+1)*VOCAB_SIZE])[-K:])

# =================
# test count_larger
import dynet as dy
import numpy as np
VOCAB_SIZE = 1000
BATCH_SIZE = 20
x = [float(np.random.randint(0, 10000)) for i in range(VOCAB_SIZE*BATCH_SIZE)]
y = [float(np.random.randint(0, 10000)) for i in range(BATCH_SIZE)]
tx = dy.inputVector(x)
tx2 = dy.reshape(tx, (VOCAB_SIZE,), BATCH_SIZE)
tx2_t = tx2.tensor_value()
ty = dy.inputVector(y)
ty2 = dy.reshape(ty, (1,), BATCH_SIZE)
ty2_t = ty2.tensor_value()

p = tx2_t.count_larger(ty2_t)

pp = []
for i in range(BATCH_SIZE):
    c = 0
    for j in range(VOCAB_SIZE):
        if x[i*VOCAB_SIZE+j] > y[i]:
            c += 1
    pp.append(c)

print(p)
print(pp)
