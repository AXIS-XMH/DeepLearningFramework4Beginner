import meta
import numpy as np

A = meta.Square()
B = meta.Exp()
C = meta.Square()

x = meta.Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)