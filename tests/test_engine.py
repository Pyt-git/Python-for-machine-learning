# simple linear regression: y_cap = xW + b

np.random.seed(0)

x = Tensor(np.random.randn(4,3))
W = Tensor(np.random.randn(3,1))
b = Tensor(np.zeros((1,)))

y_true = Tensor(np.random.randn(4,1))

y_pred = x.matmul(W) + b 
loss = ((y_pred - y_true)**2).sum()

print("loss:", loss.data)

loss.backward()

print("dL/dW:", W.grad)
print("dL/db:", b.grad)
