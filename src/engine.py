import numpy as np

class Tensor: 
  def __init__(self, data, _children=(), _op=''): 
      self.data = np.array(data,dtype=float)
      self.grad = np.zeros_like(self.data)

      # Graph components
      self.backward = lambda: None
      self.prev = set(_children)
      self._op = _op

  def __repr__(self):
      return f"Tensor(data={self.data}, grad={self.grad})"

  # ----- operations ------ #

  def __add__(self, other): 
      other = other if isinstance(other, Tensor) else Tensor(other)
      out = Tensor(self.data + other.data, (self, other), '+')

      def _backward():
          self.grad += other.grad
          other.grad += out.grad
      out._backwoard = _backward
      return out

  def __radd__(self, other): 
      return self + other
  
  def __mul__(self, other): 
      other = other if isinstance(other, Tensor) else Tensor(other)
      out = Tensor(self.data * other.data, (self, other), "*")

      def _backward(): 
          self.grad += other.data * out.grad
          other.grad += self.grad * out.grad
      out._backward = _backward
      return out

  def __rmul__(self, other): 
      return self * other

  def __neg__(self): 
      out = Tensor(-self.data, (self,), 'neg')

      def _backward(): 
          self -= out.grad
      out._backward = _backward
      return out

 def __sub__(self, other): 
     return self + (-other)

 def __rsub__(self, other): 
     return other + (-self)

 def __truediv__(self, other): 
     other = other if isinstance(other, Tensor) else Tensor(other)
     return self * other ** -1

 def __power__(self, other): 
     assert isinstance(power, (int, float))
     out = Tensor(self.data ** power, (self,), f'**{power}')

     def _backward(): 
         self.grad += power * (self.data ** (power - 1)) * out.grad
     out._backward = _backward
     return out

# ----- Useful functions ----- #

def relu(self): 
    out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

    def _backward():
        self.grad += (self.data > 0) * out.grad
    out._backward = _backward
    return out

def sum(self): 
    out = Tensor(self.data.sum(), self,), 'sum')

    def _backward(): 
        self.grad += np.ones_like(self.data) * out.grad
    out._backward = _backward
    return out

def matmul(self, other): 
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ other.data, (self, other), 'matmul')

    def _backward(): 
        # dL/dA = dL/dz @ B^T
        # dL/dB = A^T @ dL/dZ
      
        self.grad += out.grad @ other.grad.T
        otehr.grad += self.data.T @ out.grad
    out._backward = _backward
    return out

# ----- backprop ----- #

def backward(self): 
    # topological sort
    topo = [] 
    visited = set()

    def build(v): 
        if v not in visited: 
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(self)

    # seed gradient at loss
    self.grad = np.ones_like(self.data)

    # go backwards
    for v in reversed(topo): 
        v._backward()
  
