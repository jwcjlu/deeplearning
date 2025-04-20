import numpy as np
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
       self.params = {}
       self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
       self.params['b1'] = np.zeros(hidden_size)
       self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
       self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
      W1, W2 = self.params['W1'], self.params['W2']
      b1, b2 = self.params['b1'], self.params['b2']
      a1 = np.dot(x, W1) + b1
      z1 = sigmoid(a1)
      a2 = np.dot(z1, W2) + b2
      y = softmax(a2)
      return  y
  

    def loss(self, x, t):
      y = self.predict(x)
      return cross_entropy_error(y, t)

    def params_update(self,x,t,rate=0.1):
        params = self.params
        loss = lambda W: self.loss(x, t)  
        params['W1']-=numerical_gradient(loss,params['W1'])*rate
        params['b1']-=numerical_gradient(loss,params['b1'])*rate
        params['W2']-=numerical_gradient(loss,params['W2'])*rate
        params['b2']-=numerical_gradient(loss,params['b2'])*rate

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
 ## 梯度计算实现   
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
      idx = it.multi_index
      tmp_val = x[idx]
      x[idx] = float(tmp_val) + h
      fxh1 = f(x)  # f(x+h)
      x[idx] = tmp_val - h
      fxh2 = f(x)  # f(x-h)
      grad[idx] = (fxh1 - fxh2) / (2 * h)
      x[idx] = tmp_val  # 还原值
      it.iternext()
    return grad

##激活函数softmax实现
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

##输出函数softmax实现
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
## 交叉熵误差
def cross_entropy_error(y, t):
      if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
      if t.size == y.size:
        t = t.argmax(axis=1)
      batch_size = y.shape[0]
      return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size