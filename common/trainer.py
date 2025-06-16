from re import I
import numpy as np
import matplotlib.pyplot as plt
import common.utils as clip_grads
from common.optimizer import *
import dask.array as da
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        self.verbose = False

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        for epoch in range(max_epoch):
            # 打乱数据
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                   # 确保输入数据格式正确
                if not isinstance(batch_x, np.ndarray):
                    #batch_x = np.array(batch_x)
                    batch_x=convert_to_numpy(batch_x)
                if not isinstance(batch_t, np.ndarray):
                    batch_t=convert_to_numpy(batch_t)
                   # batch_t = np.array(batch_t)

                # 计算梯度
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 消除公共层的重复计算
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 评估模型
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    if self.verbose: print('| epoch %d |  iter %d / %d | loss %.2f'
                                           % (self.current_epoch + 1, iters + 1, max_iters, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()

def convert_to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif hasattr(array, "get"):       # CuPy
        return array.get()
    elif hasattr(array, "compute"):   # Dask
        return array.compute()
    elif hasattr(array, "numpy"):     # PyTorch/TensorFlow
        return array.numpy()
    else:
        return np.array(array)        # 其他情况尝试转换



def remove_duplicate(params, grads):
    """
    把相同的权重从params和grads中删除。
    注意，这里的params和grads是列表形式，它们的长度相同。
    此函数假定它们是成对出现的，并且按顺序一一对应。
    例：
    params = [W1, W2, W3, W2]   
    grads = [dW1, dW2, dW3, dW2]
    调用此函数后，结果将是：
    params = [W1, W2, W3]
    grads = [dW1, dW2, dW3] 
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 寻找相同的权重
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 把梯度加起来
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 翻转权重
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads