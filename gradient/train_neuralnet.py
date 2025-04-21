# coding: utf-8
import pickle
from re import T
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# 获取当前脚本的父目录的父目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))  # 将根目录加入 Python 路径
from gradient import two_layer_net
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))  
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    file_path = Path(__file__).parent / 'mnist.pkl'
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
network = two_layer_net.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)
iters_num = 10000  
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_acc_list = []
test_acc_list = []
train_loss_list = []
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 参数更新
    network.params_update_with_gradient(x_batch, t_batch,learning_rate)
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        acc_test = network.accuracy(x_test,t_test)
        acc_train = network.accuracy(x_train,t_train)
        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)
        print("train acc, test acc | " + str(acc_train) + ", " + str(acc_test))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
    