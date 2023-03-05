import math
from itertools import chain
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# from model_avg import build_dir
from utils.data_process import device, get_mape
from model.models import BiLSTM
from scipy.interpolate import make_interp_spline
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import os

def build_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 如果不存在目录figure_save_path，则创建

def test(args, test_data, path, max_load, min_load,arg_name, arg_val,idx):
    print('user ' + str(idx) + ' is testing')
    pred = []
    y = []
    model = BiLSTM(args).to(device)

    # 加载相关的state_dict
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    for (seq, target) in test_data:
        target = list(chain.from_iterable(target.data.tolist()))    # chain.from_iterable 将多个迭代器逐个迭代出来
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    # 反归一化操作
    y = (max_load - min_load) * y + min_load
    pred = (max_load - min_load) * pred + min_load
    # print('mape:', get_mape(y, pred))
    if args.show_result:
        plot(y, pred,arg_name, arg_val,idx)
    # get_mape(y, pred),
    return [mse(y,pred),mae(y,pred),math.sqrt(mse(y,pred))]


def plot(y, pred,arg_name, arg_val,idx):
    # plot
    x = [i for i in range(1, len(y)+1)]

    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y)(x_smooth)   # 数据平滑处理（插值填充）
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred)(x_smooth)


    build_dir('./result_img/test_pred/' + arg_name)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(
        './result_img/test_pred/' + arg_name + '/' + arg_name + '=' + str(arg_val) + '-' + str(idx) + '.png')
    plt.show()
