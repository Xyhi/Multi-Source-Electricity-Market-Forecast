from itertools import chain
import numpy as np
import torch
from tqdm import tqdm
from utils.data_process import device, get_mape
from model.models import BiLSTM, CNN_LSTM, CNN_LSTM_2


def test(args, test_data, path, max_load, min_load, flag):
    pred = []
    y = []
    print('loading model...')
    if flag == 1:
        model = BiLSTM(args).to(device)
    elif flag == 2:
        model = CNN_LSTM(args).to(device)
    elif flag == 3:
        model = CNN_LSTM_2(args).to(device)

    # 加载相关的state_dict
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(test_data):
        target = list(chain.from_iterable(target.data.tolist()))
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
    print('mape:', get_mape(y, pred))
    # plot(y, pred)

    return get_mape(y, pred)

