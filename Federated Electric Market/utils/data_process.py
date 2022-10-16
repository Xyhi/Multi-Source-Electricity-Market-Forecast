import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载数据集
def load_data(path):
    data = pd.read_excel(path)
    return data


# 利用torch.Dataset构建数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# 搭建序列多输出数据集  num为预测数，应该与输出层个数相等
def seq_to_mul(seq_len, B, num, path):
    data = load_data(path)
    # 划分训练集,验证集,测试集
    # 由于是时序模型,因此划分时要注意不能shuffle,只能顺序构建

    data_len = data.shape[0]

    # 按照6:2:2方式划分训练集,验证集,测试集
    train_data = data[:int(data_len * 0.6)]
    val_data = data[int(data_len * 0.6):int(data_len * 0.8)]
    test_data = data[int(data_len * 0.8):data_len]

    # 先简单预测每天的平均载荷
    # 利用max_load,min_load进行最大最小标准化
    # -1为数据集中载荷的列标
    load_index = -1
    max_load = np.max(train_data[train_data.columns[load_index]])
    min_load = np.min(train_data[train_data.columns[load_index]])

    # 进行数据处理
    def process(dataset, batch_size, step, shuffle):
        # 对load进行标准化处理
        load = dataset[dataset.columns[load_index]]
        load = (load - min_load) / (max_load - min_load)
        load = load.tolist()

        dataset = dataset.values.tolist()
        seq = []

        # 构建多输出seq数据集,step为构建步长
        for i in range(0, len(dataset) - seq_len - num, step):
            # 训练seq数据
            train_seq = []
            # 训练监督标签,以预测后num个load数据作为监督学习标签
            train_label = []

            # 将seq序列以及相应的天气特征加入
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(1, 3):
                    x.append(dataset[j][c])
                train_seq.append(x)

            # 将预测数目num的未来序列作为监督学习标签
            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            # 将seq转成float,将label进行flatten
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # 将seq以及对应的label构造DataLoader
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    # 构建LSTM训练的训练集,验证集,测试集
    Train_data = process(train_data, B, step=1, shuffle=True)
    Val_data = process(val_data, B, step=1, shuffle=True)
    Test_data = process(test_data, B, step=num, shuffle=False)

    # 返回训练数据,验证数据,测试数据,最大最小载荷
    return Train_data, Val_data, Test_data, max_load, min_load


# 得到mape值,作为模型性能衡量指标
def get_mape(x, y):
    return np.mean(np.abs((x - y) / x))
