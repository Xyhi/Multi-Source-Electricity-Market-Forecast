import pandas as pd

from utils.options import args
from model_train import train, load_data
from model_test import test
import matplotlib.pyplot as plt

if __name__ == '__main__':
    loss_df = pd.DataFrame(columns=['BiLSTM', 'CNN_LSTM', 'CNN_LSTM_2'])

    args = args()
    train_data, val_data, test_data, max_load, min_load = load_data(args, args.batch_size, '../source_data/data0.xlsx')
    for i in range(1, 4):
        loss_list = []
        for j in range(args.repeated):
            train(args, train_data, val_data, '../network/network_info.pkl', i)
            result = test(args, test_data, '../network/network_info.pkl', max_load, min_load, i)
            loss_list.append(result)
        loss_df.iloc[:, i-1] = loss_list

    plt.figure(figsize=(20, 16), dpi=200)
    print(loss_df)
    loss_df.boxplot()
    plt.show()



