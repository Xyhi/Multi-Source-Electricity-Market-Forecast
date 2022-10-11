import pandas as pd
from utils.options import args
from model_train import train, load_data
from model_test import test
import matplotlib.pyplot as plt

# 通过model_select选择了BiLSTM作为最后的预测模型,现在对所有的client进行训练

if __name__ == '__main__':
    num_client = 50
    file_list = ['../source_data/data'] * num_client
    name_list = ['block'] * num_client
    for i in range(len(file_list)):
        name_list[i] += str(i)
        file_list[i] += str(i) + '.xlsx'
    loss_df = pd.DataFrame(columns=name_list)

    args = args()
    for i in range(len(name_list)):
        train_data, val_data, test_data, max_load, min_load = load_data(args, args.batch_size, file_list[i])
        loss_list = []
        for j in range(args.repeated):
            train(args, train_data, val_data, '../network/network_info.pkl', 1)
            result = test(args, test_data, '../network/network_info.pkl', max_load, min_load, 1)
            loss_list.append(result)
        loss_df.iloc[:, i] = loss_list

    print(loss_df)
    loss_df.boxplot()
    plt.show()



