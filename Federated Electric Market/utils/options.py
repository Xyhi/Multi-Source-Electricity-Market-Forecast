import argparse
import torch


# 多输出LSTM
def args():
    parser = argparse.ArgumentParser()
    # model selection and train
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=3, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--repeated', type=int, default=10, help='the repeated num of training')

    # federated params
    parser.add_argument('--local_epochs', type=int, default=10, help='the epochs of local client training')
    parser.add_argument('--local_bs', type=int, default=10, help='the batch size of local client training')
    parser.add_argument('--frac', type=float, default=0.01, help='the ratio of selected clients per epoch')
    parser.add_argument('--num_users', type=float, default=112, help='the total num of clients')
    parser.add_argument('--all_clients', type=bool, default=False, help='aggregation over all clients')
    parser.add_argument('--tol_lr', type=float, default=0.01, help='learning rate of server training')
    parser.add_argument('--tol_epochs', type=int, default=10, help='rounds of total training')
    parser.add_argument('--local_repeated', type=int, default=1, help='indicates the local clients update')
    parser.add_argument('--contrast', type=bool, default=True, help='indicates whether to use local epochs')
    parser.add_argument('--show_result', type=bool, default=True, help='indicates whether to show the result')
    parser.add_argument('--round', type=int, default=8, help='The number of digits retained')

    args = parser.parse_args()

    return args

