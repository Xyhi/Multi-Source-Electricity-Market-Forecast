import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from utils.data_process import seq_to_mul
from model.models import BiLSTM, CNN_LSTM, CNN_LSTM_2
import copy


# 构建多输入多输出的模型序列
def load_data(args, batch_size, filepath):
    train_data, val_data, test_data, max_load, min_load = seq_to_mul(seq_len=args.seq_len,
                                                                     B=batch_size,
                                                                     num=args.output_size,
                                                                     path=filepath)
    return train_data, val_data, test_data, max_load, min_load


# 获取验证集的loss
def get_val_loss(args, model, Val):
    # 进行模型评估
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args, train_data, val_data, path, flag):
    # 获取构建的model
    if flag == 1:
        model = BiLSTM(args).to(args.device)
    elif flag == 2:
        model = CNN_LSTM(args).to(args.device)
    elif flag == 3:
        model = CNN_LSTM_2(args).to(args.device)

    # 定义损失函数MSE
    loss_function = nn.MSELoss().to(args.device)
    # 定义优化器
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)

    # 使用StepLR作为learning rate的学习
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 定义模型训练超参数
    min_epochs = 2
    best_model = None
    min_val_loss = 5

    if args.contrast:
        num = args.local_epochs
    else:
        num = args.epochs

    for epoch in tqdm(range(num)):
        train_loss = []
        for (seq, label) in train_data:
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 使用gamma调整学习率
        scheduler.step()
        # 进行模型验证,获取验证集误差
        val_loss = get_val_loss(args, model, val_data)
        # 当epoch达到最小epoch,且两轮迭代之后损失函数的值小于阈值,则停止训练
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict()}
    torch.save(state, path)
