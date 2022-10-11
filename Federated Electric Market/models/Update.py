from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn


class LocalUpdate(object):
    def __init__(self, args, train_data, val_data):
        self.args = args
        self.train_data = train_data
        self.val_data = val_data

    def loacl_update(self, model):
        args = self.args
        train_data = self.train_data

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

        # 总训练误差
        tol_loss = []
        for epoch in range(args.local_epochs):
            train_loss = []
            for (seq, label) in train_data:
                seq, label = seq.to(args.device), label.to(args.device)
                model.zero_grad()
                y_pred = model(seq)
                loss = loss_function(y_pred, label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tol_loss.append(sum(train_loss)/len(train_loss))
            # 使用gamma调整学习率
            scheduler.step()

            # print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
            model.train()

        # 返回最好的模型参数,以及平均训练误差
        return model.state_dict(), sum(tol_loss) / len(tol_loss)
