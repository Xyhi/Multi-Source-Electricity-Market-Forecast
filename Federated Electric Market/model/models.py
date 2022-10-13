from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_directions = 2
        self.lstm = nn.LSTM(args.input_size, args.hidden_size, args.num_layers,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.num_directions * args.hidden_size, args.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.args.num_layers, batch_size, self.args.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.args.num_layers, batch_size, self.args.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]

        return pred


class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)
        self.num_directions = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.input_size, out_channels=args.output_size, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )
        self.lstm = nn.LSTM(input_size=args.output_size, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.num_directions * args.hidden_size, args.output_size)

    def forward(self, input_seq):
        input_seq = input_seq.permute(0, 2, 1)
        input_seq = self.conv(input_seq)
        input_seq = input_seq.permute(0, 2, 1)

        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.args.num_layers, batch_size, self.args.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.args.num_layers, batch_size, self.args.hidden_size).to(device)

        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.fc(output)
        pred = pred[:, -1, :]

        return pred


class CNN_LSTM_2(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM_2, self).__init__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)
        self.num_directions = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.input_size, out_channels=args.output_size, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )
        self.lstm1 = nn.LSTM(input_size=args.output_size, hidden_size=args.hidden_size,
                             num_layers=args.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=self.num_directions*args.hidden_size, hidden_size=args.hidden_size,
                             num_layers=args.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.num_directions * args.hidden_size, args.output_size)

    def forward(self, input_seq):
        input_seq = input_seq.permute(0, 2, 1)
        input_seq = self.conv(input_seq)
        input_seq = input_seq.permute(0, 2, 1)

        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.args.num_layers, batch_size, self.args.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.args.num_layers, batch_size, self.args.hidden_size).to(device)

        output, _ = self.lstm1(input_seq, (h_0, c_0))
        output, _ = self.lstm2(output)
        pred = self.fc(output)
        pred = pred[:, -1, :]

        return pred

