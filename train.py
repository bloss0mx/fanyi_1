import sys
from word2vec import train_ebd
from torch.utils.tensorboard import SummaryWriter
import math
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
from tools import loadRcdsFromFile, getCtr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


train_file_path = './data/train_ebd.csv'
# train_file_path = './data/zdealtags_ctr.train.csv'
savex = './em_x.pkl'

learning_rate = 1e-3
weight_decay = 1e-8
epoch_num = 5
batch_size = 100

ctr = getCtr(loadRcdsFromFile(train_file_path))


def train(embedding_replaced_word, ctr):

    if len(embedding_replaced_word) != len(ctr):
        print('error lenth', len(embedding_replaced_word), len(ctr))
        return 1

    class BaseDnnModel(nn.Module):
        def __init__(self):
            super(BaseDnnModel, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(len(embedding_replaced_word[0]), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        # def save_modal(self):

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    loss_fn = nn.L1Loss()
    model = BaseDnnModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    use_cuda = True
    test_num = math.floor(len(embedding_replaced_word) * 0.05)
    writer = SummaryWriter()

    class TrainData(Dataset):
        def __init__(self, prepared_data):
            self.data = prepared_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    prepared_data = []
    for idx in range(len(embedding_replaced_word)):
        X = torch.tensor(
            embedding_replaced_word[idx].numpy(), dtype=torch.float32)
        y = torch.tensor([ctr[idx]])
        prepared_data.append((X, y))

    test_data = prepared_data[:test_num]
    prepared_data = prepared_data[test_num:]
    valid_data = prepared_data[:test_num]
    train_data = prepared_data[test_num:]

    if torch.cuda.is_available() and use_cuda:
        train_data = list(
            map(lambda x: (x[0].cuda(), x[1].cuda()), train_data))
        test_data = list(map(lambda x: (x[0].cuda(), x[1].cuda()), test_data))
        valid_data = list(
            map(lambda x: (x[0].cuda(), x[1].cuda()), valid_data))
        model.to('cuda')

    train_loader = DataLoader(TrainData(train_data),
                              batch_size=batch_size, shuffle=True)

    start_time = time.time()
    losses = []
    for epoch in range(epoch_num):
        total_loss = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            pred = model.forward(X)
            loss = loss_fn(pred, y)
            writer.add_scalar("Loss/train", loss, i)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss / len(X))

    end_time = time.time()

    test(test_data, model, loss_fn, test_num, losses, True)
    torch.save(model.state_dict(), 'ctr_pred.pt')
    writer.flush()


def test(test_data, model, loss_fn, test_num, losses, print_table=False):
    test_table = []
    limit = 0.035
    loss_deadline = 1.0e-04
    correct_num = 0
    less_than_deadline_num = 0
    start_time = time.time()
    for idx in tqdm(range(len(test_data))):
        x, y = test_data[idx]
        pred = model.forward(x)
        loss = loss_fn(pred, y)
        less_than_deadline = '+' if loss < loss_deadline else ' '
        test_table.append([idx + 1, pred.item(), y.item(),
                          loss.item(), less_than_deadline])
        if (pred < limit and y < limit) or (pred > limit and y > limit):
            correct_num += 1
        if loss < loss_deadline:
            less_than_deadline_num += 1
    end_time = time.time()

    print('  >>>  correct rate:' if print_table ==
          True else 'correct rate:', str(correct_num / test_num),
          'total:', test_num, end=' ' if print_table == True else '\n'
          )


if __name__ == "__main__":
    learning_rate = float(sys.argv[1])
    weight_decay = float(sys.argv[2])
    epoch_num = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    print('lr:', learning_rate,
          'reg:', weight_decay,
          'epoch:', epoch_num,
          'batch:', batch_size,
          end=' '
          )
    #
    start_time = time.time()
    embedding_replaced_word = train_ebd()
    train(embedding_replaced_word, list(map(lambda x: float(x), ctr)))
    end_time = time.time()
    print('totally uses', int(end_time - start_time), 's')
