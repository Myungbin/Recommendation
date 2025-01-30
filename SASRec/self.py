import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import args
from model import SASRec


def data_partition():

    columns = ["userid", "movieid", "rating", "timestamp"]
    data = pd.read_table("./data/ml-1m/ratings.dat", names=columns, sep="::", encoding="latin1", engine="python")
    raw_data = data[["userid", "movieid"]]

    user_num = raw_data["userid"].max()
    item_num = raw_data["movieid"].max()
    user_item_dict = data.groupby("userid")["movieid"].agg(list).to_dict()

    user_train, user_valid, user_test = {}, {}, {}

    for user in user_item_dict:
        feedback_num = len(user_item_dict[user])
        if feedback_num < 5:
            user_train[user] = user_item_dict[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = user_item_dict[user][:-2]
            user_valid[user] = []
            user_valid[user].append(user_item_dict[user][-2])
            user_test[user] = []
            user_test[user].append(user_item_dict[user][-1])

    return [user_train, user_valid, user_test, user_num, item_num]


dataset = data_partition()
user_train, user_valid, user_test, usernum, itemnum = dataset


class SeqRecDataset(Dataset):
    def __init__(self, user_train, user_valid, user_test, usernum, itemnum):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum

    def __len__(self):
        return self.usernum

    def __getitem__(self, idx):
        user = idx + 1
        train = self.user_train[user]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        pos = np.zeros([args.maxlen], dtype=np.int32)
        neg = np.zeros([args.maxlen], dtype=np.int32)
        next_item = train[-1]
        index = args.maxlen - 1

        user_interacted_items = set(train)
        for i in reversed(train[:-1]):
            seq[index] = i
            pos[index] = next_item
            if next_item != 0:
                neg[index] = self.negative_sampling(user_interacted_items, 1)[0]
            next_item = i
            index -= 1
            if index == -1:
                break

        return (torch.LongTensor([user]), torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(neg))

    # def collate_fn(self, batch):
    #     input_sequence = []
    #     positive_sequence = []
    #     negative_sequence = []

    #     for sequence in batch:
    #         if len(sequence) >= args.maxlen:
    #             sequence = sequence[-args.maxlen :]
    #         else:
    #             sequence = [0] * (args.maxlen - len(sequence)) + sequence

    #     return (
    #         torch.LongTensor(input_sequence),
    #         torch.LongTensor(positive_sequence),
    #         torch.LongTensor(negative_sequence),
    #     )

    def negative_sampling(self, sequence, n_negative):
        negative_samples = []
        for _ in range(n_negative):
            negative = np.random.randint(1, self.itemnum + 1)
            while negative in sequence:
                negative = np.random.randint(1, self.itemnum + 1)
            negative_samples.append(negative)
        return negative_samples


dataset = SeqRecDataset(user_train, user_valid, user_test, usernum, itemnum)
dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

model = SASRec(usernum, itemnum, args)
for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass


model = SASRec(usernum, itemnum, args)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model.to(args.device)
model.train()


for epoch in range(args.num_epochs):
    total_loss = 0
    for batch_idx, (user, sequence, positive, negative) in enumerate(tqdm(dataLoader)):
        sequence = sequence.to(args.device)
        positive = positive.to(args.device)
        negative = negative.to(args.device)

        pos_score, neg_score = model(user, sequence, positive, negative)

        pos_labels = torch.ones_like(pos_score)  # Positive labels (1)
        neg_labels = torch.zeros_like(neg_score)  # Negative labels (0)

        loss = criterion(pos_score, pos_labels) + criterion(neg_score, neg_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() / len(dataLoader)

    print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss:.4f}")
