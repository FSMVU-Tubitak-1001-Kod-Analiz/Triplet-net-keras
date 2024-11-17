#!/usr/bin/env python
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

from torch import optim
import copy
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.utils.data as dt
from code_utils.utils import save_annotation, get_now
import torch
import pandas as pd
import json

__all__ = [
    "create_triplet",
    "load_embeddings"
]


def _normalize(X):
    return X / np.linalg.norm(X, ord=2, axis=1)[:, None]


def load_embeddings(embeds_path, train_label_path, test_label_path, test_embeds_path=None, normalize=True):
    train_labels = []
    with open(train_label_path, "r") as label:
        for line in label.readlines():
            train_labels.append(json.loads(line))

    test_labels = []
    with open(test_label_path, "r") as label:
        for line in label.readlines():
            test_labels.append(json.loads(line))

    train_df = pd.DataFrame(train_labels)
    y_train = train_df.smellKey

    test_df = pd.DataFrame(test_labels)
    y_test = test_df.smellKey

    if test_embeds_path is None:  # only one embedding file
        """Index relation is very complicated. Basically something like this will return true:
       X_train[decode(97733277265)[0]] == X_total[train_df.loc[decode(97733277265)[0]]["index"]]
       If I try to explain this, what I would say is we have 3 main characters: X_train for train_embeds, X_total for
       all_embeds and train_df to keep track of label connections. In the train_df there is column called index that
       connects that row to the embedding file. So for example in order to get the tensor value of the first sample
       in train_df you would get the index value of that sample and look at that index value in the embeddings file.
       It is a very fragile building that is extremely easy to collapse so I'm not really fond of it. What I should
       have done was to create separate embedding files for test and train samples but me be lazy.
        """
        X_total = np.load(embeds_path)
        X_total = X_total.reshape(-1, np.prod(X_total.shape[1:]))
        X_total = _normalize(X_total)

        X_train = X_total[train_df.loc[:, "index"]]
        X_test = X_total[test_df.loc[:, "index"]]

    else:
        X_train = np.load(embeds_path)
        X_test = np.load(test_embeds_path)

        X_train = X_train.reshape(-1, np.prod(X_train.shape[1:]))

        X_test = X_test.reshape(-1, np.prod(X_test.shape[1:]))

        if normalize:
            X_train = _normalize(X_train)
            X_test = _normalize(X_test)

    return X_train, X_test, y_train, y_test


def create_triplet(embeds_path, train_label_path, test_label_path, save_path, train_batch_size, epoch, test_embeds_path=None):

    device = torch.device("cuda")
    save_annotation(Path(save_path).stem, "Starting triplet training for " + embeds_path)

    X_train, X_test, y_train, y_test = load_embeddings(embeds_path, train_label_path, test_label_path, test_embeds_path)

    key_size = len(y_train) + len(y_test)

    def decode(coded):
        coded = int(coded)
        n = coded % key_size
        p = (coded // key_size) % key_size
        a = (coded // key_size // key_size) % key_size
        return a, p, n

    def encode(a, p, n):
        assert np.all(a < key_size) and np.all(p < key_size) and np.all(n < key_size)
        return (a * key_size * key_size) + (p * key_size) + n

    X_, y_ = X_train, y_train
    data_xy = tuple([X_, y_])

    ind_list = []

    for data_class in sorted(set(data_xy[1])):
        """
        The triplet selection process is a crucial part of the model’s success, but it is not exhaustively
        explored. The authors should provide a deeper discussion of how triplets were generated,
        especially for negative samples.
        """
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]

        a = np.random.choice(same_class_idx, int(1.5e6))
        p = np.random.choice(same_class_idx, int(1.5e6))
        n = np.random.choice(diff_class_idx, int(1.5e6))

        ind_list.append(encode(a, p, n))

    X_train_indices = np.array(list(set(np.concatenate(ind_list))))

    class TripletDataset(dt.Dataset):
        def __init__(self, X_, triplet_indices, device):
            self.X_tensor = torch.from_numpy(X_).to(device).float()
            self.triplet_indices = triplet_indices
            self.device = device

        def __len__(self):
            return len(self.triplet_indices)

        def __getitem__(self, index):
            a, p, n = decode(self.triplet_indices[index])
            return self.X_tensor[a], self.X_tensor[p], self.X_tensor[n]

    class BaseNetwork(nn.Module):
        def __init__(self, input_size):
            super(BaseNetwork, self).__init__()
            self.linear1 = nn.Linear(input_size, 1000)
            self.linear2 = nn.Linear(1000, 500)
            self.linear3 = nn.Linear(500, input_size)
            self.actRelu = nn.LeakyReLU()
            self.drop = nn.Dropout(0.15)

        def forward(self, x):
            out = self.linear1(x)
            out = self.actRelu(out)
            out = self.drop(out)
            out = self.linear2(out)
            out = self.actRelu(out)
            out = self.drop(out)
            out = self.linear3(out)
            return out

    class TripletArchitecture(nn.Module):
        def __init__(self, input_size):
            super(TripletArchitecture, self).__init__()
            self.bn = BaseNetwork(input_size)

        def forward(self, a, p, n):
            a_out = self.bn(a)
            p_out = self.bn(p)
            n_out = self.bn(n)
            return a_out, p_out, n_out

    triplet_model = TripletArchitecture(X_train.shape[1]).to(device)

    triplet_optim = optim.Adam(triplet_model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    triplet_criterion = nn.TripletMarginLoss()

    triplet_dataset = TripletDataset(X_train, X_train_indices, device)

    triplet_loader = dt.DataLoader(triplet_dataset, shuffle=True, batch_size=train_batch_size)

    triplet_model.train()
    triplet_model.to(device)
    best_model = None
    best_loss = 1000
    default_patience = 2
    patience = default_patience

    history = []

    for i in range(epoch):
        data_iter = tqdm(
            enumerate(triplet_loader),
            desc="EP_%s:%d" % ("test", i),
            total=len(triplet_loader),
            bar_format="{l_bar}{r_bar}",
        )
        total_loss = 0

        for j, (a_, p_, n_) in data_iter:
            a_.require_grad = False
            p_.require_grad = False
            n_.require_grad = False

            a_o, p_o, n_o = triplet_model(a_, p_, n_)
            loss = triplet_criterion(a_o, p_o, n_o)
            loss.backward()
            total_loss += loss.item()
            data_iter.set_postfix({"Loss": total_loss / (j + 1)})
            triplet_optim.step()

        total_loss /= len(triplet_loader)
        print(progress := f'Epoch [{i + 1}/{100}], Loss: {total_loss:.4f}')
        history.append(progress)
        if total_loss < best_loss - 0.0001:
            best_loss = total_loss
            best_model = copy.deepcopy(triplet_model)
            patience = default_patience
        else:
            patience -= 1
        if patience == 0:
            break

    best_model.eval()

    X_test = torch.from_numpy(X_test).to(device).float()
    triplet_test_loader = dt.DataLoader(dt.TensorDataset(X_test), batch_size=256)

    output_embeds = []

    history.append("Starting embedding building at " + get_now())
    with torch.no_grad():
        for data_ in tqdm(triplet_test_loader, total=len(triplet_test_loader)):
            a, _, _ = best_model(data_[0], data_[0], data_[0])
            output_embeds.append(a.cpu().detach().numpy())
    history.append("Embedding build finished at " + get_now())
    embeds = np.vstack(output_embeds)

    np.save(save_path, embeds)
    save_annotation(save_path, f"Training for {embeds_path} complete at the timestamp given in the filename.\n" + "\n".join(history))