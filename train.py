# import numpy as np
import pandas as pd
import seaborn as sns

# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def read_data():
    # read data
    df = pd.read_csv("./data/spine.csv")
    df.head()
    return df


def prep_data(df):
    # drop last column
    df = df.drop(df.columns[-1], axis=1)
    df.head()
    return df


def inspect_data(df):
    # class distribution
    sns.countplot(x="Class_att", data=df)


def map_target_labels(df):
    # NA
    # -https://stackoverflow.com/questions/67039036/changing-category-names-in-a-pandas-data-frame
    # df = df['Class_att'].replace(map_dict)

    # map target labels as numerical - 0 = normal, 1 = abnormal
    df["Class_att"] = df["Class_att"].astype("category")
    map_dict = {"Abnormal": 1, "Normal": 0}
    df["Class_att"].replace(map_dict, inplace=True)
    return df


def get_X_and_y(df):
    # create input and output data
    X = df.iloc[:, 0:-1]
    # NAA. Not sure why there is an extra Unnamed column.
    #   I had to manually drop last column.
    y = df.iloc[:, -1]
    return X, y


def get_train_test(X, y):
    # split data for train, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=69
    )


def standardize_input(X_train, X_test):
    # standardize input (i.e., mean = 0, std = 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_train, X_test


def dump_X_and_y(X_train, y_train):
    print(type(X_train))
    print(type(y_train))


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class BinaryClassification(nn.Module):
    # 2 layer FF DNN with BatchNorm and Dropout
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(12, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def get_model(device):
    LEARNING_RATE = 0.001
    # initialize optimizer, loss function
    model = BinaryClassification()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train(model, train_loader, device, optimizer, criterion):
    # model parameters
    EPOCHS = 50
    BATCH_SIZE = 64
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # NAA. Important !!! Zero gradients each epoch run
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    print(
        f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}"
    )


def get_predictions(model, test_loader, device):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    return y_pred_list


def main():
    df = read_data()
    df = prep_data(df)
    inspect_data(df)
    df = map_target_labels(df)

    X, y = get_X_and_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=69
    )
    X_train, X_test = standardize_input(X_train, X_test)
    dump_X_and_y(X_train, y_train)

    train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_data = TestData(torch.FloatTensor(X_test))

    BATCH_SIZE = 64
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = check_device()
    model, criterion, optimizer = get_model(device)
    train(model, train_loader, device, optimizer, criterion)

    y_pred_list = get_predictions(model, test_loader, device)
    confusion_matrix(y_test, y_pred_list)

    # dump precision, recall, F1
    print(classification_report(y_test, y_pred_list))


if __name__ == "__main__":
    main()
