import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/nalbarr/pytorch-spine-binary-classifier/blob/master/pytorch_spine_binary_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # pytorch-spine-binary-classifier

    #### References
    - https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
    - https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset?select=Dataset_spine.csv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataset

    310 Observations, 13 Attributes (12 Numeric Predictors, 1 Binary Class Attribute - No Demographics)

    Lower back pain can be caused by a variety of problems with any parts of the complex, interconnected network of spinal muscles, nerves, bones, discs or tendons in the lumbar spine. Typical sources of low back pain include:

    The large nerve roots in the low back that go to the legs may be irritated
    The smaller nerves that supply the low back may be irritated
    The large paired lower back muscles (erector spinae) may be strained
    The bones, ligaments or joints may be damaged
    An intervertebral disc may be degenerating
    An irritation or problem with any of these structures can cause lower back pain and/or pain that radiates or is referred to other parts of the body. Many lower back problems also cause back muscle spasms, which don't sound like much but can cause severe pain and disability.

    While lower back pain is extremely common, the symptoms and severity of lower back pain vary greatly. A simple lower back muscle strain might be excruciating enough to necessitate an emergency room visit, while a degenerating disc might cause only mild, intermittent discomfort.

    This data set is about to identify a person is abnormal or normal using collected physical spine details/data.
    """)
    return


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    return (
        DataLoader,
        Dataset,
        StandardScaler,
        classification_report,
        confusion_matrix,
        nn,
        optim,
        pd,
        sns,
        torch,
        train_test_split,
    )


@app.cell
def _(subprocess):
    # ls
    subprocess.call(["ls"])
    return


@app.cell
def _(subprocess):
    #! ls ../data
    subprocess.call(["ls", "./data"])
    return


@app.cell
def _(pd):
    # read data
    # df = pd.read_csv("pytorch-spine-binary-classifier/spine.csv")
    df = pd.read_csv("./data/spine.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    # drop last column
    df_1 = df.drop(df.columns[-1], axis=1)
    df_1.head()
    return (df_1,)


@app.cell
def _(df_1, sns):
    # class distribution
    sns.countplot(x="Class_att", data=df_1)
    return


@app.cell
def _(df_1):
    # map target labels as numerical - 0 = normal, 1 = abnormal
    df_1["Class_att"] = df_1["Class_att"].astype("category")
    encode_map = {"Abnormal": 1, "Normal": 0}
    df_1["Class_att"].replace(encode_map, inplace=True)
    return


@app.cell
def _(df_1):
    df_1["Class_att"]
    return


@app.cell
def _(df_1):
    # create input and output data
    X = df_1.iloc[:, 0:-1]
    # NAA. Not sure why there is an extra Unnamed column.  I had to manually drop last column.
    y = df_1.iloc[:, -1]
    return X, y


@app.cell
def _(y):
    y
    return


@app.cell
def _(X, train_test_split, y):
    # split data for train, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=69
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler, X_test, X_train):
    # standardize input (i.e., mean = 0, std = 1)
    scaler = StandardScaler()
    X_train_1 = scaler.fit_transform(X_train)
    X_test_1 = scaler.fit_transform(X_test)
    return X_test_1, X_train_1


@app.cell
def _(X_train_1, y_train):
    print(type(X_train_1))
    print(type(y_train))
    return


@app.cell
def _():
    # model parameters
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    return BATCH_SIZE, EPOCHS, LEARNING_RATE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Custom Data Loaders
    """)
    return


@app.cell
def _(Dataset, X_train_1, torch, y_train):
    ## train data
    class trainData(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return (self.X_data[index], self.y_data[index])

        def __len__(self):
            return len(self.X_data)

    train_data = trainData(torch.FloatTensor(X_train_1), torch.FloatTensor(y_train))
    return (train_data,)


@app.cell
def _(Dataset, X_test_1, torch):
    ## test data
    class testData(Dataset):
        def __init__(self, X_data):
            self.X_data = X_data

        def __getitem__(self, index):
            return self.X_data[index]

        def __len__(self):
            return len(self.X_data)

    test_data = testData(torch.FloatTensor(X_test_1))
    return (test_data,)


@app.cell
def _(BATCH_SIZE, DataLoader, test_data, train_data):
    # initialize data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Model

    Note that we did not use the Sigmoid activation in our final layer during training. That’s because, we use the nn.BCEWithLogitsLoss() loss function which automatically applies the the Sigmoid activation. We however, need to use Sigmoid manually during inference.
    """)
    return


@app.cell
def _(nn):
    # 2 layer FF DNN with BatchNorm and Dropout
    class binaryClassification(nn.Module):
        def __init__(self):
            super(binaryClassification, self).__init__()
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

    return (binaryClassification,)


@app.cell
def _(torch):
    # check GPU, etc.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # NAA
    # - will output cuda:0 if GPU
    return (device,)


@app.cell
def _():
    #! nvidia-smi
    # subprocess.call(['nvidia-smi'])
    return


@app.cell
def _(LEARNING_RATE, binaryClassification, device, nn, optim):
    # initialize optimizer, loss function
    model = binaryClassification()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return criterion, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train model
    """)
    return


@app.cell
def _(torch):
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    return (binary_acc,)


@app.cell
def _(EPOCHS, binary_acc, criterion, device, model, optimizer, train_loader):
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
            optimizer.zero_grad()  # NAA. Important !!! Zero gradients each epoch run
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
            epoch_acc = epoch_acc + acc.item()
        print(
            f"Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Model
    """)
    return


@app.cell
def _(device, model, test_loader, torch):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch_1 in test_loader:
            X_batch_1 = X_batch_1.to(device)
            y_test_pred = model(X_batch_1)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    return (y_pred_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluate Model
    """)
    return


@app.cell
def _(confusion_matrix, y_pred_list, y_test):
    confusion_matrix(y_test, y_pred_list)
    return


@app.cell
def _(classification_report, y_pred_list, y_test):
    # dump precision, recall, F1
    print(classification_report(y_test, y_pred_list))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
