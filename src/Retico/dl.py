import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model for task 1
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.hidden0 = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden0(x))
        x = F.relu(self.hidden1(x))
        x = self.output(x)
        return x


def convert_datatype(X_df, y_df):

    # Tensorize X and y
    tensor_X = torch.FloatTensor(X_df)
    tensor_y = torch.LongTensor(y_df).view(-1)

    return tensor_X, tensor_y


def pred_coord(X, y):
    """Task 1: Predict coordinate (y-hat)"""

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    # Hyperparameter
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = 15 + 2  # p + No instruction given + Language
    lr = 0.01
    batch_size = 512
    n_epochs = 10
    report_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(input_size, hidden_size, output_size).to(device)

    # Training
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    optimizer = optim.Adam(net.parameters())
    uncert_train = []
    uncert_test = []

    for epoch in range(n_epochs):
        total_loss = 0
        net.train()
        ### in training loop ###
        for i in range(int(len(X_train) / batch_size)):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size].to(device)
            y_batch = y_train[i * batch_size:(i + 1) * batch_size].to(device)
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()

        # Testing    
        if ((epoch + 1) % report_every) == 0:
            net.eval()
            with torch.no_grad():
                # Calculate train accuracy
                target = y_train.to(device)
                pred = net(X_train.to(device))
                pred_label_train = torch.argmax(pred, dim=1)
                train_acc = torch.sum(pred_label_train == target) / len(y_train)

                # Calculate test accuracy
                target = y_test.to(device)
                pred = net(X_test.to(device))
                pred_label_test = torch.argmax(pred, dim=1)
                test_acc = torch.sum(pred_label_test == target) / len(y_test)

            print('epoch: %d, loss: %.4f, train acc: %.4f, test acc: %.4f' % (epoch, total_loss / int(len(X_train) / batch_size), train_acc, test_acc))

    return net


def pred_uncert(X, y):
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    # Hyperparameter for task 2
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = 2
    lr = 0.01
    batch_size = 512
    n_epochs = 10
    report_every = 1

    net = Net(input_size, hidden_size, output_size).to(device)

    # Training
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(n_epochs):
        total_loss_2 = 0
        net.train()
        ### in training loop ###
        for i in range(int(len(X_train) / batch_size)):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size].to(device)
            y_batch = y_train[i * batch_size:(i + 1) * batch_size].to(device)
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            total_loss_2 += loss
            loss.backward()
            optimizer.step()

        # Testing    
        if ((epoch + 1) % report_every) == 0:
            net.eval()
            with torch.no_grad():
                # Calculate train accuracy
                target = y_train.to(device)
                pred = net(X_train.to(device))
                pred_label_train = torch.argmax(pred, dim=1)
                train_acc_2 = torch.sum(pred_label_train == target) / len(y_train)

                # Calculate test accuracy
                target = y_test.to(device)
                pred = net(X_test.to(device))
                pred_label_test = torch.argmax(pred, dim=1)
                test_acc_2 = torch.sum(pred_label_test == target) / len(y_test)

            print('epoch: %d, loss: %.4f, train acc: %.4f, test acc: %.4f' % (epoch, total_loss_2 / int(len(X_train) / batch_size), train_acc_2, test_acc_2))

    return net


if __name__ == '__main__':
    start = time.time()
    # Load the dataset
    print("Loading dataset")
    with open('data/X_DM.pickle', 'rb') as X_file:
        X_df = pickle.load(X_file)
    with open('data/y_DM.pickle', 'rb') as y_file:
        y_df = pickle.load(y_file)
    tensor_X, tensor_y = convert_datatype(X_df, y_df)
    tensor_X, tensor_y = tensor_X[:], tensor_y[:]
    print(tensor_X.shape, tensor_y.shape)

    half = int(len(tensor_X)/2)
    X_1 = tensor_X[:half]
    y_1 = tensor_y[:half]
    X_2 = tensor_X[half:]
    y_2 = tensor_y[half:]

    print("Training model 1")
    net_task_1 = pred_coord(X_1, y_1)
    net_task_1.eval()

    print("Generating y for model 2 using model 1")
    y_uncert = net_task_1(X_2.to(device))
    y_uncert_class = torch.argmax(y_uncert, dim=1)
    bool_tensor = (y_uncert_class != y_2)
    long_tensor = bool_tensor.to(torch.long)

    # Task 2: Predict Uncertainity of the y-hat prediction
    print("Training model 2")
    net_task_2 = pred_uncert(X_2, long_tensor)
    net_task_2.eval()
    end = time.time()
    print(f"Runtime: {end-start:.2f} seconds")

