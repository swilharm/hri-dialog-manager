import random
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
X_df = pd.read_json('data/X_DM.json', orient = 'index')
y_df = pd.read_json('data/y_DM.json', orient = 'index')

def convert_datatype(X_df, y_df):
    ### convert the datatype from list object to float ####
    df_new = X_df.drop(['Instruction', 'Coordinate'], axis=1)
    instr_df = pd.DataFrame(df_new['Instruction Confidence'].to_list())
    coord_df = pd.DataFrame(df_new['Coordinate Confidence'].to_list())
    coord_df0 = pd.DataFrame(coord_df[0].to_list())
    coord_df1 = pd.DataFrame(coord_df[1].to_list())

    # column name
    no_puzzlepieces = 15
    a = [f'Instr{i}' for i in range(3)]
    b = [f'LV{i}' for i in range(no_puzzlepieces)]
    c = [f'G{i}' for i in range(no_puzzlepieces)]

    col_names = []
    col_names.extend(a)
    col_names.extend(b)
    col_names.extend(c)

    final_df_noisy = pd.concat([instr_df, coord_df0, coord_df1], axis=1)
    final_df_noisy.columns = col_names

    # Tensorize X and y
    array_X = final_df_noisy.values
    tensor_X = torch.FloatTensor(array_X) 
    tensor_y = torch.LongTensor(y_df.values).view(-1)

    return tensor_X, tensor_y
    
tensor_X, tensor_y = convert_datatype(X_df, y_df)
print (tensor_X.shape, tensor_y.shape)

# Data split
X_train, X_test, y_train, y_test = train_test_split(tensor_X, tensor_y, test_size=1/3, random_state=42)

# Task 1: Predict coordinate (y-hat)
# Hyperparameter
input_size = X_train.shape[1]
hidden_size = 32
output_size = 15 + 2 # p + No instruction given + Language
lr = 0.01
batch_size = 512
n_epochs = 10
report_every = 1

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

def pred_coord(X_train, X_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(input_size, hidden_size, output_size).to(device)

    # Training
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=lr)
    optimizer = optim.Adam(net.parameters())
    uncert_train = []
    uncert_test = []

    for epoch in range(n_epochs):
        total_loss = 0
        net.train()    
        ### in training loop ###
        for i in range(int(len(X_train)/batch_size)):
            X_batch = X_train[i*batch_size:(i+1)*batch_size].to(device)
            y_batch = y_train[i*batch_size:(i+1)*batch_size].to(device)
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()
            
        # Testing    
        if ((epoch+1) % report_every) == 0:
            net.eval()
            with torch.no_grad():
                # Calculate train accuracy
                target = y_train.to(device)
                pred = net(X_train.to(device))
                pred_label_train = torch.argmax(pred, dim=1)
                train_acc = torch.sum(pred_label_train==target) / len(y_train)
                # For task 2
                for i in range(len(y_train)):
                    if pred_label_train[i] == target[i]:
                        uncert_train.append(0)
                    else:
                        uncert_train.append(1) 

                # Calculate test accuracy
                target = y_test.to(device)
                pred = net(X_test.to(device))
                pred_label_test = torch.argmax(pred, dim=1)
                test_acc = torch.sum(pred_label_test==target) / len(y_test)
                # For task 2
                for i in range(len(y_test)):
                    if pred_label_test[i] == target[i]:
                        uncert_test.append(0)
                    else:
                        uncert_test.append(1) 

        print('epoch: %d, loss: %.4f, train acc: %.2f, test acc: %.2f' % (epoch, total_loss/int(len(X_train)/batch_size), train_acc, test_acc))
    return(uncert_train, uncert_test)

uncert_train, uncert_test = pred_coord(X_train, X_test, y_train, y_test)

# Task 2: Predict Uncertainity of the y-hat prediction
# Dataset
y_train_2 = torch.LongTensor(uncert_train).view(-1)
y_test_2 = torch.LongTensor(uncert_test).view(-1)

# Hyperparameter for task 2
input_size = X_train.shape[1]
hidden_size = 32
output_size = 15 + 2 # p + No instruction given + Language
lr = 0.01
batch_size = 512
n_epochs = 1000
report_every = 100

# model for task 2
device_2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_2 = Net(input_size, hidden_size, output_size).to(device_2)

def pred_uncert(X_train, X_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(input_size, hidden_size, output_size).to(device)

    # Training
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=lr)
    optimizer = optim.Adam(net.parameters())

    for epoch in range(n_epochs):
        total_loss_2 = 0
        net.train()    
        ### in training loop ###
        for i in range(int(len(X_train)/batch_size)):
            X_batch = X_train[i*batch_size:(i+1)*batch_size].to(device)
            y_batch = y_train[i*batch_size:(i+1)*batch_size].to(device)
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            total_loss_2 += loss
            loss.backward()
            optimizer.step()
            
        # Testing    
        if ((epoch+1) % report_every) == 0:
            net.eval()
            with torch.no_grad():
                # Calculate train accuracy
                target = y_train.to(device)
                pred = net(X_train.to(device))
                pred_label_train = torch.argmax(pred, dim=1)
                train_acc_2 = torch.sum(pred_label_train==target) / len(y_train)

                # Calculate test accuracy
                target = y_test.to(device)
                pred = net(X_test.to(device))
                pred_label_test = torch.argmax(pred, dim=1)
                test_acc_2 = torch.sum(pred_label_test==target) / len(y_test)

        #print('epoch: %d, loss: %.4f, train acc: %.2f, test acc: %.2f' % (epoch, total_loss_2/int(len(X_train)/batch_size), train_acc_2, test_acc_2))
        print('epoch: %d, loss: %.4f, train acc: %.2f, test acc: %.2f' % (epoch, total_loss_2/int(len(X_train)/batch_size), train_acc_2, test_acc_2))

pred_uncert(X_train, X_test, y_train_2, y_test_2)





"""
# Training

# Update the weights
# Loss Function
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=lr)
optimizer = optim.Adam(net.parameters())

for epoch in range(n_epochs):
    total_loss = 0
    
    net.train()    
    ### in training loop ###
    for i in range(int(len(X_train)/batch_size)):
        X_batch = X_train[i*batch_size:(i+1)*batch_size].to(device_2)
        y_batch_2 = y_train_2[i*batch_size:(i+1)*batch_size].to(device_2)
        optimizer.zero_grad()
        pred = net_2(X_batch)
        loss = criterion(pred, y_batch_2)
        total_loss += loss
        loss.backward()
        optimizer.step()
        
    if ((epoch+1) % report_every) == 0:
        net.eval()
        with torch.no_grad():
            # Calculate train accuracy
            target = y_train_2.to(device_2)
            pred = net(X_train.to(device_2))
            pred_label_train = torch.argmax(pred, dim=1)
            print (X_train.shape, pred_label_train, pred_label_train.shape, target.shape)
            train_acc = torch.sum(pred_label_train==y_train_2) / len(y_train_2)

            # Calculate test accuracy
            target = y_test_2.to(device_2)
            pred = net(X_test.to(device_2))
            pred_label = torch.argmax(pred, dim=1)
            test_acc = torch.sum(pred_label==target) / len(y_test_2)
        print('epoch: %d, loss: %.4f, train acc: %.2f, test acc: %.2f' % (epoch, total_loss/int(len(X_train)/batch_size), train_acc, test_acc))

"""