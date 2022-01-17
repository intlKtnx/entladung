from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import h5py
from pathlib import Path
from torch.utils import data
import logging
from collections import Counter
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import pandas as pd
import sys
import datetime


# Wrapping the dataset with load function
class CustomDataset(data.Dataset):
    def __init__(self, file_path, pattern="normalized*512*.h5",  transform=None):
        super().__init__()
        self.train_data_cache = []
        self.test_data_cache = []
        self.manual_data_cache = []
        self.full_test_data_cache = []
        self.transform = transform
        self.label_count = [0, 0, 0]
        # Search for all h5 files
        p = Path(file_path)
        files = p.glob(pattern)
        logging.debug(files)
        for h5dataset_fp in files:
            logging.debug(h5dataset_fp)
            with h5py.File(h5dataset_fp.resolve()) as h5_file:
                # Walk through all groups, extracting datasets
                for gname, group in h5_file.items():
                    k = 0
                    j = 0
                    l = 0
                    if gname == 'referenz':
                        label = 0
                    elif gname == 'spitze':
                        label = 1
                    elif gname == 'grenzflaeche':
                        label = 2

                    logging.debug(group.items())
                    for dname, ds in tqdm(group.items()):
                        if k < 3000:
                            for i in np.split(ds, 2):
                                self.train_data_cache.append([label, torch.tensor(i).unsqueeze(0).type(torch.float32)])
                            k += 1
                        elif j < 400:
                            self.full_test_data_cache.append([label, ds])
                            for i in np.split(ds, 2):
                                self.test_data_cache.append([label, torch.tensor(i).unsqueeze(0).type(torch.float32)])
                            j += 1
                        elif l < 100:
                            for i in np.split(ds, 2):
                                self.manual_data_cache.append([label, torch.tensor(i).unsqueeze(0).type(torch.float32)])
                            l += 1
                        if k == 3000 and j == 400 and l == 100:
                            break

    def __getitem__(self, index):
        return self.data_cache[index]

    def get_test_data(self):
        return self.test_data_cache

    def get_train_data(self):
        return self.train_data_cache

    def __len__(self):
        return len(self.data_cache)


# Defining the network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(1, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(8, 16, kernel_size=3, padding=1)# endsize 1536 maxpool 3
        #self.conv5 = nn.Conv1d(16, 32, kernel_size=3, padding=1) #endsize 1024 maxpool 3
        # self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1) # endsize 512 maxpool 3

        self.fc1 = nn.Linear(112, 3) #input 1000 / 4 384//8

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 3)
        x = F.max_pool1d(F.relu(self.conv4(x)), 3)
        #x = F.max_pool1d(F.relu(self.conv5(x)), 4)
        # x = F.max_pool1d(F.relu(self.conv6(x)),3)
        #logging.debug(x.shape)
        x = torch.flatten(x, 1)
        #logging.debug(x.shape)
        x = F.softmax(self.fc1(x), dim=1)

        return x


# the train loop
def train(dataloader, optimizer, criterion, model, device):
    model.train()
    running_loss = 0.0
    j = 0
    loss_values = []
    for i, data in enumerate(train_dataloader, 0):

        # get the inputs; data is a list of [labels, inputs]

        inputs = data[1]
        labels = data[0]
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / (i + 1)


# testing the accuracy on single 1024 snippets
def split_test(dataloader,criterion, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    ACC = []
    true = []
    pred = []
    right_pred = []
    wrong_pred = []
    test_loss, correct = 0, 0
    with torch.no_grad():
        for labels, inputs in dataloader:
            labels, inputs = labels.to(device), inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            ACC.append((torch.argmax(outputs,axis=1)==labels).float().mean().item())
            pred.extend(list((torch.argmax(outputs,axis=1).cpu().numpy())))
            true.extend(list(labels.cpu().numpy()))
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            """
            for i, output in enumerate(outputs):
                if output.argmax(0) != labels[i]:
                    wrong_pred.append([inputs[i], labels[i], output])
                elif output.argmax(0) == labels[i]:
                    right_pred.append([inputs[i], labels[i], output])
            """
    test_loss /= num_batches
    correct /= size
    logging.info(f"Acc: {(100 * correct):>0.1f}%, test loss: {test_loss:>8f}")
    return (confusion_matrix(true, pred), confusion_matrix(true, pred, normalize='true')), test_loss


def test_complete(dataloader, optimizer, criterion, model):
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    correct = 0
    ACC = []
    true = []
    pred = []
    with torch.no_grad():
        for labels, inputs in dataloader:
            labels = labels.to(device)
            outputs = np.array([])
            for sample in inputs:
                #splitted_input = torch.reshape(sample, (4, 250)).unsqueeze(0).to(device)
                splitted_input
                splitted_output = model(splitted_input)
                answer_count = np.array([0,0,0])
                for i in splitted_output:
                    answer_count[i.argmax(0)] += 1
                #print(answer_count, answer_count.sum())
                outputs = np.append(outputs, answer_count.argmax(0))
                ACC.append((torch.argmax(outputs,axis=1)==labels).float().mean().item())
                pred.extend(list((torch.argmax(outputs,axis=1).cpu().numpy())))
                true.extend(list(labels.cpu().numpy()))
            for i in range(len(outputs)):
                if outputs[i] == labels[i]:
                    correct += 1
            #correct += (outputs == labels).type(torch.float).sum().item()
    correct /= size
    print(f"Full Sample Error: Accuracy: {(100*correct):>0.1f}%")
    return confusion_matrix(true, pred), confusion_matrix(true, pred, normalize='true')


def get_total_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_key_distribution(data):
    cache = []
    for i, val in enumerate(data):
        cache.append(test_data[i][0])
    return Counter(cache).keys(), Counter(cache).values()


# training loop
def train_model(epochs, optimizer, criterion, model, device):
    test_loss_values = []
    train_loss_values = []
    logging.info(get_total_params(model))
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss= train(train_dataloader, optimizer, criterion, model, device)
        logging.info(f"[{epoch}] Train Loss: {train_loss}")
        CM, test_loss = split_test(test_dataloader, criterion, model, device)
        test_loss_values.append(test_loss)
        train_loss_values.append(train_loss)
    logging.info('Finished Training')
    return CM, (train_loss_values, test_loss_values)


def display_confusion_matrix(confusion_matrix,):
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[0], display_labels=[0, 1, 2])
    display.plot()
    plt.show()
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[1], display_labels=[0, 1, 2])
    display.plot()
    plt.show()


def save_confusion_matrix(confusion_matrix, save_dir):
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[0], display_labels=[0, 1, 2])
    display.plot()
    plt.savefig(f"{save_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_cm0.png")
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[1], display_labels=[0, 1, 2])
    display.plot()
    plt.savefig(f"{save_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_cm1.png")


def save_seed():
    seed = np.random.get_state()
    df = pd.DataFrame(data = seed[1])
    df.to_csv("entladung_randomseed3.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    arguments = sys.argv
    logging.info(arguments)
    path = arguments[1]
    pattern = arguments[2]
    save_dir = arguments[3]

    # loading the data

    customData = CustomDataset(path, pattern)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))
    model = Network().to(device)

    # defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # defining train and test sets
    train_data = customData.get_train_data()
    test_data = customData.get_test_data()
    full_test_data = customData.full_test_data_cache
    full_test_data = DataLoader(full_test_data, batch_size=64, shuffle=True, pin_memory=False, num_workers=0)
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=False, num_workers=0)
    manual_dataloader = DataLoader(customData.manual_data_cache, batch_size=4, shuffle=False, pin_memory=False, num_workers=0)
    # torch.save(model.state_dict(), "/home/marcus/Dokumente/entladung/best_model")
    confusion_matrix, loss_values = train_model(60, optimizer, criterion, model, device)
    save_confusion_matrix(confusion_matrix, save_dir)
    plt.plot(loss_values[0])
    plt.plot(loss_values[1])
    plt.savefig(f"{save_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_loss.png")

