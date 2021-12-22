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
import matplotlib.pyplot as plt

# Wrapping the dataset with load function
class CustomDataset(data.Dataset):
    def __init__(self, file_path, transform=None):
        super().__init__()
        self.train_data_cache = []
        self.test_data_cache = []
        self.manual_data_cache = []
        self.transform = transform
        self.label_count = [0, 0, 0]
        # Search for all h5 files
        p = Path(file_path)
        files = p.glob('*.h5')
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
                            for i in np.split(ds, 4):
                                self.train_data_cache.append([label, torch.tensor(i).unsqueeze(0).type(torch.float32)])
                            k += 1
                        elif j < 400:
                            for i in np.split(ds, 4):
                                self.test_data_cache.append([label, torch.tensor(i).unsqueeze(0).type(torch.float32)])
                            j += 1
                        elif l < 100:
                            for i in np.split(ds, 4):
                                self.manual_data_cache.append([label, torch.tensor(i).unsqueeze(0).type(torch.float32)])
                            l += 1
                        else:
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
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # endsize 1536 maxpool 3
        # self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1) #endsize 1024 maxpool 3
        # self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1) # endsize 512 maxpool 3

        self.fc1 = nn.Linear(384, 3)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = F.max_pool1d(F.relu(self.conv3(x)), 3)
        x = F.max_pool1d(F.relu(self.conv4(x)), 3)
        # x = F.max_pool1d(F.relu(self.conv5(x)),3)
        # x = F.max_pool1d(F.relu(self.conv6(x)),3)
        #logging.debug(x.shape)
        x = torch.flatten(x, 1)
        #logging.debug(x.shape)
        x = F.softmax(self.fc1(x), dim=1)

        return x


# the train loop
def train(dataloader, optimizer, criterion, model):
    model.train()
    running_loss = 0.0
    j = 0
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
        loss.backward()
        optimizer.step()

        # log statistics
        running_loss += loss.item()
        if i % 500 == 499:
            logging.debug(f"[{epoch}]Loss: {running_loss / 500} ")


# testing the accuracy on single 1024 snippets
def split_test(dataloader, optimizer, criterion, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for labels, inputs in dataloader:
            labels, inputs = labels.to(device), inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logging.debug(f" Random Teilstück Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")


# testing the accuracy on whole samples via majority vote
def test_complete(dataloader, optimizer, criterion, model):
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for labels, inputs in dataloader:
            labels = labels.to(device)
            outputs = np.array([])
            for sample in inputs:
                split_input = torch.reshape(sample, (300, 1024)).unsqueeze(1).to(device)
                split_input = split_input[split_input.sum(dim=2) != 0].unsqueeze(1)
                split_output = model(split_input)
                answer_count = np.array([0, 0, 0, 0, 0, 0, 0])
                for i in split_output:
                    answer_count[i.argmax(0)] += 1
                outputs = np.append(outputs, answer_count.argmax(0))
            for i in range(len(outputs)):
                if outputs[i] == labels[i]:
                    correct += 1
    correct /= size

    logging.debug(f"Full Sample Error: Accuracy: {(100 * correct):>0.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # looking for cuda device and selecting it if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.debug('Using {} device'.format(device))

    # loading the data
    customData = CustomDataset("/home/marcus/Dokumente/entladung/")

    # defining the model and moving it to the correct device
    model = Network().to(device)

    # defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # defining train and test sets
    train_data = customData.get_train_data()
    test_data = customData.get_test_data()
    #split_test_data = customData.split_test_data_cache
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=False, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=False, num_workers=4)
    manual_dataloader = DataLoader(customData.manual_data_cache, batch_size=4, shuffle=False, pin_memory=False, num_workers=4)
    #split_test_dataloader = DataLoader(split_test_data, batch_size=32, shuffle=True, pin_memory=False, num_workers=4)
    train_array = np.array(train_data)
    test_array = np.array(test_data)
    test_counts = [0, 0, 0]
    train_counts = [0, 0, 0]
    for i in range(3):
        train_counts[i] = np.count_nonzero([x[0] == i for x in train_array])
        test_counts[i] = np.count_nonzero([x[0] == i for x in test_array])
    plt.bar(range(3), test_counts)
    plt.show()
    plt.bar(range(3), train_counts)
    plt.show()




    # training loop
    for epoch in range(100):  # loop over the dataset multiple times
        train(train_dataloader, optimizer, criterion, model)
        split_test(test_dataloader, optimizer, criterion, model)
        #test_complete(test_dataloader, optimizer, criterion, model)

        if epoch % 10 == 9:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                        }, f"/home/marcus/Dokumente/munzwurf/model{epoch + 1}.tar")

    model.eval()
    for labels, inputs in manual_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logging.debug(model(inputs) + labels)

    logging.debug('Finished Training')
