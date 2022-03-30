import matplotlib.pyplot as plt
import torch
import numpy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import h5py
from pathlib import Path
from torch.utils import data
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import pandas
import random
import sys


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def create_worker_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(data.Dataset):
    def __init__(self, file_path, pattern, transform=None):
        super().__init__()
        self.train_data_cache = []
        self.test_data_cache = []
        self.validation_data_cache = []
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
                    logging.info(gname)
                    if gname == 'neg_grenzflaeche':
                        label = 0
                    elif gname == 'neg_spitze':
                        label = 1
                    elif gname == 'pos_grenzflaeche':
                        label = 2
                    elif gname == 'pos_spitze':
                        label = 3

                    logging.debug(group.items())
                    for dname, ds in tqdm(group.items()):
                        if k < 1801:  # 3000
                            self.train_data_cache.append(
                                [label, torch.tensor(ds[:20000]).unsqueeze(0).type(torch.float32)])
                            k += 1
                        elif j < 601:  # 400
                            self.test_data_cache.append(
                                [label, torch.tensor(ds[:20000]).unsqueeze(0).type(torch.float32)])
                            j += 1
                        elif l < 601:
                            self.validation_data_cache.append(
                                [label, torch.tensor(ds[:20000]).unsqueeze(0).type(torch.float32)])
                        if k == 1801 and j == 601 and l == 601:
                            break

    def __len__(self):
        return len(self.test_data_cache) + len(self.train_data_cache) + len(self.validation_data_cache)

    def get_train_data(self):
        return self.train_data_cache

    def get_test_data(self):
        return self.test_data_cache

    def get_validation_data(self):
        return self.validation_data_cache


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(conv_factor ** 0, conv_factor ** 1, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv1_size = numpy.floor(
            numpy.floor((20000 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.conv2 = nn.Conv1d(conv_factor ** 1, conv_factor ** 2, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv2_size = numpy.floor(
            numpy.floor((conv1_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.conv3 = nn.Conv1d(conv_factor ** 2, conv_factor ** 3, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv3_size = numpy.floor(
            numpy.floor((conv2_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.conv4 = nn.Conv1d(conv_factor ** 3, conv_factor ** 4, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv4_size = numpy.floor(
            numpy.floor((conv3_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.fc1 = nn.Linear(int(conv_factor ** 4 * conv4_size), 4)  # input 1000 / 4 384//8

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), pool_size)
        x = F.max_pool1d(F.relu(self.conv2(x)), pool_size)
        x = F.max_pool1d(F.relu(self.conv3(x)), pool_size)
        x = F.max_pool1d(F.relu(self.conv4(x)), pool_size)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc1(x), dim=1)
        return x


def train(dataloader, optimizer, criterion, model, epoch, device):
    model.train()
    size = len(dataloader.dataset)
    running_loss = 0.0
    loss_values = []
    correct = 0
    for i, data in enumerate(dataloader, 0):
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

        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        # log statistics
        running_loss += loss.item()
    correct /= size
    # logging.info(f"[{epoch}] Train Loss: {running_loss / (i + 1):>8f}, Train accuracy: {correct * 100:>0.1f}% ")
    return running_loss / (i + 1), correct * 100


def test(dataloader, criterion, model, epoch, device):
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
            for i, output in enumerate(outputs):
                if output.argmax(0) != labels[i]:
                    wrong_pred.append([inputs[i], labels[i], output])
                elif output.argmax(0) == labels[i]:
                    right_pred.append([inputs[i], labels[i], output])
    test_loss /= num_batches
    correct /= size
    # logging.info(f" Random Teilstück Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return confusion_matrix(true, pred), confusion_matrix(true, pred, normalize='true'), wrong_pred, right_pred, \
           test_loss, 100*correct


def validation(dataloader, model, criterion, device):
    acc = []
    pred = []
    true = []
    model.eval()
    wrong_pred = []
    right_pred = []
    validation_loss = 0
    correct = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    with torch.no_grad():
        for labels, inputs in dataloader:
            labels, inputs = labels.to(device), inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            acc.append((torch.argmax(outputs, axis=1)==labels).float().mean().item())
            pred.extend(list((torch.argmax(outputs, axis=1).cpu().numpy())))
            true.extend(list(labels.cpu().numpy()))
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            for i, output in enumerate(outputs):
                if output.argmax(0) != labels[i]:
                    wrong_pred.append([inputs[i], labels[i], output])
                elif output.argmax(0) == labels[i]:
                    right_pred.append([inputs[i], labels[i], output])
    validation_loss /= num_batches
    correct /= size
    return correct * 100, validation_loss


def training_loop(epochs, optimizer, criterion, model, train_dataloader, test_dataloader, device):

    test_loss = []
    test_accuracy = []
    train_loss = []
    train_accuracy = []

    # training the model
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        train_loss_current, train_accuracy_current = train(train_dataloader, optimizer, criterion, model, epoch, device)
        train_loss.append(train_loss_current)
        train_accuracy.append(train_accuracy_current)
        confusion_matrix_raw, confusion_matrix_normalized, wrong_predictions, right_predictions, \
            test_loss_current, test_accuracy_current = test(test_dataloader, criterion, model, epoch, device)
        test_loss.append(test_loss_current)
        test_accuracy.append(test_accuracy_current)
    logging.info('Finished Training')

    """
    # plotting loss
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()

    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.show()

    # displaying confusation matrices
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_raw, display_labels=[0, 1, 2, 3])
    disp.plot()
    plt.show()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_normalized, display_labels=[0, 1, 2, 3])
    disp.plot()
    plt.show()
    """

    return test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
           wrong_predictions, right_predictions


def total_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def network_training(epochs, stride, padding, kernel_size, pool_size, dilation, conv_factor, data_path, pattern):

    # setting the gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    # initializing dataset

    customData = CustomDataset(data_path, pattern)

    test_loss_array = []
    test_accuracy_array = []
    train_loss_array = []
    train_accuracy_array = []
    confusion_matrix_raw_array = []
    confusion_matrix_normalized_array = []
    wrong_predictions_array = []
    right_predictions_array = []
    validation_loss_array = []
    validation_accuracy_array = []

    for i in range(20):
        # seeding
        seed = i
        worker_generator = create_worker_generator(seed)
        seed_all(seed)
        logging.info(f"seed={seed}")



        model = Network().to(device)

        # setting loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # logging number of parameters
        total_parameters = total_params(model)
        logging.info(total_parameters)

        # creating Dataloaders
        train_dataloader = DataLoader(customData.get_train_data(), batch_size=256, shuffle=True,
                                      worker_init_fn=seed_worker, generator=worker_generator)
        test_dataloader = DataLoader(customData.get_test_data(), batch_size=64, shuffle=True,
                                     worker_init_fn=seed_worker, generator=worker_generator)
        validation_dataloader = DataLoader(customData.get_validation_data(), batch_size=64, shuffle=True,
                                           worker_init_fn=seed_worker, generator=worker_generator)

        # training the network
        test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, wrong_predictions, right_predictions \
            = training_loop(epochs, optimizer, criterion, model, train_dataloader, test_dataloader, device)

        validation_accuracy, validation_loss = validation(validation_dataloader, model, criterion, device)

        test_loss_array.append(test_loss)
        test_accuracy_array.append(test_loss)

        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy)

        confusion_matrix_raw_array.append(confusion_matrix_raw)
        confusion_matrix_normalized_array.append(confusion_matrix_normalized)

        wrong_predictions_array.append(wrong_predictions)
        right_predictions_array.append(right_predictions)

        validation_loss_array.append(validation_loss)
        validation_accuracy_array.append(validation_accuracy)

    metrics = pandas.DataFrame({
        'parameters': total_parameters,
        'epochs': epochs,
        'stride': stride,
        'padding': padding,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'dilation': dilation,
        'conv_factor': conv_factor,

        'test_loss': test_loss_array,
        'test_accuracy': test_accuracy_array,
        'train_loss': train_loss_array,
        'train_accuracy': train_accuracy_array,
        'confusion_matrix': confusion_matrix_raw_array,
        'confusion_matrix_normalized': confusion_matrix_normalized_array,
        'validation_loss': validation_loss_array,
        'validation_accuracy': validation_accuracy_array
    })

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # home_path = "/home/marcus/Dokumente/entladung/"
    #data_path = "/home/marcus/Dokumente/entladung/modified_data"
    #pattern = 'normalize*.h5'

    arguments = sys.argv
    logging.info(arguments)

    path = arguments[1]
    pattern = arguments[2]
    save_dir = arguments[3]

    # setting hyperparameters
    epochs = 100
    padding = 1
    # kernel_size = 3
    pool_size = 3
    dilation = 1
    conv_factor = 2
    stride = 4

    for kernel_size in range(2, 10):
        results = network_training(epochs, stride, padding, kernel_size, pool_size, dilation, conv_factor, path, pattern)
        results.to_csv(f"{save_dir}_stride{stride}_network_metrics_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
