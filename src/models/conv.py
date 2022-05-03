from base_functions import *
import torch.nn as nn
import numpy
import sys
from datetime import datetime


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        out_size_conv = lambda l_in: numpy.floor(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1)
                                                  / stride) + 1)
        out_size_pool = lambda l_in: numpy.floor(((l_in + 2 * pool_padding - pool_dilation * (pool_size - 1) - 1)
                                                  / pool_stride) + 1)
        conv1_size = out_size_conv(input_size)
        # conv2_size = out_size_conv(conv1_size)
        pool1_size = out_size_pool(conv1_size)
        self.model = nn.Sequential(
            nn.Conv1d(conv_factor**0, conv_factor**1, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride, padding=pool_padding),
            # nn.Conv1d(conv_factor ** 1, conv_factor ** 2, kernel_size=kernel_size, padding=padding, stride=stride,
            #           dilation=dilation),
            # nn.ReLU(),
            # nn.MaxPool1d(pool_size, stride=pool_stride, padding=pool_padding, dilation=pool_dilation),
            # nn.AvgPool1d(kernel_size=int(conv2_size)),
            nn.Flatten(),
            nn.Linear(int(pool1_size * conv_factor**1), 4),
            nn.Softmax(dim=1),
            # fc -> convlayer + maxpool -> poolsize& stride drastisch erhöht -> conv stride erhöht
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path, pattern, save_dir = path_init(sys.argv)
    device = device_init()

    # Setting Hyperparameters
    epochs = 100
    input_size = 20002
    number_of_seeds = 20

    # conv paramters
    kernel_size = 3
    padding = 1
    stride = 1
    dilation = 1
    conv_factor = 2

    # maxpool parameters
    pool_size = 31
    pool_padding = 15
    pool_stride = pool_size
    pool_dilation = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    # print_model_params(Network, device)
    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
        wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(Network, device, CustomDataset(data_path, pattern), epochs, number_of_seeds)

    metrics = pandas.DataFrame({
        'parameters': total_params(Network().to(device)),
        'epochs': epochs,
        'stride': stride,
        'padding': padding,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'dilation': dilation,
        'conv_factor': conv_factor,

        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'confusion_matrix': confusion_matrix_raw,
        'confusion_matrix_normalized': confusion_matrix_normalized,
        'validation_loss': validation_loss,
        'validation_accuracy': validation_accuracy
    })

    metrics.to_csv(
        f"{save_dir}_conv_fc_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
