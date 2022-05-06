from base_functions import *
import torch.nn as nn
import numpy
import sys
from datetime import datetime


class CONV_NO_FC(nn.Module):
    def __init__(self):
        super(CONV_NO_FC, self).__init__()

        out_size_conv = lambda l_in: numpy.floor(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1)
                                                  / stride) + 1)
        out_size_pool = lambda l_in: numpy.floor(((l_in + 2 * pool_padding - pool_dilation * (pool_size - 1) - 1)
                                                  / pool_stride) + 1)
        conv1_size = out_size_conv(input_size)
        self.model = nn.Sequential(
            nn.Conv1d(conv_factor**0, conv_factor**1, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation),
            nn.ReLU(),

            nn.Conv1d(conv_factor ** 1, conv_factor ** 2, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=int(conv1_size)),
            nn.Flatten(),
            nn.Softmax(dim=1),
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
    pool_size = 3
    pool_padding = 1
    pool_stride = pool_size
    pool_dilation = 1

    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
        wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(CONV_NO_FC, device, CustomDataset(data_path, pattern), epochs, number_of_seeds)

    for i in confusion_matrix_raw:
        disp = ConfusionMatrixDisplay(i, display_labels=[0, 1, 2, 3])
        disp.plot()
        plt.show()

    metrics = pandas.DataFrame({
        'parameters': total_params(CONV_NO_FC().to(device)),
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
        f"{save_dir}conv_without_fc_3pool_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
