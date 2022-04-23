from base_functions import *
import torch.nn as nn
import numpy


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        out_size_conv = lambda l_in: numpy.floor(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1)
                                                  / stride) + 1)
        out_size_pool = lambda l_in: numpy.floor(((l_in + 2 * pool_padding - pool_dilation * (pool_size - 1) - 1)
                                                  / pool_stride) + 1)
        conv1_size = out_size_conv(input_size)
        conv2_size = out_size_conv(conv1_size)
        self.model = nn.Sequential(
            nn.Conv1d(conv_factor**0, conv_factor**1, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(conv_factor ** 1, conv_factor ** 2, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation),
            nn.ReLU(),
            # nn.MaxPool1d(pool_size, stride=pool_stride, padding=pool_padding, dilation=pool_dilation),
            nn.AvgPool1d(kernel_size=int(conv2_size)),
            nn.Flatten(),
            # nn.Linear(int(conv1_size * conv_factor**2), 4),
            nn.Softmax(dim=1),
            # fc -> convlayer + maxpool -> poolsize& stride drastisch erhöht -> conv stride erhöht
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # home_path = "/home/marcus/Dokumente/entladung/"
    data_path = "/home/marcus/Dokumente/entladung/modified_data"
    pattern = 'raw_data_.h5'

    """
    arguments = sys.argv
    logging.info(arguments)

    path = arguments[1]
    pattern = arguments[2]
    save_dir = arguments[3]
    """
    # Setting Hyperparameters
    epochs = 75
    input_size = 20002
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

    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
    wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(Network, device, CustomDataset(data_path, pattern), epochs, 10)

    model = Network().to(device)
    print(model)
    params_per_layer = list((p.numel() for p in model.parameters() if p.requires_grad))
    print(params_per_layer)
    print(sum(params_per_layer))

    for i in confusion_matrix_raw:
        disp = ConfusionMatrixDisplay(i, display_labels=[0, 1, 2, 3])
        disp.plot()
        plt.show()
