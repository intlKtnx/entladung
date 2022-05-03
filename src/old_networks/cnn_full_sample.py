from base_functions import *
import sys
from datetime import datetime


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(conv_factor ** 0, conv_factor ** 1, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv1_size = numpy.floor(
            numpy.floor((20002 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.conv2 = nn.Conv1d(conv_factor ** 1, conv_factor ** 2, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv2_size = numpy.floor(
            numpy.floor((conv1_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.conv3 = nn.Conv1d(conv_factor ** 2, conv_factor ** 3, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation)
        conv3_size = numpy.floor(
            numpy.floor((conv2_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        # self.conv4 = nn.Conv1d(conv_factor ** 3, conv_factor ** 4, kernel_size=kernel_size, padding=padding,
        #                       stride=stride, dilation=dilation)
        # conv4_size = numpy.floor(
        #    numpy.floor((conv3_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

        self.fc1 = nn.Linear(int(conv_factor ** 3 * conv3_size), 4)  # input 1000 / 4 384//8

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), pool_size)
        x = F.max_pool1d(F.relu(self.conv2(x)), pool_size)
        x = F.max_pool1d(F.relu(self.conv3(x)), pool_size)
        # x = F.max_pool1d(F.relu(self.conv4(x)), pool_size)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc1(x), dim=1)
        return x


def network_training(epochs, stride, padding, kernel_size, pool_size, dilation, conv_factor, data_path, pattern,
                     Network):
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

    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
        wrong_predictions, right_predictions, validation_accuracy, validation_loss = seed_loop(Network, device, customData, epochs, 20)

    test_loss_array.append(test_loss)
    test_accuracy_array.append(test_accuracy)

    train_loss_array.append(train_loss)
    train_accuracy_array.append(train_accuracy)

    confusion_matrix_raw_array.append(confusion_matrix_raw)
    confusion_matrix_normalized_array.append(confusion_matrix_normalized)

    wrong_predictions_array.append(wrong_predictions)
    right_predictions_array.append(right_predictions)

    validation_loss_array.append(validation_loss)
    validation_accuracy_array.append(validation_accuracy)

    metrics = pandas.DataFrame({
        'parameters': total_params(Network().to(device)),
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
    # data_path = "/home/marcus/Dokumente/entladung/modified_data"
    # pattern = 'normalize*.h5'

    arguments = sys.argv
    logging.info(arguments)

    path = arguments[1]
    pattern = arguments[2]
    save_dir = arguments[3]

    # setting hyperparameters
    epochs = 100
    kernel_size = 3
    padding = int(numpy.floor(kernel_size / 2))
    pool_size = 3
    stride = 3
    dilation = 1
    conv_factor = 3

    # 3 conv layer

    for stride in range(3, 4):
        results = network_training(epochs, stride, padding, kernel_size, pool_size, dilation, conv_factor, path,
                                   pattern, Network)
        results.to_csv(f"{save_dir}stride{stride}_3layers_network_metrics_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
