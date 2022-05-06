from base_functions import *
import torch.nn as nn
import sys
from datetime import datetime


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)

        out = out[:, -1, :]
        out = self.softmax(self.fc(out))
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path, pattern, save_dir = path_init(sys.argv)
    device = device_init()

    # Setting Hyperparameters
    num_classes = 4
    input_size = 20
    sequence_length = 1000
    number_of_seeds = 20

    hidden_size = 512
    num_layers = 1
    epochs = 100


    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
    wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(RNN, device, CustomDataset(data_path, pattern, rnn=True), epochs, number_of_seeds, rnn=True, sequence_length=sequence_length, input_size=input_size)

    metrics = pandas.DataFrame({
        'parameters': total_params(RNN().to(device)),
        'epochs': epochs,
        'input_size': input_size,
        'sequence_length': sequence_length,
        'hidden_size': hidden_size,
        'num_layers': num_layers,

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
        f"{save_dir}rnn_128{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")

