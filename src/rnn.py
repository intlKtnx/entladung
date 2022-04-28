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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # out, _ = self.rnn(x, h0)
        # or:
        out, _ = self.lstm(x, (h0,c0))

        out = out[:, -1, :]
        out = self.fc(out)
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_dir = "/home/marcus/Dokumente/entladung/"
    data_path = "/home/marcus/Dokumente/entladung/modified_data"
    pattern = 'raw_data_.h5'

    arguments = sys.argv
    logging.info(arguments)
    if len(sys.argv) >= 2:

        data_path = arguments[1]
        pattern = arguments[2]
        save_dir = arguments[3]

    # Setting Hyperparameters
    num_classes = 4
    input_size = 20000
    sequence_length = 1
    hidden_size = 64
    num_layers = 1
    epochs = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    # print_model_params(Network, device)
    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
    wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(RNN, device, CustomDataset(data_path, pattern, rnn=True), epochs, 20, rnn=True, sequence_length=sequence_length, input_size=input_size)

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
        f"{save_dir}rnn{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
