from base_functions import *
import torch.nn as nn
from datetime import datetime
import sys


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20002, 100),
            nn.Linear(100, 4),
            # nn.Linear(4, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    """
    home_path = "/home/marcus/Dokumente/entladung/"
    data_path = "/home/marcus/Dokumente/entladung/modified_data"
    pattern = 'raw_data_.h5'

    """
    arguments = sys.argv
    logging.info(arguments)

    data_path = arguments[1]
    pattern = arguments[2]
    save_dir = arguments[3]

    epochs = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
        wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(Network, device, CustomDataset(data_path, pattern), epochs, 20)
    """
    for i in confusion_matrix_raw:
        disp = ConfusionMatrixDisplay(i, display_labels=[0, 1, 2, 3])
        disp.plot()
        plt.show()
    """

    metrics = pandas.DataFrame({
        'parameters': total_params(Network().to(device)),

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
        f"{save_dir}fully_connected_without_relu{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
