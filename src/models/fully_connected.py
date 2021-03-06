from src.models.base_functions import *
import torch.nn as nn
from datetime import datetime
import sys


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
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
    data_path, pattern, save_dir = path_init(sys.argv)
    device = device_init()

    epochs = 50
    number_of_seeds = 20

    test_loss, test_accuracy, train_loss, train_accuracy, confusion_matrix_raw, confusion_matrix_normalized, \
        wrong_predictions, right_predictions, validation_accuracy, validation_loss = \
        seed_loop(FC, device, CustomDataset(data_path, pattern), epochs, number_of_seeds)


    metrics = pandas.DataFrame({
        'parameters': total_params(FC().to(device)),

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
