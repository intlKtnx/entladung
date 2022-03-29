import logging
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
from tqdm import tqdm
import librosa
import seaborn
from string import ascii_lowercase,ascii_uppercase, digits


def data_loader(file_path, pattern):
    p = Path(file_path)
    files = p.glob(pattern)
    data = []

    for h5_dataset_fp in files:
        with h5py.File(h5_dataset_fp.resolve()) as h5_file:
            logging.info(str(h5_dataset_fp))
            for gname, group in h5_file.items():
                group_data = []
                i = 0
                for dname, ds in tqdm(group.items()):
                    arr = np.array(ds[:])
                    max_value = np.amax(arr)
                    peaks = np.count_nonzero(abs(arr) > 80)
                    #logging.info(f"{max_value}, {peaks}")

                    if peaks == 0 and gname == "pos_spitze":
                        plt.plot(arr)
                        plt.savefig(f"/home/marcus/Dokumente/entladung/2_peak_samples/{gname}_{dname}")
                        plt.show()
                    i += 1
            #data.append(numpy.array(group_data))
    return np.array(data, dtype=object)


def whatever(file_path, pattern):
    data = data_loader(file_path, pattern)
    for i in data:
        break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_loader("/home/marcus/Dokumente/entladung/", "correct_rnn*.h5")
