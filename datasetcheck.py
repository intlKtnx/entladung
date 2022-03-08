import logging
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import librosa
from string import ascii_lowercase,ascii_uppercase, digits

def check_data(file_path):
    p = Path(file_path)
    files = p.glob("*pos*_G*.mat")
    for h5_dataset_fp in files:
        with h5py.File(h5_dataset_fp) as h5_file:
            logging.info(str(h5_dataset_fp))
            for gname, group in h5_file.items():
                for i in range(5):
                    for j in tqdm(np.split(group[i], 3500)):
                        plt.plot(j)
                    plt.show()




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_data("/home/marcus/Dokumente/entladung/data")