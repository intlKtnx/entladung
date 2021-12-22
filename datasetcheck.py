import logging
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import librosa


def check_data(file_path):
    p = Path(file_path)
    files = p.glob("*data.h5")
    for h5_dataset_fp in files:
        with h5py.File(h5_dataset_fp) as h5_file:
            for gname, group in h5_file.items():
                for dname, data in group.items():
                    logging.info(f"Label:{gname}, Sample:{dname}, Länge:{len(data)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_data("/home/marcus/Dokumente/entladung/")