
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def normalize(file_path):
    p = Path(file_path)
    files = p.glob('labeled_data_512.h5')
    maximum = 0
    minimum = 0
    save_file = h5py.File("normalized_labeled_data_512.h5", mode="w")
    grenzflaeche = save_file.create_group("grenzflaeche")
    spitze = save_file.create_group("spitze")
    referenz = save_file.create_group("referenz")
    for h5dataset_fp in files:
        with h5py.File(h5dataset_fp.resolve()) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in tqdm(group.items()):
                    data = np.array(ds)
                    if np.max(data) > maximum:
                        maximum = max(ds)
                    if np.min(ds) < minimum:
                        minimum = min(ds)
            print(maximum, minimum)
            for gname, group in h5_file.items():
                for dname, ds in tqdm(group.items()):
                    data = np.array(ds)
                    data = np.apply_along_axis(lambda x: (x - minimum) / (maximum - minimum), axis=0, arr=data)
                    if np.max(data) > 1:
                        print("error max to high")
                    elif np.min(data) < 0:
                        print("error max to low")
                    if gname == 'referenz':
                        referenz.create_dataset(f"{dname}", data=data)
                    elif gname == 'spitze':
                        spitze.create_dataset(f"{dname}", data=data)
                    elif gname == 'grenzflaeche':
                        grenzflaeche.create_dataset(f"{dname}", data=data)


if __name__ == "__main__":
    normalize("//")