import logging
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import librosa
from enum import Enum
from string import ascii_lowercase,ascii_uppercase, digits

ugly_samples1 = ['0y', '2P', '2R', '3w', '88', '97', '9m', 'DD', 'Dv', 'Ex', 'FQ', 'I8', 'IA', 'JQ', 'PA', 'Qy', 'S7',
                 'Tbb', 'Tf', 'Up', 'VT', 'WB', 'WT', 'WW', 'Wf', 'Zi', 'ag', 'cn', 'f3', 'fl', 'fu', 'ih', 'jf', 'k',
                 'kg', 'oP', 'tx', 'tz', 'yM']
ugly_samples2 = ['0D', '0E', '0K', '0c', '0d', '0e', '0p', '0t', '0x', '14', '1B', '1D', '1U', '1Z', '1c', '1e', '1f',
                 '1g', '1h', '1o', '1r', '1t', '1u', '1w', '2B', '2D', '2K', '2c', '2d', '2e', '2f', '2t', '2u', '3D',
                 '3E', '3L', '3O', '3c', '3d', '3e', '3f', '3k', '3r', '3t', '3u', '3z', '4', '4B', '4D', '4K', '4U',
                 '4c', '4d', '4f', '4t', '4u', '4z', '5', '53', '5D', '5K', '5P', '5c', '5d', '5e', '5f', '5g', '5k',
                 '5o', '5u', '5x', '5z', '6E', '6K', '6L', '6T', '6U', '6c', '6e', '6f', '6k', '6o', '6t', '6x', '6z',
                 '7A', '7D', '7L', '7Q', '7Y', '7e', '7f', '7t', '7x', '7z', '8D', '8K', '8L', '8M', '8P', '8d', '8e',
                 '8f', '8t', '9D', '9S', '9T', '9U', '9c', '9d', '9e', '9r', '9t', '9w', '9x', '9z', 'A', 'AE', 'AH',
                 'AT', 'AU', 'Ae', 'Af', 'Ag', 'Ah', 'Al', 'Ao', 'Av', 'B', 'BE', 'BU', 'Bc', 'Be', 'Bg', 'Bh', 'Bl',
                 'Bo', 'Bs', 'Bt', 'Bv', 'By', 'C', 'CD', 'CK', 'CL', 'CQ', 'Cf', 'Co', 'Cs', 'Cu', 'Cv', 'D', 'D1',
                 'DD', 'DK', 'DP', 'De', 'Df', 'Di', 'Dl', 'Dp', 'Dv', 'E', 'E1', 'EU', 'EV', 'Ef', 'Eg', 'Eh', 'Ek',
                 'Em', 'Er', 'Et', 'Eu', 'F', 'FH', 'FK', 'FT', 'Ff', 'Fh', 'Fl', 'Fm', 'Fv', 'Fx', 'G', 'GD', 'GF',
                 'GY', 'Gb', 'Gc', 'Gf', 'Gh', 'Gk', 'Gm', 'Gq', 'Gs', 'Gu', 'Gv', 'H', 'HD', 'HM', 'HN', 'HR', 'HU',
                 'Hc', 'Hd', 'Hf', 'Hg', 'Ho', 'Hs', 'Hu', 'Hv', 'Hx', 'I', 'I4', 'IF', 'Ib', 'Ie', 'If', 'Ih', 'Il',
                 'Is', 'It', 'Iu', 'Iv', 'J', 'JA', 'Jb', 'Jf', 'Jg', 'Jk', 'Jl', 'Jr', 'Ju', 'Jv', 'Jw', 'K', 'Kb',
                 'Kd', 'Kk', 'Ko', 'Ku', 'Kv', 'Kz', 'L', 'LG', 'LK', 'LU', 'Ld', 'Le', 'Lf', 'Lg', 'Lh', 'Lt', 'Lu',
                 'Lz', 'M', 'M4', 'MB', 'MD', 'MR', 'MT', 'MY', 'Mb', 'Me', 'Mf', 'Mg', 'Ml', 'Mp', 'Ms', 'Mu', 'Mz',
                 'N', 'N4', 'NB', 'NO', 'NV', 'NZ', 'Nc', 'Ne', 'Nf', 'Ng', 'Nh', 'Nl', 'Np', 'Nt', 'Nu', 'O', 'O3',
                 'O4', 'OD', 'OH', 'OO', 'OY', 'Oc', 'Od', 'Oe', 'Of', 'Og', 'Ol', 'Os', 'Ot', 'Ou', 'Ov', 'Oz', 'P',
                 'PD', 'PH', 'PO', 'PP', 'PQ', 'Pb', 'Pd', 'Pe', 'Pf', 'Pl', 'Pt', 'Pu', 'Px', 'Pz', 'Q', 'QD', 'QH',
                 'QO', 'Qb', 'Qd', 'Qe', 'Qo', 'Qs', 'Qt', 'Qu', 'Qz', 'R', 'R4', 'RD', 'RO', 'Rb', 'Rc', 'Rd', 'Re',
                 'Rf', 'Rp', 'Rs', 'Rt', 'Ru', 'S', 'SD', 'SJ', 'SP', 'Sg', 'Sh', 'So', 'Sp', 'Su', 'T', 'TD', 'TH',
                 'TK', 'Td', 'Tl', 'Tp', 'Tt', 'Tu', 'U', 'UD', 'UU', 'Uc', 'Ue', 'Ug', 'Uh', 'Up', 'Ut', 'V', 'VB',
                 'VC', 'VW', 'Vb', 'Vc', 'Ve', 'Vf', 'Vh', 'Vo', 'Vp', 'Vt', 'Vu', 'Vx', 'W2', 'WB', 'WN', 'WO', 'WV',
                 'Wc', 'Wd', 'We', 'Wf', 'Wh', 'Wo', 'Wp', 'Wt', 'Wu', 'Ww', 'Wx', 'X2', 'XD', 'XH', 'XM', 'XO', 'Xc',
                 'Xe', 'Xf', 'Xj', 'Xr', 'Xt', 'Xu', 'YB', 'YO', 'YR', 'YZ', 'Yb', 'Yd', 'Ye', 'Yf', 'Yl', 'Yr', 'Ys',
                 'Yt', 'Yu', 'ZA', 'ZB', 'ZD', 'ZH', 'ZJ', 'ZW', 'ZZ', 'Zb', 'Zc', 'Zd', 'Ze', 'Zf', 'Zi', 'Zr', 'Zt',
                 'Zu', 'Zv', 'a4', 'aE', 'aL', 'aM', 'aS', 'ad', 'af', 'aq', 'au', 'ay', 'b', 'bC', 'bK', 'bL', 'bM',
                 'bS', 'bb', 'bd', 'be', 'bf', 'bg', 'bh', 'bn', 'bx', 'c', 'cC', 'cE', 'cK', 'cM', 'cS', 'cV', 'cc',
                 'ce', 'cf', 'cl', 'cu', 'cv', 'cy', 'd', 'd2', 'dA', 'dC', 'dE', 'dF', 'dJ', 'dS', 'df', 'dn', 'do',
                 'du', 'dv', 'dy', 'e', 'e2', 'eC', 'eF', 'eH', 'eS', 'eb', 'ed', 'ee', 'ef', 'eh', 'es', 'eu', 'ev',
                 'ex', 'ey', 'f', 'fD', 'fS', 'fU', 'fe', 'ff', 'fg', 'fo', 'ft', 'fu', 'fy', 'g', 'gE', 'gM', 'gQ',
                 'ge', 'gf', 'gg', 'gq', 'gu', 'h', 'h4', 'hA', 'hE', 'hF', 'hL', 'hN', 'hQ', 'hS', 'hW', 'hb', 'he',
                 'hf', 'hn', 'hq', 'hu', 'hv', 'i', 'iF', 'iW', 'ib', 'if', 'ig', 'is', 'iu', 'ix', 'iy', 'jC', 'jH',
                 'jK', 'jV', 'jb', 'je', 'jf', 'jh', 'jn', 'jo', 'js', 'jt', 'ju', 'jv', 'jw', 'jx', 'jy', 'k', 'kC',
                 'kf', 'kh', 'kq', 'ks', 'ku', 'kv', 'l', 'lE', 'lF', 'lG', 'lO', 'lY', 'le', 'lf', 'lp', 'ls', 'lu',
                 'lv', 'lx', 'ly', 'm', 'm1', 'mI', 'mQ', 'md', 'me', 'mf', 'mg', 'mh', 'mk', 'mo', 'mq', 'ms', 'mu',
                 'mv', 'mx', 'my', 'n', 'n1', 'nN', 'nb', 'ne', 'nf', 'ng', 'nk', 'nq', 'nu', 'nv', 'o', 'o1', 'o4',
                 'oA', 'oE', 'oQ', 'oT', 'oe', 'of', 'oh', 'oo', 'ou', 'ov', 'ox', 'p', 'p5', 'pA', 'pO', 'pd', 'pf',
                 'pl', 'po', 'pr', 'pt', 'pu', 'pv', 'py', 'q', 'qC', 'qO', 'qP', 'qb', 'qe', 'qf', 'qh', 'ql', 'qo',
                 'qr', 'qt', 'qu', 'qv', 'qz', 'r', 'rD', 'rL', 'rX', 'rb', 'rc', 're', 'rf', 'rl', 'rr', 'ru', 'rv',
                 'ry', 's', 'sB', 'sF', 'sX', 'sY', 'se', 'sf', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sz', 't',
                 't1', 'tB', 'tL', 'tO', 'tS', 'tX', 'te', 'tf', 'th', 'tl', 'tn', 'tu', 'ty', 'tz', 'u', 'u1', 'uB',
                 'uD', 'uL', 'uM', 'uO', 'ue', 'uf', 'ug', 'ul', 'uo', 'ut', 'uu', 'uv', 'uw', 'uy', 'v', 'v1', 'vA',
                 'vH', 'vL', 'vX', 'vb', 've', 'vf', 'vl', 'vn', 'vo', 'vq', 'vv', 'vz', 'wA', 'wc', 'wf', 'wh', 'wl',
                 'wr', 'ws', 'wu', 'wv', 'wz', 'xA', 'xE', 'xK', 'xM', 'xN', 'xU', 'xZ', 'xc', 'xe', 'xf', 'xg', 'xh',
                 'xl', 'xn', 'xo', 'xr', 'xv', 'y', 'yC', 'ye', 'yf', 'yh', 'yl', 'yr', 'yv', 'z', 'z5', 'zG', 'zH',
                 'zW', 'zc', 'zd', 'ze', 'zf', 'zh', 'zl', 'zt', 'zu', 'zv']


class Datensatz(Enum):
    GRENZFLÄCHE = 1
    SPITZE = 2


def neg_data_toh5(file_path, save_file):
    neg_grenzflache = save_file.create_group("neg_grenzflaeche")
    neg_spitze = save_file.create_group("neg_spitze")
    p = Path(file_path)
    files = p.glob('*neg*.mat')
    i = 0


    for h5_dataset_fp in files:
        with h5py.File(h5_dataset_fp.resolve()) as h5_file:
            #logging.info("Spitze" in str(h5_dataset_fp))
            if "Spitze" in str(h5_dataset_fp):
                datensatz_name = "ohne"
            else:
                datensatz_name = "mit"
            logging.info(str(h5_dataset_fp))
            for gname, group in h5_file.items():
                if isinstance(group, h5py.Group):
                    sample_sum = 0
                    neg_grenzflache_sum = 0
                    neg_spitze_sum = 0
                    for samplename, group2 in tqdm(group.items()):
                        if gname == '#refs#':
                            if not len(group2) == 5:
                                continue
                            else:
                                if isinstance(group2[2], np.ndarray):
                                    #logging.info(f"{samplename}_{datensatz_name}")
                                    if datensatz_name == "mit":
                                        neg_grenzflache.create_dataset(name=f"{samplename}_{datensatz_name}", data=group2[2])
                                        neg_grenzflache_sum += 1
                                    elif datensatz_name == "ohne":
                                        neg_spitze.create_dataset(name=f"{samplename}_{datensatz_name}", data=group2[2])
                                        neg_spitze_sum += 1
    logging.info(sample_sum)
    logging.info(f"neg_grenzfläche:{neg_grenzflache_sum}, "
                 f"neg_spitze:{neg_spitze_sum}")


def pos_data_toh5(file_path, save_file):
    p = Path(file_path)
    pos_grenzflaeche = save_file.create_group("pos_grenzflaeche")
    pos_spitze = save_file.create_group("pos_spitze")
    files = p.glob("*pos*.mat")
    letters = ascii_uppercase + ascii_lowercase
    for h5_dataset_fp in files:
        with h5py.File(h5_dataset_fp) as h5_file:
            logging.info(str(h5_dataset_fp))
            if "ohne" in str(h5_dataset_fp):
                datensatz_name = "ohne"
                laenge = 3000
            else:
                datensatz_name = "mit"
                laenge = 3500
            data_one_dimension = None
            pos_grenzflaeche_sum = 0
            pos_spitze_sum = 0
            for gname, group in h5_file.items():
                data = np.array(np.reshape(group[4], (1, laenge, 20002)))
                if data_one_dimension is None:
                        data_one_dimension = data
                else:
                    data_one_dimension = np.concatenate(data_one_dimension, data, axis=0)
                letters = ascii_uppercase + ascii_lowercase
            for i in tqdm(range(laenge)):
                digit1 = i % len(letters)
                digit2 = int(((i - digit1) / len(letters)) % len(letters))
                digit3 = int(((i - (digit1 + digit2 * len(letters))) / (len(letters)**2)) % len(letters))
                index = letters[digit3] + letters[digit2] + letters[digit1]
                #logging.info(index)
                if datensatz_name == "mit":
                    pos_grenzflaeche.create_dataset(name=f"{index}_{datensatz_name}", data=data_one_dimension[0, i, :])
                    pos_grenzflaeche_sum += 1
                elif datensatz_name == "ohne":
                    pos_spitze.create_dataset(name=f"{index}_{datensatz_name}", data=data_one_dimension[0, i, :])
                    pos_spitze_sum += 1


def toh5(file_path, save_file):
    save_file = h5py.File(save_file, mode="w")
    pos_data_toh5(file_path, save_file)
    neg_data_toh5(file_path, save_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    toh5("/home/marcus/Dokumente/entladung/data", "rnn_data.h5")

