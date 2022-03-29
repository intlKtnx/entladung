import logging
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import librosa
from enum import Enum

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


def toh5(file_path):
    save_file = h5py.File("labeled_data_512.h5", mode="w")
    grenzflaeche = save_file.create_group("grenzflaeche")
    spitze = save_file.create_group("spitze")
    referenz = save_file.create_group("referenz")
    p = Path(file_path)
    files = p.glob('*.mat')
    i = 0
    top_db = 1
    frame_length = 512
    referenz_samples = 0

    for h5dataset_fp in files:

        i += 1
        with h5py.File(h5dataset_fp.resolve()) as h5_file:
            for gname, group in h5_file.items():
                if True:
                    if isinstance(group, h5py.Group):
                        sample_sum = 0
                        referenz_sum = 0
                        spitze_sum = 0
                        grenzflaeche_sum = 0
                        fig1 = plt.figure()
                        ax1 = fig1.add_subplot(title=f"ton librosa äquivalent {Datensatz(i).name} mit top_db={top_db}")
                        fig2 = plt.figure()
                        ax2 = fig2.add_subplot(title=f"groundtruth librosa {Datensatz(i).name} mit top_db={top_db}")
                        ugly_samples = []
                        for samplename, group2 in tqdm(group.items()):
                            if gname == '#refs#':
                                if not len(group2) == 5:
                                    continue
                                else:
                                    if isinstance(group2[2], np.ndarray):

                                        trimmed, indices = librosa.effects.trim(group2[2],
                                                                                top_db=top_db, frame_length=512, hop_length=64)
                                        length = indices[1] - indices[0]
                                        if length < 1000:
                                            sample_frame = group2[4][indices[0]+100:indices[0] + frame_length + 100]
                                            padded_frame = np.pad(sample_frame, (0, max(0, frame_length - len(sample_frame))))
                                            ax1.plot(range(len(trimmed)), group2[4][indices[0]:indices[1]])

                                            groundtruth_frame = group2[2][indices[0]+100:indices[0] + frame_length + 100]
                                            groundtruth_padded = np.pad(groundtruth_frame,
                                                                        (0, max(0, frame_length - len(groundtruth_frame))))
                                            ax2.plot(range(len(trimmed)), trimmed)
                                            sample_sum += 1

                                            leading_silence = np.array(group2[4][:indices[0]+100])
                                            leading_silence = np.array_split(leading_silence,
                                                                             max(1, np.ceil(len(leading_silence)/frame_length)))
                                            trailing_silence = np.array(group2[4][indices[0]+frame_length:])
                                            trailing_silence = np.array_split(trailing_silence,
                                                                              max(1, np.ceil(len(trailing_silence)/frame_length)))
                                            
                                            if i == 1:
                                                grenzflaeche.create_dataset(f"{samplename}_{i}", data=padded_frame)
                                                grenzflaeche_sum += 1
                                            elif i == 2: 
                                                spitze.create_dataset(f"{samplename}_{i}", data=padded_frame)
                                                spitze_sum += 1
                                            for j, data in enumerate(leading_silence):
                                                if referenz_samples < 3000:
                                                    data = np.pad(data, (0, frame_length-len(data)))
                                                    referenz.create_dataset(name=f"{samplename}_{j}_{i}", data=data)
                                                    referenz_samples += 1
                                                referenz_sum += 1
                                            for k, data in enumerate(trailing_silence):
                                                if referenz_samples < 6000:
                                                    data = np.pad(data, (0, frame_length - len(data)))
                                                    referenz.create_dataset(name=f"{samplename}_0{k}_{i}", data=data)
                                                    referenz_samples += 1
                                                referenz_sum += 1

                        logging.info(sample_sum)
                        logging.info(f"grenzfläche:{grenzflaeche_sum}, spitze:{spitze_sum}, referenz:{referenz_sum}")
                        logging.info(ugly_samples)
                        plt.show()
                        break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    toh5("/home/marcus/Dokumente/entladung/")

