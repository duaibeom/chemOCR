from math import ceil

import cv2
import numpy as np

from rdkit import Chem

from utils.emnist import PredictAtomChar

b_type_list = {
    3: Chem.BondType.SINGLE,
    4: Chem.BondType.DOUBLE,
    5: Chem.BondType.TRIPLE,
    6: Chem.BondType.SINGLE,
    7: Chem.BondType.SINGLE,
}
b_dir_list = {
    6: Chem.BondDir.BEGINWEDGE,
    7: Chem.BondDir.BEGINDASH,
}

a_list = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "E": 9,
    "R": 9,
    "S": 16,
    "Cl": 17,
}


def sqrt_einsum(a, b):
    a_min_b = a - b
    return np.sqrt(np.einsum("i,i->", a_min_b, a_min_b))


def cal_avg_dist(contours):
    _length = []
    limit = 0
    for _polygon in contours:
        limit += 1
        if limit == 30:
            break
        _length.append(sqrt_einsum(_polygon.max(axis=0)[0], _polygon.min(axis=0)[0]))
    _length = np.array(_length, dtype=np.float16)
    return _length.mean()


# def chk_pair(dist_arr):
#     _, _dist_arr = np.where(dist_arr == 1)
#     if np.all(dist_arr.sum(axis=1) == 2):
#         return _dist_arr.reshape(-1, 2).astype(int)
#     else:
#         plt.imshow(dist_arr)
#         raise ValueError


def get_pair(array1, array2):
    _dist = np.abs(array1[None, :, :] - array2[:, None, :]).max(axis=2)
    _dist.min(axis=0).argsort()
    min_dist = _dist.min(axis=0)
    min_index = min_dist.argsort()
    limit_idx = min_index[:2]
    # print(min_dist[limit_idx[1]])
    if min_dist[limit_idx[1]] > 6.7:
        return
    return limit_idx


def extreme_points(contour):
    extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
    extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
    return extreme_left, extreme_right, extreme_top, extreme_bottom


def get_min_max_polygon(_polygon):
    max_x, max_y = _polygon.max(axis=0)[0]
    min_x, min_y = _polygon.min(axis=0)[0]
    _max = [ceil(max_x) + 3, ceil(max_y) + 3]
    _min = [int(min_x) - 2, int(min_y) - 2]
    return [_max, _min]


def chunk_char(array, cutoff):
    _dist_arr = np.abs(array[None, :, :] - array[:, None, :]).max(axis=2)
    _dist_arr[np.triu_indices(_dist_arr.shape[0])] = 999
    _dist = np.where(_dist_arr < cutoff * 1.2)
    _dist = np.stack(_dist, axis=1)
    return _dist


def get_mol_conn_info(out, image, char_model):

    contours = {}
    char_pos = []
    for idx in range(1, 3):
        _contours, _ = cv2.findContours(
            255 * np.array(out[0][idx] > 0.2, dtype=np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        _ctrs = []
        if _contours.__len__() == 0:
            contours[idx] = None
            continue
        for _polygon in _contours:
            if _polygon.shape.__len__() > 1:
                rect = cv2.minAreaRect(_polygon)
                (x, y), (w, h), ang = rect
                if w * h > 5:
                    box = cv2.boxPoints(rect)
                    _ctrs.append([x, y])
                    if idx == 2:
                        char_pos.append(get_min_max_polygon(_polygon))
        contours[idx] = np.array(_ctrs, dtype=np.float16)

    pred_char_list, pred_img_char_list = char_model(char_pos, image)
    heavy_atom = np.ones(contours[2].__len__(), dtype=np.uint8)

    b_pair = {}
    polygons = {}
    avg_b_dist = 0
    bond_length = []
    for idx in range(3, 8):
        _contours, _ = cv2.findContours(
            255 * np.array(out[0][idx] > 0.2, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        _ctrs = []
        _poly = []
        if _contours.__len__() == 0:
            contours[idx] = None
            polygons[idx] = None
            continue
        for _polygon in _contours:
            # _polygon = _polygon.squeeze()
            if cv2.contourArea(_polygon) > 1:
                rect = cv2.minAreaRect(_polygon)
                (x, y), (w, h), ang = rect
                if (w * h > 30) and idx > 3:
                    append_func(_ctrs, rect, x, y, w, h, bond_length, _poly)
                elif (w * h > 9) and idx == 3:
                    append_func(_ctrs, rect, x, y, w, h, bond_length, _poly)

        contours[idx] = np.array(_ctrs, dtype=np.float16)
        polygons[idx] = np.array(_poly, dtype=np.float16)

    bond_avg_length = np.array(bond_length, dtype=np.float16).max(axis=1).mean()
    chunked_char = chunk_char(contours[2], bond_avg_length)

    for i in chunked_char:
        a = pred_char_list[i[0]]
        b = pred_char_list[i[1]]
        if a in ["H", "2"]:
            heavy_atom[i[0]] = 0
        elif b in ["H", "2"]:
            heavy_atom[i[1]] = 0

    contours[2] = contours[2][np.where(heavy_atom == 1)]
    pred_heavy_char_list = np.array(pred_char_list)[np.where(heavy_atom == 1)]

    pts = np.concatenate([contours[1], contours[2]], axis=0)

    for idx in range(3, 8):
        if polygons[idx] is not None:
            b_pair[idx] = []
            for _polygon in polygons[idx]:
                _pair_idx = get_pair(pts, _polygon)
                if _pair_idx is not None:
                    b_pair[idx].append(_pair_idx)

    return contours, b_pair, pred_heavy_char_list, pred_char_list, pred_img_char_list


def get_mol(contours, pred_char_list, b_pair):

    mol = Chem.RWMol()

    a_idx = 0
    for _ in contours[1]:
        atom = Chem.Atom(6)
        mol.AddAtom(atom)
        a_idx += 0

    for _ in pred_char_list:
        atom = Chem.Atom(a_list[_])
        mol.AddAtom(atom)
        a_idx += 0

    b_idx = 0
    chiral_b_idx = {}
    for idx in b_pair:
        for i, j in b_pair[idx]:
            i = int(i)
            j = int(j)
            mol.AddBond(i, j, order=b_type_list[idx])
            if idx > 5:
                _bond = mol.GetBondWithIdx(b_idx)
                _bond.SetBondDir(b_dir_list[idx])
                chiral_b_idx[b_idx] = [i, j]
            b_idx += 1

    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf)

    Chem.SanitizeMol(mol)
    # rdDepictor.Compute2DCoords(mol)
    Chem.AssignChiralTypesFromBondDirs(mol)

    if chiral_b_idx.__len__() != 0:
        for _idx in chiral_b_idx:
            _b = mol.GetBondWithIdx(_idx)
            _bDir = _b.GetBondDir()
            _aI = _b.GetBeginAtom()
            _aJ = _b.GetEndAtom()
            type_bool1 = _aI.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
            type_bool2 = _aJ.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
            if type_bool1 and type_bool2:
                j, i = chiral_b_idx[_idx]
                mol.RemoveBond(j, i)
                mol.AddBond(i, j, order=Chem.BondType.SINGLE)
                _b = mol.GetBondWithIdx(b_idx - 1)
                _b.SetBondDir(_bDir)

    Chem.AssignChiralTypesFromBondDirs(mol)

    # Chem.DetectBondStereochemistry(mol)
    # Chem.AssignAtomChiralTagsFromStructure(mol)
    # Chem.AssignStereochemistry(mol)

    Chem.SanitizeMol(mol)
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    return mol, smi


def append_func(_ctrs, rect, x, y, w, h, bond_length, _poly):
    box = cv2.boxPoints(rect)
    _ctrs.append([x, y])
    _poly.append(box)
    bond_length.append([w, h])
