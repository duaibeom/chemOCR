import numpy as np
import pyvips

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdCoordGen

# from utils.utils import timer

dict_bondDir = {_value: _key for _key, _value in Chem.rdchem.BondDir.values.items()}
dict_bondType = {_value: _key for _key, _value in Chem.rdchem.BondType.values.items()}
dict_chiralType = {
    _value: _key for _key, _value in Chem.rdchem.ChiralType.values.items()
}

# rdDepictor.SetPreferCoordGen(True)
cgparams = rdCoordGen.CoordGenParams()
precision_list = [
    cgparams.sketcherQuickPrecision,
    cgparams.sketcherCoarsePrecision,
    cgparams.sketcherStandardPrecision,
    cgparams.sketcherBestPrecision,
]


class MolSVG:
    def __init__(
        self,
        mol: str or Chem.rdchem.Mol,
        precision: int = 0,
        comic: bool = False,
        bondLineWidth: int = 2,
        scalingFactor: int = 20,
        padding: float = 0.05,
        additionalAtomLabelPadding: float = 0.0,
        multipleBondOffset: float = 0.15,
        fixedFontSize: int = -1,
        gray: bool = False,
        mono_color_image=True,
    ) -> None:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        self.comic = comic
        self.mol = mol
        self.draw_ops = self.get_draw_options(
            bondLineWidth = bondLineWidth,
            scalingFactor = scalingFactor,
            padding = padding,
            additionalAtomLabelPadding = additionalAtomLabelPadding,
            multipleBondOffset = multipleBondOffset,
            fixedFontSize = fixedFontSize,
            gray = gray,
            )
        self.d2d = self.get_d2d(precision)
        self.svg_raw = self.d2d.GetDrawingText()

        _img = pyvips.Image.svgload_buffer(self.svg_raw.encode())
        if mono_color_image:
            _img = _img.colourspace("b-w").numpy()[:, :, 0]

        self.image = _img
        self.image_size = tuple(_img.shape[:2])

    def get_mol_mask(self):

        atom_comps, atom_chars, bond_comps = self.decompose_svg()

        atom_arr = self.get_atom_info()
        char_atoms_list = atom_comps.keys()

        scaleFactor = self.draw_ops.scalingFactor * (1 + 2 * self.draw_ops.padding)

        atom_size = scaleFactor * 0.18

        ## BBOX
        atom_bboxes = self.gen_atom_bbox(atom_arr, atom_size)
        atom_bboxes = [
            _ for idx, _ in enumerate(atom_bboxes) if (idx not in char_atoms_list)
        ]

        char_masks = self._get_mask(atom_chars)
        # char_bboxes = self.get_char_bbox(atom_chars)
        bond_data = {}
        bond_data["mask"] = self._get_mask(bond_comps["svg"].values())
        bond_data["dir"] = self.get_bond_dir(bond_comps)

        return char_masks, bond_data, atom_bboxes

    def get_bond_dir(self, bond_comps):
        _dir_list = []
        bond_info = self.get_bond_info()
        for idx in range(bond_comps["cnt"].__len__()):
            try:
                bInfo = bond_info[idx]
                bType = bInfo[0]
                if bType == 1:
                    if bond_comps["fill"][idx] == 1:
                        bType = 4
                    elif bond_comps["cnt"][idx] > 1:
                        bType = 5
                elif bType in (2, 3):
                    pass
                elif bType == 12:
                    bType = bond_comps["cnt"][idx]
                else:
                    raise ValueError("Unknown BondType")
            except IndexError:
                bCnt = bond_comps["cnt"][idx]
                if bCnt == 1:
                    bType = 4
                elif bCnt > 1:
                    bType = 5
                else:
                    raise PermissionError("cnt is zero !")
            _dir_list.append(bType)
        return _dir_list

    def gen_atom_bbox(self, atom_array, size_ratio):
        atom_bbox = []

        for atom_vec in atom_array:
            _aCx = atom_vec[0]
            _aCy = atom_vec[1]
            _aHx = size_ratio
            _aHy = size_ratio
            _atom_box = [
                (_aCx - _aHx, _aCy + _aHy),
                (_aCx + _aHx, _aCy + _aHy),
                (_aCx + _aHx, _aCy - _aHy),
                (_aCx - _aHx, _aCy - _aHy),
            ]
            atom_bbox.append(_atom_box)

        return atom_bbox

    def get_draw_options(
        self,
        bondLineWidth: int = 2,
        scalingFactor: int = 20,
        padding: float = 0.05,
        additionalAtomLabelPadding: float = 0.0,
        multipleBondOffset: float = 0.15,
        fixedFontSize: int = -1,
        gray: bool = False,
    ) -> rdMolDraw2D.MolDrawOptions:
        _d2d_ops = rdMolDraw2D.MolDrawOptions()
        # _d2d_ops.useAvalonAtomPalette()
        _d2d_ops.bondLineWidth = bondLineWidth  # 1 ~ 3
        _d2d_ops.fixedFontSize = fixedFontSize
        _d2d_ops.scalingFactor = scalingFactor  # 20 ~ 30
        _d2d_ops.padding = padding
        _d2d_ops.additionalAtomLabelPadding = additionalAtomLabelPadding  # ~ 0.17
        _d2d_ops.multipleBondOffset = multipleBondOffset  # (0.08 ~ 0.22)  default: 0.15
        # _d2d_ops.fontFile()

        if gray:
            _d2d_ops.useBWAtomPalette()

        return _d2d_ops

    # @timer
    def get_d2d(self, precision: int = 1):
        _mol = rdMolDraw2D.PrepareMolForDrawing(self.mol)

        _d2d = Draw.MolDraw2DSVG(-1, -1)
        # if type == 'img':
        #     # ref: rdkit/Chem/Draw/__init__.py
        #     _d2d = Draw.MolDraw2DCairo(-1, -1)
        # elif type == 'svg':
        #     _d2d = Draw.MolDraw2DSVG(-1, -1)
        # else:
        #     raise ValueError(f"{type} type is not allowed.")

        cgparams.minimizerPrecision = precision_list[precision]
        rdCoordGen.AddCoords(_mol, cgparams)

        _d2d.SetDrawOptions(self.draw_ops)

        if self.comic:
            Draw.SetComicMode(_d2d.drawOptions())

        _d2d.DrawMolecule(_mol)
        _d2d.FinishDrawing()
        return _d2d

    def get_atom_info(self):
        atoms = self.mol.GetAtoms()

        _atom_info = []

        idx = 0
        for atom in atoms:
            _coords = self.d2d.GetDrawCoords(idx)
            # _atmIdx = atom.GetIdx()
            _atmId = atom.GetAtomicNum()
            _nHs = atom.GetTotalNumHs()
            _x = _coords.x.__round__(3)
            _y = _coords.y.__round__(3)
            # print(_x, _y, _atmId, _nHs)
            _atom_info.append([_x, _y, _atmId, _nHs])
            idx += 1

        return np.array(_atom_info)

    def get_bond_info(self):

        bonds = self.mol.GetBonds()

        _bond_info = []

        # for idx, bond in enumerate(bonds):
        for bond in bonds:
            # _bondRing = bond.IsInRing()
            _bondDir = dict_bondDir[bond.GetBondDir()]
            _bondType = dict_bondType[bond.GetBondType()]
            # _bondBegin = bond.GetBeginAtomIdx()
            # _bondEnd = bond.GetEndAtomIdx()
            # _beginAtom = self.mol.GetAtomWithIdx(_bondBegin)
            # _beginAtomChiral = dict_chiralType[_beginAtom.GetChiralTag()]
            # _endAtom = self.mol.GetAtomWithIdx(_bondEnd)
            # _endAtomChiral = dict_chiralType[_endAtom.GetChiralTag()]
            # print(_bondBegin, _bondEnd, _bondRing, _bondType, _bondDir)
            _bond_info.append((_bondType, _bondDir))

        return _bond_info

    def decompose_svg(self):

        from xml.dom import expatbuilder

        # from copy import deepcopy
        import re

        _expat_xml = expatbuilder.parseString(self.svg_raw)

        bond_child = {}
        bond_child_cnt = {}
        bond_child_fill = {}
        atom_child = {}

        for child_node in _expat_xml.documentElement.childNodes[5:]:

            if child_node.nodeName == "path":
                _attr = child_node.attributes.items()
                parsed_class = re.split(" |-", _attr[0][1])
                _id = int(parsed_class[1])
                _type = parsed_class[0]
                _hasFill = re.match("^fill:#000000", _attr[2][1])

                if _type == "bond":
                    try:
                        bond_child[_id].append(child_node)
                        bond_child_cnt[_id] += 1
                    except:
                        bond_child[_id] = [child_node]
                        bond_child_cnt[_id] = 1
                        bond_child_fill[_id] = 0
                    if _hasFill is not None:
                        bond_child_fill[_id] += 1
                elif _type == "atom":
                    try:
                        atom_child[_id].append(child_node)
                    except:
                        atom_child[_id] = [child_node]
                else:
                    raise PermissionError

            _expat_xml.documentElement.removeChild(child_node)

        def _get_mol_comp(child: dict):
            for _ in child:
                _expat_xml.documentElement.appendChild(_)
            _comp_xml = _expat_xml.toxml()  # text
            # _comp_xml = deepcopy(_expat_xml)  # binary
            for _ in child:
                _expat_xml.documentElement.removeChild(_)
            return _comp_xml

        atom_svg_comp = {_id: _get_mol_comp(atom_child[_id]) for _id in atom_child}
        bond_svg_comp = {_id: _get_mol_comp(bond_child[_id]) for _id in bond_child}
        char_svg_comp = [
            _get_mol_comp([atom_char])
            for _id in atom_child
            for atom_char in atom_child[_id]
        ]
        bond_comp = {}
        bond_comp["svg"] = bond_svg_comp
        bond_comp["cnt"] = bond_child_cnt
        bond_comp["fill"] = bond_child_fill

        return atom_svg_comp, char_svg_comp, bond_comp

    def _get_mask(self, svg_comp):

        _svg_mask = []

        for _comp in svg_comp:
            _img = pyvips.Image.svgload_buffer(_comp.encode()).colourspace("b-w")
            _array = np.array(_img)[:, :, 0]
            _array = _array < 255
            _svg_mask.append(_array)

        return _svg_mask

    def get_char_bbox(self, atom_svg_char):

        _atom_svg_char_bboxes = []

        for _atom_svg_char in atom_svg_char:
            _atom_char_img = pyvips.Image.svgload_buffer(
                _atom_svg_char.encode()
            ).colourspace("b-w")
            _atom_char_array = np.array(_atom_char_img)[:, :, 0]

            y, x = np.where(_atom_char_array != 255)
            xmin = np.min(x) - 1
            xmax = np.max(x) + 1
            ymin = np.min(y) - 1
            ymax = np.max(y) + 1

            _atom_svg_char_bboxes.append(
                [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
            )

        return _atom_svg_char_bboxes
