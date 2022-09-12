import numpy as np

import cv2
from PIL import ImageDraw, Image


class DBNetTargets:
    def __init__(self, shrink_ratio=0.6, thr_min=0.3, thr_max=0.7, min_short_size=8):
        super().__init__()
        self.shrink_ratio = shrink_ratio
        # self.thr_min = thr_min
        self.thr_min = round(255 * thr_min)
        # self.thr_max = thr_max
        self.thr_max = round(255 * thr_max)
        self.thr_min_max_diff = self.thr_max - self.thr_min

    def _buffered_polygon_coords(self, polygon, shrink=False):
        area = polygon.area
        peri = polygon.length
        distance = round(area * (1 - self.shrink_ratio * self.shrink_ratio) / peri, 2)
        if shrink:
            distance *= -1

        buffer_polygon = polygon.buffer(distance=distance, cap_style=2, join_style=2)
        return list(buffer_polygon.exterior.coords)

    def _draw_polygon(self, image, polygon_coords, outline, fill, width=1):
        ImageDraw.Draw(image).polygon(
            polygon_coords, outline=outline, fill=fill, width=width
        )

    def get_bond_mask_cv2(self, bond_data, image_size):

        bond_data_masks = bond_data["mask"]
        bond_data_bType = bond_data["dir"]

        bond_mask = []
        bond_buffered_mask = []

        idx = 0
        for _arr in bond_data_masks:

            _dummy = np.zeros(image_size, dtype=np.uint8)
            _dummy_buffered = np.zeros(image_size, dtype=np.uint8)

            contours, _ = cv2.findContours(
                255 * np.array(_arr, dtype=np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contours = np.concatenate(contours, axis=0)
            hull = cv2.convexHull(contours)

            bType = bond_data_bType[idx] + 2

            cv2.fillPoly(_dummy, [hull], bType)  # v7
            # cv2.drawContours(_dummy, [hull], 0, 0, 1)  # v7
            bond_mask.append(np.array(_dummy, dtype=np.uint8))
            cv2.fillPoly(_dummy_buffered, [hull], 1)  # v7
            cv2.drawContours(_dummy_buffered, [hull], 0, 1, 2)  # v7
            bond_buffered_mask.append(np.array(_dummy_buffered, dtype=np.uint8))
            idx += 1

        bond_mask = np.array(bond_mask, dtype=np.uint8)
        bond_buffered_mask = (
            np.array(bond_buffered_mask, dtype=np.uint8).sum(axis=0) == 1
        )
        bond_mask = bond_mask.sum(axis=0) * bond_buffered_mask

        return {
            "mask": bond_mask,
            "buffered_mask": bond_buffered_mask,
        }

    def get_char_mask_cv2(self, char_data, image_size):

        char_data_masks = char_data

        char_mask = []
        char_buffered_mask = []

        idx = 0
        for _arr in char_data_masks:

            _dummy = np.zeros(image_size, dtype=np.uint8)

            contours, _ = cv2.findContours(
                255 * np.array(_arr, dtype=np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contours = np.concatenate(contours, axis=0)
            rect = cv2.minAreaRect(contours)
            box = cv2.boxPoints(rect).astype(np.int32)[:, None, :]
            (x, y), (w, h), ang = rect

            cv2.fillPoly(_dummy, [box], 1)  # v7
            if (w > 5) and (h > 5):
                cv2.drawContours(_dummy, [box], 0, 0, 2)  # v7
            elif (w < 4) or (h < 4):
                cv2.drawContours(_dummy, [box], 0, 1, 2)  # v7
            # if (w < 3) or (h < 3):
            #     cv2.drawContours(_dummy, [box], 0, 1, 2)  # v7

            char_mask.append(np.array(_dummy, dtype=np.uint8))
            cv2.fillPoly(_dummy, [box], 1)
            cv2.drawContours(_dummy, [box], 0, 1, 1)  # v7
            char_buffered_mask.append(np.array(_dummy, dtype=np.uint8))

        char_mask_list = np.array(char_mask, dtype=np.uint8)
        # char_buffered_mask = (
        #     np.array(char_buffered_mask, dtype=np.uint8).sum(axis=0) == 1
        # )
        # char_mask = char_mask_list.sum(axis=0) * char_buffered_mask
        char_buffered_mask = (
            np.array(char_buffered_mask, dtype=np.uint8).sum(axis=0) > 0
        )
        char_mask = char_mask_list.sum(axis=0) > 0

        return {
            "mask": char_mask,
            "buffered_mask": char_buffered_mask,
        }

    def get_non_char_atom_mask_cv2(self, non_char_atom_data, image_size):

        atom_mask = []
        atom_buffered_mask = []

        for _arr in non_char_atom_data:

            _dummy = np.zeros(image_size, dtype=np.uint8)
            _poly = np.array(_arr).round().astype(np.int32)[:, None, :]

            cv2.fillPoly(_dummy, [_poly], 1)  # v7
            atom_buffered_mask.append(np.array(_dummy, dtype=np.uint8))
            # cv2.drawContours(_dummy, [_poly], 0, 0, 1)  # v7
            # cv2.drawContours(_dummy, [_poly], 0, 0, 6)  # v7
            atom_mask.append(np.array(_dummy, dtype=np.uint8))

        atom_mask = np.array(atom_mask, dtype=np.uint8).sum(axis=0)
        atom_buffered_mask = np.array(atom_buffered_mask, dtype=np.uint8).sum(axis=0)

        return {
            "mask": atom_mask,
            "buffered_mask": atom_buffered_mask,
        }

    def generate_targets(self, mol_svg):

        char_atom_masks, bond_data, non_char_atom_polygons = mol_svg.get_mol_mask()
        img_size = mol_svg.image_size

        _bond = self.get_bond_mask_cv2(bond_data, img_size)
        bond_mask = _bond["mask"]
        bond_buffered_mask = _bond["buffered_mask"]

        _char = self.get_char_mask_cv2(char_atom_masks, img_size)
        char_mask = _char["mask"]
        char_buffered_mask = _char["buffered_mask"]

        _atom = self.get_non_char_atom_mask_cv2(non_char_atom_polygons, img_size)
        atom_mask = _atom["mask"]
        atom_buffered_mask = _atom["buffered_mask"]

        shr_map = bond_mask + char_mask + atom_mask
        shr_mask = np.array(shr_map >= 1, dtype=np.uint8)

        _shr_buffered_mask = (
            bond_buffered_mask + char_buffered_mask + atom_buffered_mask
        ) >= 1

        thr_map = np.zeros(img_size, dtype=np.uint8)
        thr_mask = np.zeros(img_size, dtype=np.uint8)

        contours, _ = cv2.findContours(
            255 * np.array(_shr_buffered_mask, dtype=np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(thr_map, contours, -1, 1, 1)
        thr_map = thr_map * self.thr_min_max_diff
        thr_map += self.thr_min

        cv2.fillPoly(thr_mask, contours, 1)
        cv2.drawContours(thr_mask, contours, -1, 1, 2)

        shr_map = np.where(char_mask, 2, shr_map)
        shr_map = np.where(atom_mask, 1, shr_map)
        shr_map = shr_map.astype(np.uint8)

        return {
            "gt_shr": shr_map,
            "gt_shr_mask": shr_mask,
            "gt_thr": thr_map,
            "gt_thr_mask": thr_mask,
        }


## Legacy
#     def generate_targets(self, mol_svg):
#         """Generate the gt targets for DBNet.
#         Args:
#             results (dict): The input result dictionary.
#         Returns:
#             results (dict): The output result dictionary.
#         """

#         char_atom_masks, bond_data, non_char_atom_polygons = mol_svg.get_mol_mask()
#         img_size = mol_svg.image_size

#         dummy_mask = np.zeros(img_size, dtype=np.uint8)

#         shrink_map = Image.fromarray(dummy_mask, mode='L')
#         threshold_map = Image.fromarray(dummy_mask, mode='L')
#         threshold_mask = Image.fromarray(dummy_mask, mode='L')

#         feature_map1 = Image.fromarray(dummy_mask, mode='L')
#         feature_map2 = Image.fromarray(dummy_mask, mode='L')
#         feature_map3 = Image.fromarray(dummy_mask, mode='L')
#         feature_map4 = Image.fromarray(dummy_mask, mode='L')
#         feature_map5 = Image.fromarray(dummy_mask, mode='L')
#         feature_map6 = Image.fromarray(dummy_mask, mode='L')
#         feature_map7 = Image.fromarray(dummy_mask, mode='L')

#         idx = 0
#         for _arr in bond_data['mask']:
#             y, x = np.where(_arr == 1)
#             xy = list(zip(x, y))
#             # polygon = MultiPoint(xy).minimum_rotated_rectangle
#             polygon = MultiPoint(xy).convex_hull
#             if polygon.area <= polygon.length:
#                 if polygon.area == 0:
#                     _ratio = 2
#                 else:
#                     _ratio = polygon.length / polygon.area
#                 # print(f'@ Bond {polygon.area} {polygon.length} {_ratio}')
#                 polygon = polygon.buffer(0.5 * _ratio, cap_style=2)
#                 polygon = MultiPoint(
#                     polygon.exterior.coords).convex_hull

#             thr_p_xy = self._buffered_polygon_coords(polygon)
#             # shr_p_xy = self._buffered_polygon_coords(polygon, shrink=True)

#             bType = bond_data['dir'][idx] + 2

#             p_xy = list(polygon.exterior.coords)
#             # self._draw_polygon(shrink_map, p_xy, 0, bType, 1) # v6
#             self._draw_polygon(shrink_map, p_xy, 0, bType, 1)  # v7
#             self._draw_polygon(thr_map, thr_p_xy, self._thr_value, 0, 2)
#             self._draw_polygon(thr_mask, thr_p_xy, 1, 1, 3)
#             # self._draw_polygon(shrink_mask, thr_p_xy, 1, 1, 3)
#             idx += 1

#         del idx

#         for _arr in char_atom_masks:
#             y, x = np.where(_arr == 1)
#             xy = list(zip(x, y))
#             polygon = MultiPoint(xy).envelope
#             if polygon.area <= polygon.length:
#                 if polygon.area == 0:
#                     _ratio = 2.5
#                 else:
#                     _ratio = polygon.length / polygon.area
#                 # print(f'@ Char {polygon.area} {polygon.length} {_ratio}')
#                 polygon = polygon.buffer(0.9 * _ratio, cap_style=2)
#                 polygon = MultiPoint(polygon.exterior.coords).envelope
#             thr_p_xy = self._buffered_polygon_coords(polygon)
#             p_xy = list(polygon.exterior.coords)
#             # self._draw_polygon(shrink_map, p_xy, 0, 2, 2) # v6
#             self._draw_polygon(shrink_map, p_xy, 0, 2, 1)  # v7
#             self._draw_polygon(thr_map, thr_p_xy, self._thr_value, 0, 2)
#             self._draw_polygon(thr_mask, thr_p_xy, 1, 1, 3)
#             # self._draw_polygon(shrink_mask, thr_p_xy, 1, 1, 3)

#         for coords in non_char_atom_polygons:
#             polygon = Polygon(coords)
#             thr_p_xy = self._buffered_polygon_coords(polygon)
#             p_xy = list(polygon.exterior.coords)
#             # self._draw_polygon(shrink_map, thr_p_xy, 0, 1, 4) # v6
#             self._draw_polygon(shrink_map, thr_p_xy, 0, 1, 2)  # v7
#             self._draw_polygon(thr_map, thr_p_xy, self._thr_value, 0, 2)
#             self._draw_polygon(thr_mask, thr_p_xy, 1, 1, 3)
#             # self._draw_polygon(shrink_mask, thr_p_xy, 1, 1, 3)

#         gt_shr = np.array(shrink_map)
#         # gt_shr_mask = np.array(shrink_mask)
#         gt_thr = np.array(thr_map) + self.thr_min
#         gt_thr_mask = np.array(thr_mask)
#         # gt_thr = np.where(gt_thr_mask == 1, gt_thr,
#         #                   np.zeros_like(gt_thr, dtype=np.uint8))

#         return {
#             'gt_shr': gt_shr,
#             # 'gt_shr_mask': gt_shr_mask,
#             'gt_thr': gt_thr,
#             'gt_thr_mask': gt_thr_mask
#         }


# class EASTTargets:

#     def __init__(self,
#                  shrink_ratio=0.6,
#                  thr_min=0.3,
#                  thr_max=0.7,
#                  min_short_size=8):
#         super().__init__()
#         self.shrink_ratio = shrink_ratio
#         # self.thr_min = thr_min
#         self.thr_min = round(255 * thr_min)
#         # self.thr_max = thr_max
#         self.thr_max = round(255 * thr_max)
#         self._thr_value = self.thr_max - self.thr_min
#         self.min_short_size = min_short_size
#         self._dummy_arr = None
