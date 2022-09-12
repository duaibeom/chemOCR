import cv2
import numpy as np

import streamlit as st

import torch

# from torchvision import transforms

from model.utils.transfroms import get_test_transform
from utils.parser import get_mol_conn_info, get_mol

from PIL import Image
from datetime import datetime

from rdkit.Chem import Draw


def save_image(img, pred, idx):
    now = datetime.now()
    cur_time_str = now.strftime("%d%m%Y_%H%M%S")

    # img = np.array(img * 255, dtype=np.uint8)
    # pil_image = Image.fromarray(img, mode="L")
    pil_image = Image.fromarray(img)
    pil_image.save(f"tmp_img/{cur_time_str}_{idx}_{pred}.png")


def denormalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ten = x.clone().detach()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(ten, 0, 1).mul_(255)


def load_structure_model():
    from model.dbnet import DBNet

    model = DBNet(
        # inner_channels=128,
        # out_channels=64,
        # head_in_channels=320,
        inner_channels=192,
        out_channels=96,
        head_in_channels=480,
        test=True,
    )

    model.load_state_dict(torch.load("model_weights.mbv3s.final.pth"))
    model.eval()

    return model


def load_emnist_char_model():
    from utils.emnist import PredictAtomChar

    def rule_func(pred):
        if pred in ["0", "D", "Q"]:
            pred = "O"
        elif pred in ["n"]:
            pred = "N"
        elif pred in ["z", "Z"]:
            pred = "2"
        elif pred in ["a"]:
            pred = "Cl"
        # elif pred in ["E", "t"]:
        #     pred = "F"
        elif pred in ["5"]:
            pred = "S"
        return pred

    model = PredictAtomChar(return_img=True, rule_func=rule_func)

    return model


def load_mmocr_char_model():
    from utils.mmocr_infer import MMOCRInferCRNN

    def rule_func(pred):
        if pred in ["0", "o"]:
            pred = "O"
        elif pred in ["n"]:
            pred = "N"
        # elif pred in ["z", "Z"]:
        #     pred = "2"
        elif pred in ["ci"]:
            pred = "Cl"
        elif pred in ["f"]:
            pred = "F"
        elif pred in ["s"]:
            pred = "S"
        elif pred in ["h"]:
            pred = "H"
        return pred

    model = MMOCRInferCRNN(return_img=True, rule_func=rule_func)

    return model


def upload_image():
    """Uoload the image
    Args:
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    # file = st.sidebar.file_uploader(
    file = st.file_uploader(
        "Upload your image (jpg, jpeg, or png)", ["jpg", "jpeg", "png"]
    )
    if file is not None:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    return None


def decode_image_string(image_string):
    image = cv2.imdecode(np.frombuffer(image_string, np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_mol(mol):
    d2d = Draw.MolDraw2DCairo(-1, -1)
    d2d.drawOptions().scalingFactor = 20  # units are roughly pixels/angstrom
    d2d.drawOptions().fixedFontSize = 14
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return decode_image_string(d2d.GetDrawingText())


def predict_mol(image, model, _transforms, char_model):

    # image = image.astype(np.float32) / 255

    image = _transforms(image=image)["image"]
    image = image[
        None,
    ]

    neck_out = model.neck(image)
    out = model.head(neck_out)
    out = out.detach().cpu().numpy()

    try:
        (
            contours,
            b_pair,
            pred_heavy_char_list,
            pred_char_list,
            pred_img_char_list,
        ) = get_mol_conn_info(out, image, char_model)
        return get_mol(contours, pred_heavy_char_list, b_pair)
    except:
        for idx, img in enumerate(pred_img_char_list):
            save_image(img, pred_char_list[idx], idx)
        raise ValueError


def main():
    st.title("ChemOCR (OCSR)")

    model = load_structure_model()
    char_model = load_mmocr_char_model()

    _transforms = get_test_transform()

    input_image = upload_image()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Input image")

        if input_image is not None:
            st.image(input_image)
        else:
            st.write("INPUT Image")

    with col2:
        st.header("Predicted image")

        if input_image is not None:
            mol, smi = predict_mol(input_image, model, _transforms, char_model)
            st.image(draw_mol(mol))
            st.write(f"`{smi}`")
        else:
            st.write("OUTPUT Results")


if __name__ == "__main__":
    main()
