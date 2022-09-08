import cv2
import numpy as np

import streamlit as st

import torch

# from torchvision import transforms

from model.dbnet import DBNet
from model.utils.transfroms import get_test_transform
from utils.parser import get_mol_conn_info, get_mol

from PIL import Image
from datetime import datetime

from rdkit.Chem import Draw


def save_image(img, pred, idx):
    now = datetime.now()
    cur_time_str = now.strftime("%d%m%Y_%H%M%S")

    img = np.array(img * 255, dtype=np.uint8)
    pil_image = Image.fromarray(img)
    pil_image.save(f"tmp_img/{cur_time_str}_{idx}_{pred}.png")


def load_model():
    model = DBNet(
        inner_channels=128,
        out_channels=64,
        head_in_channels=320,
        test=True,
    )

    # model.load_state_dict(torch.load("model_weights.v9.mbv3s.final.pth"), strict=False)
    model.load_state_dict(torch.load("model_weights.v9_rgb.mbv3s.5n192h480.final.pth"))
    model.eval()
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
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    return None


def decode_image_string(image_string):
    image = cv2.imdecode(np.fromstring(image_string, np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_mol(mol):
    d2d = Draw.MolDraw2DCairo(-1, -1)
    d2d.drawOptions().scalingFactor = 20  # units are roughly pixels/angstrom
    d2d.drawOptions().fixedFontSize = 14
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return decode_image_string(d2d.GetDrawingText())


def predict_mol(image, model, _transforms):

    image = image.astype(np.float32) / 255

    image = _transforms(image=image)["image"]
    image = image[
        None,
    ]

    neck_out = model.neck(image)
    out = model.head(neck_out)
    out = out.detach().cpu().numpy()

    (
        contours,
        b_pair,
        pred_heavy_char_list,
        pred_char_list,
        pred_img_char_list,
    ) = get_mol_conn_info(out, image)

    for idx, img in enumerate(pred_img_char_list):
        save_image(img, pred_char_list[idx], idx)

    return get_mol(contours, pred_heavy_char_list, b_pair)


def main():
    st.title("ChemOCR (OCSR)")

    model = load_model()

    _transforms = get_test_transform()

    input_image = upload_image()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Input image")

        if input_image is not None:
            st.image(input_image)
        else:
            st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.header("Predicted image")

        if input_image is not None:
            mol, smi = predict_mol(input_image, model, _transforms)
            st.image(draw_mol(mol))
            st.write(f"`{smi}`")
        else:
            st.image("https://static.streamlit.io/examples/dog.jpg")


if __name__ == "__main__":
    main()
