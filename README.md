<h1 align="center">
ChemOCR(OCSR)
</h1>

<h3 align="center">
DB(Differentiable Binarization)-based Optical Chemical Structure Recognition  
</h3> 

<img src="data/demo_image.png" align="center">

---

```sh
streamlit run demo.py
```

### Backbone

- Swin
- ResNet
- MobileNetV3

## Parser
**Rule-based method**

## DATA

- ChEMBL: https://www.ebi.ac.uk/chembl/

### Training data sample
<img src="data/sample_data.png">

### Limitation
- Non-charged atom
- (Inner bridged) Complex ring is not possible.
- Specific resolution of image
- Small text atoms (like Iodin) are not recognized.

### TODO
- [x] Character recognition (pretrained model from MMOCR RCNN)
- [x] Web front-end (streamlit)
- [ ] Bond direction analysis

## References

1. https://github.com/MhLiao/DB
2. https://github.com/open-mmlab/mmocr
3. https://github.com/rdkit/rdkit
