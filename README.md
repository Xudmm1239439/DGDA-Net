# DGDA-Net
# Dual-branch Graph Domain Adaptation for Cross-scenario Multi-modal Emotion Recognition

This is an official implementation of 'Dual-branch Graph Domain Adaptation for Cross-scenario Multi-modal Emotion Recognition' :fire:.

<div  align="center"> 
  <img src="https://github.com/Xudmm1239439/DGDA-Net/blob/main/DGDA-Net.png" width=100% />
</div>



## 🚀 Installation

```bash
aiohappyeyeballs    2.4.4
aiohttp             3.10.11
aiosignal           1.3.1
async-timeout       5.0.1
attrs               25.1.0
certifi             2022.12.7
charset-normalizer  2.1.1
cmake               3.25.0
contourpy           1.1.1
cycler              0.12.1
filelock            3.13.1
fonttools           4.56.0
frozenlist          1.5.0
fsspec              2025.3.0
idna                3.4
importlib_resources 6.4.5
Jinja2              3.1.4
joblib              1.4.2
kiwisolver          1.4.7
lit                 15.0.7
MarkupSafe          2.1.5
matplotlib          3.7.1
mpmath              1.3.0
multidict           6.1.0
networkx            3.0
numpy               1.24.3
packaging           24.2
pandas              1.5.3
pillow              10.2.0
pip                 24.2
propcache           0.2.0
protobuf            5.29.3
psutil              7.0.0
pyg-lib             0.4.0+pt20cu118
pyparsing           3.1.4
python-dateutil     2.9.0.post0
pytz                2025.1
requests            2.28.1
scikit-learn        1.2.2
scipy               1.10.1
seaborn             0.13.2
setuptools          75.1.0
six                 1.17.0
sympy               1.13.1
tensorboardX        2.6.2.2
threadpoolctl       3.5.0
torch               2.0.0+cu118
torch-cluster       1.6.3+pt20cu118
torch-geometric     2.6.1
torch-scatter       2.1.2+pt20cu118
torch-sparse        0.6.18+pt20cu118
torch-spline-conv   1.2.2+pt20cu118
torchaudio          2.0.1+cu118
torchvision         0.15.1+cu118
tqdm                4.67.1
triton              2.0.0
typing_extensions   4.12.2
urllib3             1.26.13
wheel               0.44.0
yarl                1.15.2
zipp                3.20.2
```

## pre_training.py
```bash
python main.py
```
- Download the preprocessed data from [here](https://pan.baidu.com/s/1WB0tWbznju6r-nz3LMw_Bg?pwd=becf), and put them into `data/`.

## Training
```bash
python test.py
```
## Acknowledgements
- Special thanks to the [COSMIC](https://github.com/declare-lab/conv-emotion) and [MMGCN](https://github.com/hujingwen6666/MMGCN) for sharing their codes and datasets.
