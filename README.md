# Fast and Lightweight Vision-Language Model for Adversarial Traffic Sign Detection
Official repository for ViLAS (Vision-Language Model for Adversarial Traffic Sign Detection) as demonstrated in the paper "Fast and Lightweight Vision-Language Model for Adversarial Traffic Sign Detection".

In this official implementation, we demonstrate the calculation of detection score proposed in ViLAS. Detection scores are calculated for both clean and attacked images. We use PGD as the adversarial attack. The expected behavior is that images will have low scores, while attacked images will have high scores. More details can be found in our [paper](https://www.mdpi.com/2079-9292/13/11/2172).


## Installation

To install the required packages:

```
pip install -r requirements.txt
```

## Pre-trained Models

Pre-trained ViT and VLM models can be downloaded from this [link](https://drive.google.com/file/d/1wqtrKffn3CQ-cIPgjpXev8ko_YXvzMBM/view). 


## Usage

```vilas.py``` function that calculates the detection score for a given image.

```run_vilas.py``` reports the detection scores for a given image's clean and attacked versions.

To calculate and report ViLAS scores on all provided images for the clean and attacked versions:

```
python run_vilas_all.py
```

## Cite

```
@article{mumcu2024fast,
  title={Fast and Lightweight Vision-Language Model for Adversarial Traffic Sign Detection},
  author={Mumcu, Furkan and Yilmaz, Yasin},
  journal={Electronics},
  volume={13},
  number={11},
  pages={2172},
  year={2024},
  publisher={MDPI}
}
```
