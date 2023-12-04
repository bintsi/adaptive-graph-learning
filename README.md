# Multimodal brain age estimation using interpretable adaptive population-graph learning

## About
This is a Pytorch Lightning implementation for the paper 
[Multimodal brain age estimation using interpretable adaptive population-graph learning](https://arxiv.org/abs/2307.04639)
(MICCAI 2023) by Kyriaki-Margarita Bintsi, Vasileios Baltatzis, Rolandos Alexandros Potamias, Alexander Hammers, and Daniel Rueckert

## Requirements
conda install -c anaconda cmake=3.19    
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch  
pip install pytorch_lightning==1.3.8  
pip install torch-geometric  

## Dataset
The dataset used for this paper is the UK Biobank. Since the data is not public, we cannot share the csv files.
You need to put the csv files in the data folder that is available.
The format that the csvs need to have is the following:
train.csv, val.csv, test.csv

For every csv:
Column 0: eid
Column 1: label (age)
Column 2-22: Non-imaging phenotypes
Column 22-90: Imaging phenotypes

## Training
To train a model for the hyperparameters chosen for the regression task run the following command:
`python train.py`

## Reference
If you find the code useful, pleace cite: 
```
@article{bintsi2023multimodal,
  title={Multimodal brain age estimation using interpretable adaptive population-graph learning},
  author={Bintsi, Kyriaki-Margarita and Baltatzis, Vasileios and Potamias, Rolandos Alexandros and Hammers, Alexander and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2307.04639},
  year={2023}
}
```
