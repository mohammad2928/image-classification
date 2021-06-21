# Image classification

Simple image classification models

## Installation

```
pip install -r requierments.py
```

## Usage

1. Clone the codes; cd image-classification
2. Config your parameters in the src/parameters.py module
3. cd src 
4. Now you can run some of the following commands for running
  1. python LeNet-with-mnist.py (running LeNet over mnist dataset)
  2. python LeNet-with-rps.py (running LeNet over rps dataset)
  3. python MLP-with-mnist.py (running MLP over mnist dataset)

## Data-sets

1. [mnist](https://keras.io/api/datasets/fashion_mnist/) (it will be downloaded by keras)
2. [rps](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip) (download rps data ind extract it into a folder by the name data so the data will be as follow ./data/rps) 

## Models

The following models are used. 

### LeNet

INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

### MLP

Contains two fully connected layers 
