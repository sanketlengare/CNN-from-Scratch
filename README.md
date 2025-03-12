# CNN from Scratch using NumPy for CIFAR-10 Classification

## Overview
This project implements a simple Convolutional Neural Network (CNN) entirely from scratch using Python and NumPy. The goal is to classify images from the CIFAR-10 dataset by manually building core deep learning components (convolution, pooling, ReLU activation, fully-connected layers, and cross-entropy loss). This hands-on project deepens understanding of neural network fundamentals and highlights the challenges of training without high-level libraries.

## Project Structure
- **src/**
  - **data.py:** Loads CIFAR-10 data, normalizes images, and applies custom data augmentation (random cropping, horizontal flip, color jitter).
  - **model.py:** Defines the CNN architecture, including forward/backward propagation and weight update routines (with momentum-based updates).
  - **layers.py:** Implements core layers such as `Conv2D`, `MaxPool2D`, `Dense`, and `ReLU`.
  - **loss.py:** Contains the cross-entropy loss function and its gradient computation.
- **main.py:** The main training script that ties everything together, manages training and validation loops, and prints performance metrics.

## Requirements
- Python 3.x
- NumPy

> **Note:** Training using only NumPy on a CPU (e.g., an M1 Mac) might be slower compared to using GPU-accelerated frameworks.

## How to Run
1. Place the CIFAR-10 dataset (in CIFAR-10 batches format) in the `./data/cifar-10-batches-py/` directory.
2. Run the main training script:
   ```bash
   python main.py
