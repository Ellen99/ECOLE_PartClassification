# Part Classifier

A deep learning project for training part classifiers using DINO features and hierarchical concept learning.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [Technical Details](#-technical-details)
- [Results](#results)

## 🔍 Overview

This project implements a part classification system using DINOv2 (self-DIstillation with NO labels) features. It trains logistic regression classifiers for different parts of concepts using a hierarchical approach, with support for parallel processing and checkpointing.

## ✨ Features

- Facebook's DINOv2 vision transformer for image processing
- Generate semantic part masks for new images
- Parallel training with multi-threading
- Automated checkpoint management
- Configurable training parameters
- Result visualization with both patch mask and elliptical mask overlays
- Support for both CPU and GPU processing

## 📁 Repository Structure

```
SOON
```

## 🚀 Installation

1. Clone the repository:
```
SOON
```

2. Create a virtual environment:
```
SOON
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage


See  [prediction](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/notebooks/prediction_demo.ipynb) and [training](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/notebooks/training_demo.ipynb) notebooks for described steps and details on how to reproduce the results


### Command Line Training

```bash
python scripts/train.py --config configs/default_training_config.yaml
```

### Command Line prediction

```bash
python scripts/predict.py --config configs/default_pred_config.yaml
```

## ⚙️ Configuration

Configuration is managed through the `TrainingConfig` and `InferenceConfig` classes


## 🔧 Technical Details for Prediction
- **Feature Extractor**: Facebook's DINOv2 Vision Transformer
- **Classifier**: Logistic Regression with L1/L2 regularization
- **Input**: RGB images + concept label
- **Output**: Part-wise segmentation masks

## 📊 Results

![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--agricultural-airplanes--agricultural-1.png.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--agricultural-airplanes--agricultural-2.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--agricultural-airplanes--agricultural-3.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--agricultural-airplanes--agricultural-4.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--attack-airplanes--attack-1.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--attack-airplanes--attack-2.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_kitchen--air%20fryer-kitchen--air%20fryer-1.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_kitchen--pizza%20cutter-kitchen--pizza%20cutter-1.jpg.png)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_vehicles--hatch%20back-vehicles--hatch%20back-1.jpg.png)


