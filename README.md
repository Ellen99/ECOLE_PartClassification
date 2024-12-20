# Part Classifier

A hierarchical part classification system using DINOv2 features for semantic part segmentation.

<!-- 
## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [Technical Details](#-technical-details)
- [Results](#results) -->

## 🔍 Overview

This project implements a part classification system using DINOv2 (self-DIstillation with NO labels) features. It trains logistic regression classifiers for different parts of concepts using a hierarchical approach, with support for parallel processing and checkpointing.

Refer to [concept hierarchy file here!](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/Concept_hierarchy-Nov27.csv)

## ✨ Features

- Facebook's DINOv2 vision transformer for image processing
- Generate semantic part masks for new images
- Parallel training with multi-threading
- Automated checkpoint management
- Configurable training parameters
- Result visualization with both patch mask and elliptical mask overlays
- Support for both CPU and GPU processing
<!-- 
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
``` -->

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

Configuration is managed through the [`TrainingConfig`](https://github.com/Ellen99/ECOLE_PartClassification/blob/8d7a889b42a39c22a0b65a7f56b625e667e29c35/src/utils/config.py#L41) and [`InferenceConfig`](https://github.com/Ellen99/ECOLE_PartClassification/blob/8d7a889b42a39c22a0b65a7f56b625e667e29c35/src/utils/config.py#L41) classes


## 🔧 Technical Details for Prediction
- **Feature Extractor**: Facebook's DINOv2 Vision Transformer
- **Classifier**: Logistic Regression with L1/L2 regularization
- **Input**: RGB images + concept label
- **Output**: Part-wise segmentation masks

## 📊 Results

![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--attack-1.jpg)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_vehicles--suv-1.jpg)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_kitchen--espresso%20machine-2.jpg)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--agricultural-3.jpg)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--agricultural-4.jpg)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_vehicles--suv-2.jpg)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--private-10.JPG)
![example output](https://github.com/Ellen99/ECOLE_PartClassification/blob/main/output/masks_airplanes--private-2.png)