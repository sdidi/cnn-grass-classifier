
# Grass Deep Learning Classifier

This project demonstrates a deep learning model built for classifying images into two categories: "Grass" and "Not Grass". The model utilizes a pre-trained MobileNetV2 model (not trained in this domain) then fine-tuned for this specific task.

## Overview

The current implementation uses a very small dataset (approximately 32 images) for training, which is not ideal for a robust model. The goal of this project is to train a deep learning model that can accurately classify images of grass versus non-grass objects. 

### Current Status
- **Model**: The model is based on the MobileNetV2 architecture, which is pre-trained on the ImageNet dataset. The model has been fine-tuned for binary classification (grass vs. non-grass).
- **Dataset**: The current dataset contains only about 32 images, which is insufficient for training a highly accurate model. Future iterations will include more data for better accuracy.

### Next Steps
- **Data Augmentation**: To improve generalization, we will add data augmentation techniques (like rotation, zoom, and shear).
- **Increase Dataset Size**: The dataset will be expanded to at least 500 images, with an equal balance between grass and non-grass images. 
- **Fine-Tuning**: Further fine-tuning will be performed to optimize the model's accuracy.
- **Validation**: The model will be validated on a separate test dataset to ensure it performs well on unseen images.

## Requirements

To run the code, ensure that you have the following dependencies installed:

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Matplotlib**
- **scikit-learn** (optional, for evaluation metrics like classification report)

You can install the required dependencies via pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
