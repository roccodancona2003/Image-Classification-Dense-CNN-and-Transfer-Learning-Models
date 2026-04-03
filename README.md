# Machine Learning Project - Rocco D'Ancona

## Project Overview
This project explores different machine learning and deep learning architectures for image classification. It evaluates the performance of a Feed-Forward Dense model, a Convolutional Neural Network (CNN), and Transfer Learning models on datasets with varying resolutions.

## Datasets
* **CIFAR-10**: Contains 60,000 images divided into 10 categories. The training set consists of 50,000 images, and the test set has 10,000 images. The images have a low resolution of 32x32x3 (RGB).
* **Cats vs Dogs**: A higher-resolution dataset loaded via `tensorflow_datasets` containing a total of 23,262 images. Because the original images have varying dimensions, they were resized to 256x256 during preprocessing.

## Models and Results

### 1. Feed-Forward Dense Model (CIFAR-10)
* **Architecture**: Uses a `Flatten` layer followed by strongly connected `Dense` layers.
* **Results**: Reached an accuracy of about 50% and a loss of 1.4.
* **Insights**: The result is not optimal because flattening the 3D images into an array of 3072 values fails to capture spatial information and specific patterns.

### 2. Convolutional Neural Network (CIFAR-10)
* **Results**: Achieved an accuracy of 76% and a loss of 0.7.
* **Insights**: Outperforms the feed-forward model because convolutions successfully capture significant spatial patterns and relationships between neighboring pixels without needing an initial flattening step.

### 3. Transfer Learning with EfficientNetB0 (CIFAR-10)
* **Architecture**: Uses the pre-trained `EfficientNetB0` model with frozen "imagenet" weights, replacing the top classification layers.
* **Results**: Very low performance, with an accuracy around 15.2% and a high loss of 2.24 on the test set.
* **Insights**: The model suffers from clear overfitting, memorizing training data while failing to generalize. The pre-trained model is unsuited for this task primarily due to the poor resolution of the CIFAR-10 images.

### 4. Transfer Learning with EfficientNetB0 (Cats vs Dogs)
* **Architecture**: Reuses the frozen `EfficientNetB0` base model applied to the high-resolution dataset, adding `GlobalAveragePooling2D`, `Dense`, and `Dropout` layers to classify 2 classes.
* **Results**: Achieved excellent results with 99.01% test accuracy and a loss of 0.0274.
* **Insights**: The training and validation curves remain very close to each other, indicating that the model generalizes very well on high-resolution data.

## Libraries Used
* `numpy`
* `matplotlib.pyplot`
* `tensorflow` & `tensorflow.keras`
* `tensorflow_datasets`
