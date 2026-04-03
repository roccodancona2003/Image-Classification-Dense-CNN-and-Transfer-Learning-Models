# [cite_start]Machine Learning Project - Rocco D'Ancona 

## Project Overview
[cite_start]This project explores different machine learning and deep learning architectures for image classification[cite: 214, 220, 231]. [cite_start]It evaluates the performance of a Feed-Forward Dense model, a Convolutional Neural Network (CNN), and Transfer Learning models on datasets with varying resolutions[cite: 3, 220, 231, 444].

## Datasets
* [cite_start]**CIFAR-10**: Contains 60,000 images divided into 10 categories[cite: 27]. [cite_start]The training set consists of 50,000 images, and the test set has 10,000 images[cite: 28, 29]. [cite_start]The images have a low resolution of 32x32x3 (RGB)[cite: 30, 31].
* [cite_start]**Cats vs Dogs**: A higher-resolution dataset loaded via `tensorflow_datasets` containing a total of 23,262 images[cite: 446, 463]. [cite_start]Because the original images have varying dimensions, they were resized to 256x256 during preprocessing[cite: 464, 466].

## Models and Results

### [cite_start]1. Feed-Forward Dense Model (CIFAR-10) [cite: 3, 214]
* [cite_start]**Architecture**: Uses a `Flatten` layer followed by strongly connected `Dense` layers[cite: 85, 87, 119]. 
* [cite_start]**Results**: Reached an accuracy of about 50% and a loss of 1.4[cite: 216].
* [cite_start]**Insights**: The result is not optimal because flattening the 3D images into an array of 3072 values fails to capture spatial information and specific patterns[cite: 218].

### [cite_start]2. Convolutional Neural Network (CIFAR-10) [cite: 220]
* [cite_start]**Results**: Achieved an accuracy of 76% and a loss of 0.7[cite: 224].
* [cite_start]**Insights**: Outperforms the feed-forward model because convolutions successfully capture significant spatial patterns and relationships between neighboring pixels without needing an initial flattening step[cite: 224]. 

### [cite_start]3. Transfer Learning with EfficientNetB0 (CIFAR-10) [cite: 231, 233, 436]
* [cite_start]**Architecture**: Uses the pre-trained `EfficientNetB0` model with frozen "imagenet" weights, replacing the top classification layers[cite: 233, 235, 238, 239].
* [cite_start]**Results**: Very low performance, with an accuracy around 15.2% and a high loss of 2.24 on the test set[cite: 323, 324, 380]. 
* [cite_start]**Insights**: The model suffers from clear overfitting, memorizing training data while failing to generalize[cite: 374]. [cite_start]The pre-trained model is unsuited for this task primarily due to the poor resolution of the CIFAR-10 images[cite: 346, 437].

### [cite_start]4. Transfer Learning with EfficientNetB0 (Cats vs Dogs) [cite: 443, 444, 445]
* [cite_start]**Architecture**: Reuses the frozen `EfficientNetB0` base model applied to the high-resolution dataset, adding `GlobalAveragePooling2D`, `Dense`, and `Dropout` layers to classify 2 classes[cite: 507, 508, 523, 524, 525, 528].
* [cite_start]**Results**: Achieved excellent results with 99.01% test accuracy and a loss of 0.0274[cite: 624].
* [cite_start]**Insights**: The training and validation curves remain very close to each other, indicating that the model generalizes very well on high-resolution data[cite: 635, 637].

## Libraries Used
* [cite_start]`numpy` [cite: 4]
* [cite_start]`matplotlib.pyplot` [cite: 5]
* [cite_start]`tensorflow` & `tensorflow.keras` [cite: 6, 7]
* [cite_start]`tensorflow_datasets` [cite: 14]
