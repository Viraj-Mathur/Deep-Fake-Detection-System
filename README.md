# Deep Fake Image Detection System Using CNNs

A robust deepfake detection system designed to distinguish between real and synthetic (deepfake) images using deep learning models. This project uses a CNN-based architecture and a ResNet model to enhance detection accuracy and evaluate their performance in identifying deepfakes. By using a large, diverse dataset from Kaggle, this system aims to provide a reliable, scalable solution for identifying manipulated media in real-world applications.

## Table of Contents
- [Deep Fake Image Detection System Using CNNs](#deep-fake-image-detection-system-using-cnns)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Models Used](#models-used)
  - [Setup and Installation](#setup-and-installation)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Future Work](#future-work)

## Overview
This project is focused on developing a deep learning-based deepfake detection system. With the rise of synthetic media, this tool aims to accurately classify images as either authentic or deepfake by learning subtle, distinguishing features within manipulated media. The system is built using:
- **Basic CNN Model**: A custom Convolutional Neural Network architecture for initial deepfake detection.
- **ResNet50 Model**: A pre-trained ResNet model, fine-tuned to enhance detection capabilities by leveraging its deeper feature extraction.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/code/shubhamkumarbhokta/cnn-2/input
), containing two classes of images:
- **Real**: Images that are unaltered and represent real content.
- **Fake**: Deepfake images manipulated using various AI techniques.

The dataset is structured into training, validation, and test sets, ensuring balanced classes to support effective supervised learning.

## Models Used
1. **CNN Model**: A basic CNN architecture with convolutional and pooling layers, followed by fully connected layers. This model is designed to extract features and classify images as real or fake.
2. **ResNet Model**: A more complex ResNet50 architecture, pre-trained on ImageNet, adapted for deepfake detection with additional dense layers to improve classification accuracy.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Viraj-Mathur/Deep-Fake-Detection-System.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset:
   - Place images in folders for training, validation, and testing under "Fake" and "Real" subdirectories as expected by the model.

## Model Training
The models are trained on the preprocessed Kaggle dataset, with the following steps:

1. **Preprocessing**: 
   - Images are resized, normalized, and augmented to improve generalization.
   - Data augmentation techniques such as rotation and flipping are applied.

2. **Training**:
   - The CNN model is trained using categorical cross-entropy as the loss function and the Adam optimizer.
   - ResNet model is fine-tuned on the same dataset, leveraging its pre-trained weights for enhanced feature extraction.

```python
# Example model training code for CNN
history = model.fit(train_flow, epochs=10, validation_data=validation_flow)
```

## Evaluation
The performance of each model is evaluated on a separate test dataset. Key metrics include accuracy, precision, recall, and F1-score. Additionally, confusion matrices are used to visualize classification performance for each class (Real and Fake).

- **CNN Model Accuracy**: 50.22%
- **ResNet Model Accuracy**: 64.68%

```python
# Example evaluation code for CNN
loss, accuracy = model.evaluate(test_flow)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

## Results
The CNN model achieved balanced but limited performance, with a test accuracy of 50.22%. The ResNet model, leveraging deeper layers, performed better with a test accuracy of 64.68%. However, both models exhibited challenges in generalizing to all types of manipulations, indicating potential for further refinement. Figures illustrating classification reports and confusion matrices provide insights into model strengths and weaknesses.

## Future Work
1. **Advanced Architectures**: Implement additional architectures such as EfficientNet or Vision Transformers to improve detection capabilities.
2. **Multi-Modal Detection**: Combine visual analysis with audio or text features to detect deepfakes in multimedia.
3. **Real-Time Detection**: Optimize models for real-time deployment for applications requiring immediate verification.


This project provides a foundation for reliable deepfake detection, with potential applications in media authentication, content verification, and safeguarding digital integrity.
