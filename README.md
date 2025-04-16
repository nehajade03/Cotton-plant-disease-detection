# 🌿 Cotton Plant Diseases Detection using CNN

## 📌 Overview

Cotton is one of India's most vital cash crops. However, it is highly susceptible to various leaf diseases such as **Bacterial Blight**, **Target Spot**, **Powdery Mildew**, and **Aphids**. This project presents a deep learning-based solution using **Convolutional Neural Networks (CNN)** to detect and classify these diseases from leaf images, enabling early diagnosis and preventive action.

---

## 🎯 Objectives

- Build a predictive model that classifies cotton leaf images into healthy or diseased categories.
- Provide an intuitive interface for uploading and analyzing leaf images.
- Increase crop yield and quality through timely disease detection.

---

## 🧪 Problem Statement

Manual identification of cotton plant diseases is time-consuming, requires expertise, and often lacks precision. This project uses **image processing and deep learning** to automate and enhance disease detection directly from leaf images.

---

## 📂 Dataset

- **Source**: Kaggle and online repositories
- **Total Images**: 1815
- **Classes**:
  - Healthy (617)
  - Bacterial Blight (304)
  - Target Spot (243)
  - Powdery Mildew (279)
  - Aphids (352)
- **Format**: All images standardized to `.png`, resized to `256x256`

---

## 🧠 Methodology

### Image Preprocessing
- Resize to 256x256
- Format standardization
- Background noise removal

### Model Architecture
- **5 Convolutional Layers** (3x3 filters)
- **6 Max Pooling Layers** (2x2)
- **ReLU Activation**
- **Fully Connected Layers**
- **Softmax Output Layer**
- **Flatten Layer** for dimensionality reduction

### Training Details
- **Train/Validation/Test Split**: 1440 / 160 / 224
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50

---

## 📊 Performance

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 91.61%    |
| Model Size   | 236 KB    |
| Format       | `.tflite` for mobile deployment |
| Augmentation | Significantly improved accuracy |

---

## 🛠️ Tools & Technologies

- **Languages**: Python 3.11
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Framework**: TensorFlow Lite
- **Hardware**: Core i5, GPU - AMD Radeon
- **Platform**: Windows 10

---

## 📱 Features

- Upload cotton leaf images for instant disease prediction
- Supports offline functionality with TFLite model
- History module to view previous diagnoses
- Lightweight and fast model for mobile use

---

## 💡 Importance of Early Detection

- Prevent yield loss and maintain fiber quality
- Reduce costs of treatment and spread containment
- Boost farmer productivity and profitability
- Enable real-time, field-level disease management

---

## 🔄 Future Improvements

- Incorporate additional disease categories
- Apply Transfer Learning (VGG, ResNet, etc.)
- Integrate with drones/mobile apps for real-time scanning
- Enhance model with weather and soil data correlation

---

## 📚 References

1. Pranita P. Gulve et al., *Leaf Disease Detection using Image Processing*, 2015  
2. A. Jenifa, R. Ramalakshmi et al., *Classification of Cotton Leaf Disease using CNN*, 2019  
3. Rakesh Chaware et al., *Detection and Recognition of Leaf Disease*, 2017  
4. Nikhil Shah, Sarika Jain, *Disease Detection in Cotton Leaf using ANN*, 2019  

---

## 📜 License

This project is intended for academic and educational use. All datasets used are publicly available and free to use.

---

## 🙌 Acknowledgements

Special thanks to the researchers and open-source contributors whose work and datasets made this project possible.

