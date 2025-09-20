# MNIST CNN Classifier - Session 5 Assignment

## 📌 Overview

This project implements a **lightweight Convolutional Neural Network (CNN)** classifier for the **MNIST digit recognition** task.

### Key Highlights

* **Parameter Efficiency**: < 20,000 trainable parameters (**17,682 total**)
* **Modern CNN Techniques**: Batch Normalization, Dropout, Global Average Pooling
* **High Accuracy**: > **99.4% test accuracy** in under **20 training epochs**

---

## 🏗️ Model Architecture

### 🔑 Key Components

1. **Depthwise Separable Convolutions (DSConv)** – efficient convolution blocks
2. **Squeeze-and-Excitation (SE) Module** – channel-wise attention
3. **Global Average Pooling (GAP)** – spatial dimension reduction
4. **Batch Normalization** – stabilizes and accelerates training
5. **Dropout** – regularization to prevent overfitting (p = 0.03)

### ⚙️ Architecture Details

* **Input**: 28×28 grayscale images
* **Stem**: Initial Conv → BatchNorm → ReLU
* **Blocks**: 4 DSConv blocks with channels: **16 → 32 → 64 → 96**
* **Pooling**: MaxPooling layers for downsampling
* **Output**: 10-class softmax (digits 0–9)

---

## ⚡ Training Configuration

* **Optimizer**: SGD with momentum (0.9), weight decay (5e-4)
* **Scheduler**: OneCycleLR
* **Loss Function**: CrossEntropyLoss
* **Data Augmentation**: Random rotations & affine transforms
* **Early Stopping**: Triggered at 99.4% test accuracy

---

## 🏆 Results

### ✅ Summary

* **Total Parameters**: 17,682 (< 20,000 requirement)
* **Best Test Accuracy**: **99.44%**
* **Training Epochs**: 19 (early stopping before 20)

### 📊 Training Progress

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc | Best Model |
| ----- | ---------- | --------- | --------- | -------- | ---------- |
| 1     | 0.6188     | 80.62%    | 0.1953    | 93.75%   | ✅ Saved    |
| 2     | 0.1041     | 96.91%    | 0.0865    | 97.47%   | ✅ Saved    |
| 5     | 0.0660     | 98.00%    | 0.0778    | 97.59%   | ✅ Saved    |
| 11    | 0.0529     | 98.47%    | 0.0761    | 97.60%   | ✅ Saved    |
| 12    | 0.0508     | 98.48%    | 0.0632    | 98.04%   | ✅ Saved    |
| 13    | 0.0489     | 98.57%    | 0.0423    | 98.87%   | ✅ Saved    |
| 16    | 0.0366     | 99.00%    | 0.0353    | 98.89%   | ✅ Saved    |
| 17    | 0.0321     | 99.09%    | 0.0300    | 99.23%   | ✅ Saved    |
| 18    | 0.0271     | 99.26%    | 0.0225    | 99.36%   | ✅ Saved    |
| 19    | 0.0226     | 99.35%    | 0.0213    | 99.44%   | ✅ Saved    |

### 🎯 Key Achievements

* **Final Test Accuracy**: 99.44% (exceeded target)
* **Training Efficiency**: Achieved target in < 20 epochs
* **Stable Training**: No overfitting observed

---

## 🔍 Model Analysis

### Analysis Results (via `model_analysis.py`)

| Requirement             | Check                              | Status   |
| ----------------------- | ---------------------------------- | -------- |
| **Total Parameters**    | 17,682 (< 20k)                     | ✅ PASSED |
| **Batch Normalization** | 4 layers (Stem + DSConv blocks)    | ✅ PASSED |
| **Dropout**             | 1 layer, p=0.03                    | ✅ PASSED |
| **GAP / FC Layer**      | AdaptiveAvgPool2d + Linear (96→10) | ✅ PASSED |

---

## 📂 Files

* `S5_assignment_final.ipynb` → Model implementation + training
* `model_analysis.py` → Script for verifying model requirements
* `README.md` → Documentation (this file)

---

## 🚀 Usage

1. Run training notebook:

   ```bash
   jupyter notebook S5_assignment_final.ipynb
   ```
2. Run analysis script:

   ```bash
   python model_analysis.py
   ```
3. View results in `model_analysis_results.json`

---

## 📈 Technical Implementation

* Depthwise separable convolutions → parameter efficiency
* Squeeze-and-Excitation → channel attention
* Dilated convolutions → larger receptive field
* OneCycleLR scheduler → optimal LR scheduling
* Gradient clipping → stable training

---

## 📐 Performance Metrics

* **Training Accuracy**: > 98%
* **Test Accuracy**: > 99.4%
* **Training Time**: \~19 epochs (early stopped)
* **Model Size**: \~2.41 MB

---

## 📜 License

This project is released for educational purposes under **ERA V4 - Session 5 Assignment** guidelines.

---