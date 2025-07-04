# 🗑️ Garbage Classification Project

Welcome to the **Garbage Classification** project!  
This repository demonstrates how to build and train a deep learning model to automatically classify images of garbage into categories using **TensorFlow** and **EfficientNetV2B2**, and also deploy it with **Gradio**.

---

## 🚀 Overview

This project walks you through:
- Checking GPU availability (for faster training on Colab)
- Preparing and processing image datasets
- Building and training a deep learning model using transfer learning
- Visualizing training metrics and results
- Evaluating performance on test data
- Deploying a simple user interface with **Gradio**

---


## 🧩 Requirements

Ensure you have the following packages installed:

```bash
pip install tensorflow gradio matplotlib numpy pillow
```

**Dependencies:**
- Python 3.x
- TensorFlow (>=2.x)
- Gradio (for the interactive UI)
- NumPy
- Matplotlib
- Pillow (PIL)

---

## 📂 Dataset

Your dataset (`garbage.zip`) should include a folder like this:

```
TrashType_Image_Dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Each subfolder should contain images of the corresponding class.

---

## 🏗️ Model Architecture

- **Base Model:** EfficientNetV2B2 (pre-trained on ImageNet)
- **Input Size:** 124x124 pixels
- **Custom Layers:**  
  - `GlobalAveragePooling2D`  
  - `Dense`, `Dropout`, `Softmax` output layer
- **Loss Function:** `SparseCategoricalCrossentropy`
- **Optimizer:** Adam
- **Epochs:** 10 (can be tuned)
- **Classes:** Automatically inferred from dataset folders

---

## 📝 How to Use

1. **Upload the dataset.**  
   Place `garbage.zip` in your working directory and unzip it.

2. **Run the Notebook.**  
   Execute all cells in  `Garbage_ClassificationFinal.ipynb`.

3. **Monitor training:**  
   - Training and validation loss/accuracy are plotted
   - Test set evaluation and predictions are shown at the end

---

## 🌐 Deployment

The project is deployed with **Gradio**, allowing users to upload an image and get predictions instantly.

🔗 **[Live Gradio App](https://huggingface.co/spaces/ashu0812/Garbage-Classification)**  


---

## 📊 Results & Visualization

- **Accuracy & Loss Curves:** Displayed after each epoch
- **Prediction Samples:** Test images with predicted vs actual labels
- **Test Accuracy:** ~100% (on small dataset; may vary depending on size and complexity)

---

## ⚠️ Notes

- Make sure your folder structure matches `image_dataset_from_directory` expectations.
- Training on larger datasets will benefit significantly from GPU acceleration.
- The Gradio interface shows only the top predicted class with confidence score.
- Overfitting may occur on small datasets — use validation and regularization wisely.

---

## 📜 License

Please ensure your dataset complies with its license before distribution or reuse.

---

## 🙌 Acknowledgments

- TensorFlow/Keras team for powerful deep learning tools
- Authors of EfficientNet
- Open-source image dataset creators

---

**Happy Classifying! 🧠♻️**
