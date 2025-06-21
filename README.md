# 🗑️ Garbage Classification Project

Welcome to the **Garbage Classification** project!  
This repository demonstrates how to build and train a deep learning model to automatically classify images of garbage into different categories using TensorFlow and EfficientNetV2B2.

---

## 🚀 Overview

This Jupyter Notebook walks you through:
- Checking GPU availability for fast training
- Unzipping and preparing your dataset
- Loading, splitting, and efficiently processing image data
- Building and training a deep learning model
- Visualizing training progress and results
- Evaluating and testing the model

---

## 🗃️ File Structure

- **`Garbage_class.ipynb`** — Main notebook (run this in [Google Colab](https://colab.research.google.com/) for best experience)
- **`garbage.zip`** — (Expected) Zipped dataset containing images in subfolders by class

---

## 🧩 Requirements

- Python 3.x
- [TensorFlow](https://www.tensorflow.org/) (>=2.x)
- [Gradio](https://gradio.app/) (for UI, optional)
- Matplotlib
- Numpy
- Pillow (PIL)

Install with:

```bash
pip install tensorflow gradio matplotlib numpy pillow
```

---

## 📂 Dataset

Place your zipped dataset (`garbage.zip`) at the root of your Colab or working directory.  
The dataset should contain a folder named `TrashType_Image_Dataset` with subdirectories for each garbage class (e.g., plastic, metal, etc.).

---

## 🏗️ Model Architecture

- **Base:** EfficientNetV2B2 (pre-trained on ImageNet)
- **Custom Layers:** GlobalAveragePooling, Dense, Dropout, Output (Softmax)
- **Input Size:** 124 x 124 px
- **Classes:** 6 (auto-detected from dataset)
- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 10 (feel free to experiment!)

---

## 📝 How to Use

1. **Upload your dataset.**  
   Make sure `garbage.zip` is present and contains the right folder structure.

2. **Run the Notebook.**  
   Open `Garbage_class.ipynb` in Colab or Jupyter and execute all cells.

3. **Monitor Progress.**  
   - Training/validation accuracy and loss are plotted after each epoch.
   - Test set performance is evaluated at the end.
   - Example predictions are visualized alongside actual labels.

---

## 📊 Results & Visualization

- **Accuracy & Loss Curves:** Track model performance over epochs.
- **Prediction Samples:** See side-by-side true and predicted labels for test images.
- **Reported Test Accuracy:** ~100% (may vary; watch for overfitting on small datasets!)

---

## 💡 Notes

- Dataset folder structure should match what `image_dataset_from_directory` expects.
- The notebook is written for Colab (GPU support).
- Gradio is imported for potential UI use, but not deployed in this notebook.
- For large datasets, training time will depend on GPU availability.

---

## 🏷️ License

Please verify your dataset’s license before use or sharing.

---

## 🙌 Acknowledgments

- TensorFlow and Keras Team
- EfficientNet authors
- Open-source dataset providers

---

**Happy Classifying! 🧹🌱**
