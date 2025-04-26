---

# 🎵 Million Song Decade Classification

Predict the **decade** a song was released in, based on **audio features** from the **Million Song Dataset**, using a **Neural Network** built with **TensorFlow/Keras**.

---

## 📚 Problem Statement

Given a song’s audio features (like timbre, pitch, loudness, etc.), can we accurately predict the **decade** in which the song was released?  
This project applies a **supervised machine learning** approach to **multi-class classification**, predicting one of 10 possible decades from 1920s to 2000s.

---

## 🗂️ Project Structure

```
📁 million-song-decade-classification/
├── 1_data_preprocessing.py
├── 2_model_architectures.py
├── 3_train_and_evaluate.py
├── 4_eda_feature_analysis.ipynb
├── 5_learning_rate_finder.py
├── 6_advanced_optimization.py
├── 7_blog_visualizations.ipynb
├── README.md
├── 📁 data/
│   └── YearPredictionMSD.txt
├── 📁 models/
│   └── best_model.h5
├── 📁 src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── model_architectures.py
```

---

## 🛠️ Technologies Used

- **Python** 🐍
- **TensorFlow** & **Keras** 🧠
- **Scikit-Learn** 🔥
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn** 📊

---

## 🏗️ Solution Architecture

> Here’s a simple high-level flow of the system:

```
Data Preprocessing ➔ Feature Scaling ➔ Train/Test Split ➔ Model Building ➔ Model Training ➔ Evaluation ➔ Visualization
```

- **Data Preprocessing**:  
  - Load `.txt` dataset.
  - Map `Year` into `Decade` buckets.
  - Standardize features using `StandardScaler`.

- **Model Architecture**:  
  - Fully Connected (Dense) Neural Network.
  - 3 Hidden Layers: (128 → 64 → 32 units, ReLU Activation)
  - Output Layer: 10 neurons (Softmax activation for 10 decades)

- **Training**:  
  - Optimizer: **Adam**
  - Loss: **Sparse Categorical Crossentropy**
  - Evaluation: Accuracy, Loss

- **Visualization**:  
  - Accuracy vs Epoch Graph 📈
  - Loss vs Epoch Graph 📉

---

## 📊 Exploratory Data Analysis (EDA)

- **Distribution of Years**:  
  Most songs are clustered between **1950 and 2000**, peaking in the 1970s and 1980s.

- **Feature Statistics**:  
  All features are numerical. Some features have large ranges, hence scaling is necessary.

- **Feature Correlation**:  
  Some features show medium-high correlation, hinting at possible redundancies.

**Sample Visualization:**

![Sample](https://user-images.githubusercontent.com/your-image-link.png)  
*(Note: Insert your EDA graphs if uploading images)*

---

## 📈 Model Performance

| Metric            | Value      |
|-------------------|------------|
| Validation Accuracy | ~68% (baseline) |
| Final Test Accuracy | ~72% after tuning |

Training and Validation curves:

```python
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

---

## 📦 How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/million-song-decade-classification.git
   cd million-song-decade-classification
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the data:
   ```bash
   python 1_data_preprocessing.py
   ```

4. Train the model:
   ```bash
   python 3_train_and_evaluate.py
   ```

5. Visualize the results:
   ```bash
   jupyter notebook 7_blog_visualizations.ipynb
   ```

---

## 🧠 Future Improvements

- Apply **Convolutional Neural Networks** (CNNs) on spectrogram images 🎶.
- Use **Transformer-based models** for sequence understanding.
- Apply **Ensemble Learning** to boost accuracy.

---

## ✍️ Authors

- **Your Name** - [@ShreyasDole](https://github.com/ShreyasDole)

---

---

# ⭐ If you like this project, don't forget to Star it!

---
