# 🔢 MNIST Digit Recognition using Random Forest

## 📌 Project Overview

This project is a Machine Learning application that recognizes handwritten digits from **0 to 9** using the **MNIST Digit Recognizer dataset**.

A Random Forest classification model was trained on handwritten digit images and integrated into a **Streamlit web application**. Users can upload a digit image, and the application predicts which digit it represents along with the model's confidence.

---

## 🎯 Objective

The main objectives of this project are:

* Recognize handwritten digits from 0 to 9.
* Preprocess image data for Machine Learning.
* Train a Random Forest classification model.
* Evaluate the model's performance.
* Save the trained model for future predictions.
* Build an interactive Streamlit prediction application.

---

## 📊 Dataset

The project uses the **Kaggle Digit Recognizer / MNIST dataset**.

Each handwritten digit image contains:

* **28 × 28 pixels**
* **784 pixel features**
* One target label representing the digit from **0 to 9**

### Dataset Files

* `train.csv` — Training dataset
* `test.csv` — Test dataset
* `sample_submission.csv` — Sample submission format

---

## 🔄 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the training and testing datasets.
2. Checked the dataset shape and columns.
3. Checked for missing values.
4. Separated features and target labels.
5. Removed the target column from the input features.
6. Normalized pixel values.
7. Split the training data into training and validation sets.
8. Prepared the data for Random Forest classification.

Pixel values were scaled from:

```text
0–255 → 0–1
```

---

## 🤖 Machine Learning Model

### Random Forest Classifier

A **Random Forest Classifier** was used for handwritten digit classification.

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make predictions.

The model learns patterns from the pixel values and predicts the corresponding handwritten digit.

The trained model was saved as:

```text
mnist_random_forest_model.pkl
```

---

## 📈 Model Evaluation

The model was evaluated using standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The evaluation was performed on the validation/test data to understand how well the model recognizes different handwritten digits.

---

## 🌐 Streamlit Web Application

A Streamlit application was developed to make the trained model interactive.

### Application Features

* Upload handwritten digit images.
* Convert uploaded images to grayscale.
* Resize images to **28 × 28 pixels**.
* Normalize pixel values.
* Flatten the image into **784 features**.
* Predict the handwritten digit.
* Display prediction confidence.

### Prediction Output

The application displays:

```text
🔢 Predicted Digit
🎯 Confidence
```

---

## 📂 Project Structure

```text
MNIST-Digit-Recognition/
│
├── app.py
├── mnist_random_forest_model.pkl
├── requirements.txt
├── README.md
│
├── train.csv
├── test.csv
├── sample_submission.csv
└── mnist_submission.csv
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

Clone this repository to your computer.

### 2. Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

### 4. Upload an Image

Open the Streamlit application in your browser and upload a handwritten digit image in:

* PNG
* JPG
* JPEG

The application will process the image and display the predicted digit.

---

## ⚠️ Limitations

The application works best with images that are similar to the MNIST dataset.

Prediction performance may decrease when:

* The image contains multiple digits.
* The digit is very small or very large.
* The background is different from the expected format.
* The handwriting is unclear.
* The digit is not centered properly.
* The uploaded image contains additional marks or noise.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Random Forest
* Joblib
* Pillow (PIL)
* Streamlit
* Google Colab
* GitHub

---

## 📚 Project Type

**Machine Learning / Computer Vision Project**

**Task:** MNIST Digit Recognition

**Model:** Random Forest Classifier

**Deployment:** Streamlit

---

## 👩‍💻 Author

**Farheen Khan**

Computer Science Student | AI/ML & Software Development

### Skills Used

* Python
* Machine Learning
* Computer Vision
* Data Preprocessing
* Scikit-learn
* Streamlit
* Git & GitHub

---

## ⭐ Conclusion

This project demonstrates the complete Machine Learning workflow, from dataset preprocessing and model training to evaluation and deployment.

The Random Forest model was integrated with a Streamlit interface to provide an easy-to-use handwritten digit recognition system.
