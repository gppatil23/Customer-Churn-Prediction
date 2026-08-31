# 🏦 Bank Customer Churn Prediction

## 📌 Overview
This project predicts **customer churn** (whether a customer will leave the bank) using the **Churn_Modelling dataset**.  
We leverage an **Artificial Neural Network (ANN)** for classification and deploy an interactive **Streamlit web app** as the frontend.

---

## 📂 Dataset
- **Source**: Churn_Modelling.csv  
- **Rows**: ~10,000 customers  
- **Features**:
  - Customer demographics (Age, Gender, Geography)
  - Account details (Balance, Credit Score, Tenure, Products)
  - Activity indicators (Has Credit Card, Is Active Member)
  - Target: **Exited** (1 = churned, 0 = retained)

---

## ⚙️ Tech Stack
- **Python 3.12**
- **TensorFlow / Keras** – ANN model
- **Pandas, NumPy, Scikit-learn** – preprocessing
- **Streamlit** – frontend UI

---

## 🧠 Model Architecture
- Input layer: Features after preprocessing (scaling, encoding)
- Hidden layers: Dense layers with ReLU activation
- Output layer: Single neuron with Sigmoid activation (binary classification)
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy

---

## 🚀 Features
- Upload customer data (CSV)
- Preprocess input automatically
- Predict churn probability using trained ANN
- Display results in a clean **Streamlit dashboard**
- Interactive visualizations (distribution plots, churn insights)

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/gppatil23/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
