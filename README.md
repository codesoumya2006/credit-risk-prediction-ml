# 💳 Credit Risk Prediction App

An end-to-end **Machine Learning Credit Risk Prediction Application** that evaluates loan applicants and predicts whether they belong to **Low (Good)** or **High (Bad)** credit risk categories based on financial and demographic data.

---

## 📌 Project Overview

Credit risk assessment is a critical task for banks and financial institutions. This project uses **Machine Learning** techniques to analyze applicant data and predict the likelihood of loan default.

The application allows users to input applicant details through a simple interface and receive **real-time credit risk predictions**.

---

## 🚀 Features

- 📊 Real-time credit risk prediction  
- 🧠 Machine learning–based classification  
- 🔢 User-friendly data input form  
- ⚡ Fast and lightweight model inference  
- 🌐 Web-based interface (Streamlit)  
- 📁 Clean and modular project structure  

---

## 🧾 Input Features

The model takes the following inputs:

- **Age**
- **Sex** (male / female)
- **Job Type** (0–3)
- **Housing** (own / rent / free)
- **Saving Accounts** (little / moderate / quite rich / rich)
- **Checking Account** (little / moderate / rich)
- **Purpose of Loan** (business, car, education, etc.)
- **Credit Amount**
- **Loan Duration (months)**

---

## 📈 Output

The application predicts:

- ✅ **LOW (Good)** – Low probability of default  
- ⚠️ **HIGH (Bad)** – High probability of default  

Example:
---

## 🧠 Machine Learning Details

- **Problem Type**: Binary Classification  
- **Algorithms Used**: XGBoost / scikit-learn  
- **Target Variable**: Credit Risk (Good / Bad)  
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

## 🛠️ Tech Stack

- **Language**: Python 3.10  
- **Machine Learning**: scikit-learn, XGBoost  
- **Data Processing**: pandas, numpy  
- **Visualization**: matplotlib, seaborn  
- **Web Framework**: Streamlit  
- **Environment**: Python Virtual Environment (.venv)  

---

## 📂 Project Structure
```
credit-risk-prediction-app/
│
├── data/ # Dataset files
├── model/ # Trained ML model
├── notebooks/ # EDA & model training notebooks
├── app.py # Streamlit application
├── train_model.py # Model training script
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .venv/ # Virtual environment
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/credit-risk-prediction-app.git
cd credit-risk-prediction-app
```
### 2️⃣ Create & Activate Virtual Environment
```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Run the Application
```bash
streamlit run app.py
```
---
## 🎯 Use Cases

- Bank loan approval systems  
- FinTech credit scoring platforms  
- Financial risk analysis  
- Academic and portfolio ML projects  

---

## 🔮 Future Enhancements

- 📊 Model explainability (SHAP / LIME)  
- 📈 Model performance dashboard  
- 🧾 Automated credit reports  
- 🔐 User authentication  
- ☁️ Cloud deployment (Streamlit Cloud / AWS / Render)  

---
