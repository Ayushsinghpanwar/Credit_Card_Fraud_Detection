# Credit_Card_Fraud_Detection

This repository contains a machine learning project focused on detecting fraudulent credit card transactions. The project uses a dataset of European credit card transactions from September 2013 to train and evaluate several classification models.

---

##  About the Dataset

- The dataset is sourced from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- It contains transactions over two days with a **severe class imbalance**: only **0.172%** of transactions are fraudulent.  
- Due to confidentiality, features have been transformed using **PCA** → resulting in 28 principal components (`V1` → `V28`).  
- The only non-transformed features are **`Time`** and **`Amount`**.  

**Goal** → Build a model that can **accurately identify fraudulent transactions** despite this imbalance.  


---

##  How It Works

1. **Data Loading & Preprocessing**
   - Load `creditcard.csv` dataset
   - Remove duplicate entries
   - Separate features (`X`) and target (`Class`)

2. **Feature Scaling**
   - Standardize `Time` and `Amount` using `StandardScaler`

3. **Handling Imbalance**
   - Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data  
   - This generates synthetic fraud samples → improves learning

4. **Model Training**
   - Train multiple models (`Logistic Regression`, `Random Forest`, `XGBoost`)  
   - Hyperparameters tuned for **XGBoost**

5. **Prediction Function**
   - Custom function `predict_fraudulent_transaction` makes predictions on **new transactions**

---

**Model Performance**

Model    	      ROC-AUC      Recall (Fraud Class)
Logistic          Regression	 0.9626	87%
Random Forest	   0.9694	    77%
XGBoost (Initial)	0.9699	    79%
XGBoost (Tuned)	0.9758	    74%

 Best Model → XGBoost (Tuned)

Highest ROC-AUC → best at ranking fraud vs. non-fraud

Recall slightly lower than Logistic Regression but overall discrimination power is superior
