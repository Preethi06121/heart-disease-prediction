# ?? Heart Disease Prediction using Logistic Regression

### ?? Short Description
A machine learning project that predicts the **10-year risk of heart disease** using the **Framingham Heart Study dataset**.  
Built with **Python (pandas, scikit-learn)** and trained using **Logistic Regression**, this model analyzes health factors such as age, blood pressure, cholesterol, and smoking habits to assess risk levels.

---

### ?? Overview
This project predicts the likelihood of a person developing heart disease within ten years using the **Framingham Heart Study dataset**.  
The model is built using **Logistic Regression**, a supervised machine learning algorithm for binary classification.

---

### ?? Dataset
**Dataset Name:** Framingham Heart Study  
**Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)

**Target Column:** `TenYearCHD`  
(1 = Has heart disease, 0 = No heart disease)

**Key Features:**
- `male` – Gender (1 = Male, 0 = Female)
- `age` – Age in years
- `currentSmoker` – Whether the person smokes
- `cigsPerDay` – Cigarettes smoked per day
- `BPMeds` – BP medication (1 = Yes)
- `prevalentStroke` – History of stroke
- `prevalentHyp` – Hypertension
- `diabetes` – Diabetes present
- `totChol` – Total cholesterol
- `sysBP` – Systolic blood pressure
- `diaBP` – Diastolic blood pressure
- `BMI` – Body Mass Index
- `heartRate` – Heart rate
- `glucose` – Glucose level

---

### ?? Model Used
**Algorithm:** Logistic Regression  
**Library:** scikit-learn  

---

### ?? Steps
1. Load dataset  
2. Handle missing values with `dropna()`  
3. Split data into training and testing sets  
4. Train model using `LogisticRegression()`  
5. Evaluate accuracy using `accuracy_score()`  
6. Predict heart disease for a new patient

### ?? Example Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("framingham.csv")
data = data.dropna()

x = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

predictions = model.predict(x_train)
print("Training Accuracy:", accuracy_score(y_train, predictions))

test_pred = model.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("Prediction (1=Yes, 0=No):", model.predict([[1,44,2,0,20,0,0,1,0,222,144,69,44,78,75]])

### ?? Example Output
```
Training Accuracy: 0.84
Test Accuracy: 0.81
Prediction (1=Yes, 0=No): [0]

### ?? Requirements
Install all dependencies using:
```bash
pip install -r requirements.txt

### ?? Future Improvements
- Add feature scaling (StandardScaler)
- Try Random Forest, SVM for comparison
- Create Streamlit web app for user-friendly prediction



### ????? Author
**Preethi**  
B.Tech (3rd Year) | Machine Learning Enthusiast  
?? Email: harshithakoyee@gmail.com 
?? GitHub: [https://github.com/yourusername](https://github.com/yourusername)
