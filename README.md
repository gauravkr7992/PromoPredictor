# 📌 Employee Promotion Prediction

## 📖 Overview
This project predicts **employee promotions** using **Machine Learning**. The dataset includes employee details, training scores, past performance, and other relevant factors. We handle **imbalanced data** using **SMOTE**, perform **feature engineering**, and apply a **Decision Tree Classifier**.

## 📊 Dataset
- **Source:** [HR Analytics Dataset](#)
- **Features:**
  - `age`, `length_of_service`, `previous_year_rating`, `awards_won?`, `KPIs_met >80%`
  - `avg_training_score`, `no_of_trainings`, `education`, `gender`, `department`
  - Target Variable: **`is_promoted` (0: Not Promoted, 1: Promoted)**

## 🔍 Methodology
1. **Data Preprocessing**
   - Handling missing values (`education` filled with mode)
   - Encoding categorical variables
   - Feature scaling using `StandardScaler`

2. **Exploratory Data Analysis (EDA)**
   - **Visualizations:**
     - **Pie Chart** (`train['is_promoted'].value_counts().plot(kind='pie')`)
     - **Bar Plot** (`sns.barplot(data=train, x='department', y='avg_training_score')`)
     - **Box Plot for Outliers** (`sns.boxenplot(data=train, x='is_promoted', y='age')`)
   - **Feature Correlation:** `sns.heatmap(correlation_matrix, annot=True, cmap='Wistia')`

3. **Handling Imbalanced Data**
   - **SMOTE (Synthetic Minority Oversampling Technique)** applied:
     ```python
     from imblearn.over_sampling import SMOTE
     smote = SMOTE()
     X_resampled, y_resampled = smote.fit_resample(X, y)
     ```

4. **Model Training**
   - Using **Decision Tree Classifier**
     ```python
     from sklearn.tree import DecisionTreeClassifier
     model = DecisionTreeClassifier()
     model.fit(X_train, y_train)
     ```

5. **Model Evaluation**
   - **Confusion Matrix & Heatmap**
     ```python
     from sklearn.metrics import confusion_matrix
     import seaborn as sns
     cm = confusion_matrix(y_test, y_pred)
     sns.heatmap(cm, annot=True, cmap='Wistia')
     ```
   - Accuracy, Precision, Recall, F1-score

## 🛠 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/employee-promotion-prediction.git
cd employee-promotion-prediction

# Install dependencies
pip install -r requirements.txt

