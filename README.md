# Titanic Data Cleaning & Preprocessing

*A complete data‑preparation pipeline for the Titanic dataset using Python, Pandas, NumPy, Matplotlib, Seaborn, and scikit‑learn.*

---

## 📁 Project Overview
This project demonstrates a clear, repeatable pipeline to clean and preprocess the Titanic dataset so it's ready for machine‑learning tasks.  
The pipeline follows these steps:

1. Explore dataset structure and missing values.  
2. Handle missing values (median for numeric, mode for categorical).  
3. Encode categorical features into numeric (one‑hot encoding).  
4. Normalize / standardize numeric features using `StandardScaler`.  
5. Visualize outliers with boxplots and remove them using the IQR method.  
6. Produce a final cleaned dataset ready for modelling and export.

---

## 🧰 Tools & Libraries
- Python (3.7+ recommended)  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn

Use the included `requirements.txt` to install these dependencies.

---

## 🔎 Files in This Repository
- `Titanic-Dataset.csv` — original raw dataset (place this in repo before running).  
- `data_cleaning_and_preprocessing.py` — the Python script implementing the full pipeline.  
- `notebook.ipynb` — (optional) Jupyter notebook version with the same steps and visuals.  
- `requirements.txt` — required Python packages.  
- `README.md` — this file.

---

## 🧭 Detailed Workflow (what the code does)

### 1) Load Dataset
```python
df = pd.read_csv("Titanic-Dataset.csv")
```
Loads the CSV into a pandas DataFrame.

### 2) Explore Basic Info
The script prints:
- `df.info()` to show columns, non-null counts, and dtypes.  
- `df.isnull().sum()` to show missing-value counts per column.  
- `df.describe()` for numeric summaries.

Typical observations on Titanic:
- `Age` has missing values.  
- `Cabin` often has many missing values.  
- `Embarked` may have a couple missing rows.

### 3) Handle Missing Values
- Numeric columns: `df[numeric] = df[numeric].fillna(df[numeric].median())`  
- Categorical columns: `df[cat] = df[cat].fillna(df[cat].mode()[0])`

Rationale: median is robust to skew; mode preserves most frequent category.

### 4) Encode Categorical Features
One‑hot encode object/string columns:
```python
df_encoded = pd.get_dummies(df, drop_first=True)
```
This converts `Sex`, `Embarked`, and other categorical fields into numeric columns.

### 5) Normalize / Standardize Numerical Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
```
Standardizing helps ML models converge and treats features on equal scale.

### 6) Visualize & Remove Outliers
Before and after boxplots for numeric features are plotted with Seaborn:
```python
sns.boxplot(data=df_encoded[numeric_features])
```
Outliers are removed using the IQR rule applied **only to numeric columns**:
```python
Q1 = df_encoded[numeric_features].quantile(0.25)
Q3 = df_encoded[numeric_features].quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df_encoded[~((df_encoded[numeric_features] < (Q1 - 1.5 * IQR)) |
                         (df_encoded[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### 7) Final Output
- The cleaned DataFrame `df_cleaned` is displayed with `.head()` and its `.shape` is printed.  
- Optionally saved to:
```python
df_cleaned.to_csv("Titanic_Cleaned.csv", index=False)
```

---

## 📊 Expected Outputs & Example Notes
When you run the script/notebook you will see:
- Printed dataset summary (columns, dtypes, missing counts).  
- Boxplot figure showing distributions before cleaning.  
- Boxplot figure after cleaning showing reduced extreme values.  
- Printed final dataset shape like: `Final cleaned dataset shape: (XYZ, N)` — where `XYZ` is rows after outlier removal, `N` is number of columns after encoding.

---

## ✅ How to Run Locally (step-by-step)

1. **Clone your repo**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Create & activate virtual environment**

Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
Windows (cmd):
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Place dataset**
Ensure `Titanic-Dataset.csv` exists in the project root.

5. **Run**
- Notebook (recommended for viewing plots inline):
```bash
jupyter notebook
# open notebook.ipynb and run all cells
```
- Or run script:
```bash
python data_cleaning_and_preprocessing.py
```

6. **Check outputs**
- `Titanic_Cleaned.csv` (if you enabled saving).  
- Terminal / notebook for printed summaries and boxplots.

---

## 📝 Tips & Next Steps
- Feature engineering: extract titles from `Name` (Mr, Mrs, Miss), combine `SibSp` and `Parch` into family size, etc.  
- Consider model‑based imputation (KNN/IterativeImputer) for better handling of missing data.  
- If `Cabin` is important, consider extracting deck letter or flagging presence/absence.  
- Use cross‑validation and try classifiers: LogisticRegression, RandomForestClassifier, XGBoost, etc.

---

If you want, I can also:
- Add screenshots of the notebook outputs to README.  
- Create a `README_with_images.md` and embed the boxplot images (you'd need to upload images).  
- Commit the README directly to your GitHub repo (if you provide repo access or a GitHub token).

