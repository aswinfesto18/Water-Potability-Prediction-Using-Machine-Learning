import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Title

st.title("💧 Water Potability Prediction App")
st.write("""
Input the water parameters below and click **Predict** to determine if water is *potable* (safe to drink) or *not potable.*  
(Uses multiple models behind the scenes.)
""")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("water_potability.csv")

df = load_data()

if st.checkbox("Show Raw Dataset"):
    st.write(df)

# Data Visualization

if st.checkbox("Show Class Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(x='Potability', data=df, ax=ax)
    ax.set_title("Potability Class Distribution")
    st.pyplot(fig)

# Preprocessing

@st.cache_resource
def preprocess(df):
    data = df.copy()
    data.fillna(data.median(), inplace=True)  # fill missing
    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler

X_res, y_res, scaler = preprocess(df)

# Train-Test split

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Model Training & Tuning

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True)
}

best_model = None
best_score = 0
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    f1 = classification_report(y_test, preds, output_dict=True)["1"]["f1-score"]
    results.append((name, f1))
    
    if f1 > best_score:
        best_score = f1
        best_model = model

# Display model performance
result_df = pd.DataFrame(results, columns=["Model", "F1 Score"])
st.subheader("📈 Model Performance (F1 Scores)")
st.dataframe(result_df)

# ROC & Confusion Matrix

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plot1, plot2 = plt.subplots()

st.subheader("📊 ROC Curve")
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(plot1)

st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# Feature Input Form

st.subheader("🔍 Enter Water Parameters")

input_ph = st.slider("pH", 0.0, 14.0, 7.0)
input_hardness = st.number_input("Hardness", 0.0, 500.0, 100.0)
input_solids = st.number_input("Solids (total dissolved solids)", 0.0, 50000.0, 1000.0)
input_chloramines = st.number_input("Chloramines", 0.0, 20.0, 5.0)
input_sulfate = st.number_input("Sulfate", 0.0, 5000.0, 100.0)
input_conductivity = st.number_input("Conductivity", 0.0, 5000.0, 500.0)
input_organic_carbon = st.number_input("Organic Carbon", 0.0, 50.0, 10.0)
input_trihalomethanes = st.number_input("Trihalomethanes", 0.0, 200.0, 50.0)
input_turbidity = st.number_input("Turbidity", 0.0, 10.0, 1.0)

# Prediction

if st.button("Predict"):
    user_data = np.array([[input_ph, input_hardness, input_solids,
                           input_chloramines, input_sulfate, input_conductivity,
                           input_organic_carbon, input_trihalomethanes, input_turbidity]])
    
    user_scaled = scaler.transform(user_data)
    pred = best_model.predict(user_scaled)[0]
    
    if pred == 1:
        st.error("❌ Not Potable (Unsafe to Drink)")
    else:
        st.success("✅ Potable (Safe to Drink)")
