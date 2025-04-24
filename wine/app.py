import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Prediction App")
st.markdown("Predict **red wine quality** using a Random Forest model with improved accuracy.")

# Load and clean data
df = pd.read_csv("winequality-red.csv", sep=';')
df.columns = df.columns.str.strip()

# Combine classes
df['quality_label'] = df['quality'].apply(
    lambda q: 'Low' if q <= 4 else ('Medium' if q <= 6 else 'High')
)

# Encode target
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['quality_encoded'] = df['quality_label'].map(label_map)

# Features and target
X = df.drop(['quality', 'quality_label', 'quality_encoded'], axis=1)
y = df['quality_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Build improved RandomForest
model = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=4, random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

# Show classification report
with st.expander("📊 Classification Report"):
    st.text(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Sidebar input form
st.sidebar.header("Enter Wine Attributes:")
user_input = {}
for col in X.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    step = (max_val - min_val) / 100
    user_input[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, step=step, format="%.4f")

# Predict user input
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

label_reverse = {v: k for k, v in label_map.items()}
predicted_label = label_reverse[prediction]

# Display prediction
st.subheader("🔍 Prediction Result")
if predicted_label == 'Low':
    st.error(f"Predicted Quality: **{predicted_label}** 🍷")
elif predicted_label == 'Medium':
    st.warning(f"Predicted Quality: **{predicted_label}** 🍷")
else:
    st.success(f"Predicted Quality: **{predicted_label}** 🍷")

# Show probabilities
with st.expander("📈 Prediction Probabilities"):
    proba_df = pd.DataFrame([proba], columns=label_map.keys())
    st.dataframe(proba_df)

# Show input summary
with st.expander("📊 Input Summary"):
    st.dataframe(input_df)
