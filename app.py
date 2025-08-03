import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Supply Chain Emission Factor (With Margin) Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("SupplyChainGHGEmissionFactors.csv")
    df.dropna(inplace=True)
    return df

df = load_data()
st.title("🌍 Supply Chain Emission Factor with Margin - ML Predictor")


features = ['2017 NAICS Title', 'GHG', 'Unit',
            'Supply Chain Emission Factors without Margins',
            'Margins of Supply Chain Emission Factors']
target = 'Supply Chain Emission Factors with Margins'

X = df[features]
y = df[target]

encoders = {}
for col in ['2017 NAICS Title', 'GHG', 'Unit']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("🔧 Input Parameters")
industry = st.sidebar.selectbox("Industry", df['2017 NAICS Title'].unique())
ghg = st.sidebar.selectbox("GHG Type", df['GHG'].unique())
unit = st.sidebar.selectbox("Unit", df['Unit'].unique())
margin = st.sidebar.slider("Margin Value", 0.0, 1.0, 0.1)
no_margin = st.sidebar.slider("Emission Factor (Without Margin)", 0.0, 10.0, 1.0)


input_data = pd.DataFrame({
    '2017 NAICS Title': [encoders['2017 NAICS Title'].transform([industry])[0]],
    'GHG': [encoders['GHG'].transform([ghg])[0]],
    'Unit': [encoders['Unit'].transform([unit])[0]],
    'Supply Chain Emission Factors without Margins': [no_margin],
    'Margins of Supply Chain Emission Factors': [margin]
})


if st.sidebar.button("🔍 Predict Emission with Margin"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Emission Factor (With Margin): {prediction:.4f} kg CO2e/USD")

st.subheader("📊 Model Performance on Test Set")
y_pred = model.predict(X_test)
st.write(f"**R² Score**: {r2_score(y_test, y_pred):.3f}")
st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

st.subheader("🔬 Top Emitting Industries (Avg Emission With Margin)")
top_emitters = df.groupby("2017 NAICS Title")[target].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_emitters.values, y=top_emitters.index, palette="rocket", ax=ax)
ax.set_xlabel("Avg Emission (With Margin)")
ax.set_title("Top 10 Emitting Industries")
st.pyplot(fig)


st.subheader("🧪 Feature Correlation Heatmap")
encoded_df = X.copy()
encoded_df[target] = y
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(encoded_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)


with st.expander("📁 Show Raw Dataset"):
    st.dataframe(df.head())

