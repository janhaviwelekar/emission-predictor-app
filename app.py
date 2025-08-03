import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Supply Chain Emission Predictor", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("SupplyChainGHGEmissionFactors.csv")

df = load_data()
df.dropna(inplace=True)


features = ['2017 NAICS Title', 'GHG', 'Unit', 'Margins of Supply Chain Emission Factors',
            'Supply Chain Emission Factors with Margins']
target = 'Supply Chain Emission Factors without Margins'

X = df[features]
y = df[target]

label_encoders = {}
for col in ['2017 NAICS Title', 'GHG', 'Unit']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

model = RandomForestRegressor(random_state=42)
model.fit(X, y)


st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("Customize your inputs to predict emission factor without margins.")

st.title("🌍 Supply Chain Emission Factor Predictor (No Margin)")

st.subheader("📈 Make a Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    industry = st.selectbox("Industry", df['2017 NAICS Title'].unique())
with col2:
    ghg = st.selectbox("GHG Type", df['GHG'].unique())
with col3:
    unit = st.selectbox("Unit", df['Unit'].unique())

col4, col5 = st.columns(2)
with col4:
    margin = st.number_input("Margin Value", min_value=0.0, value=0.1)
with col5:
    emission_with_margin = st.number_input("Emission With Margin", min_value=0.0, value=1.0)

if st.button("🔍 Predict"):
    industry_enc = label_encoders['2017 NAICS Title'].transform([industry])[0]
    ghg_enc = label_encoders['GHG'].transform([ghg])[0]
    unit_enc = label_encoders['Unit'].transform([unit])[0]
    
    input_data = np.array([[industry_enc, ghg_enc, unit_enc, margin, emission_with_margin]])
    prediction = model.predict(input_data)[0]
    st.success(f"💡 Predicted Emission Factor (Without Margin): **{prediction:.3f} kg CO2e/USD**")


st.markdown("---")
st.subheader("📊 Top 10 Emitting Industries (on avg)")

top_emissions = (
    df.groupby("2017 NAICS Title")[target]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_emissions.values, y=top_emissions.index, palette="Reds_r", ax=ax)
ax.set_xlabel("Avg Emission Factor (without Margins)")
ax.set_ylabel("Industry")
ax.set_title("Top 10 Industries by Emissions")
st.pyplot(fig)


st.markdown("---")
st.subheader("🧪 Explore by GHG Type")

selected_ghg = st.selectbox("Choose GHG to analyze:", df['GHG'].unique())
filtered = df[df['GHG'] == selected_ghg]

st.write(f"Showing {len(filtered)} rows with GHG type: {selected_ghg}")
st.dataframe(filtered[['2017 NAICS Title', 'Supply Chain Emission Factors without Margins']].sort_values(by='Supply Chain Emission Factors without Margins', ascending=False).head(10))


st.markdown("---")
st.subheader("🧠 Insights")
st.markdown(f"""
- There are **{df['2017 NAICS Title'].nunique()}** unique industries.
- The most common GHG type is **{df['GHG'].mode()[0]}**.
- Highest emission (without margin) is approximately **{df[target].max():.2f}**.
- Lowest emission (without margin) is approximately **{df[target].min():.2f}**.
""")
