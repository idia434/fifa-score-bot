import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Prédiction de score FIFA virtuel")

@st.cache_data
def load_data():
    return pd.read_csv("resultats_fifa.csv")

data = load_data()

if st.checkbox("Voir les données brutes"):
    st.write(data.head())

X = data.drop(columns=["score_exact"])
y = data["score_exact"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.header("Faire une prédiction")

col1, col2 = st.columns(2)
with col1:
    home_attack = st.slider("Attaque équipe domicile", 50, 100, 75)
    home_defense = st.slider("Défense équipe domicile", 50, 100, 75)
with col2:
    away_attack = st.slider("Attaque équipe extérieure", 50, 100, 75)
    away_defense = st.slider("Défense équipe extérieure", 50, 100, 75)

input_data = pd.DataFrame({
    "home_attack": [home_attack],
    "home_defense": [home_defense],
    "away_attack": [away_attack],
    "away_defense": [away_defense]
})

prediction = model.predict(input_data)[0]
st.success(f"Score exact prédit : {prediction}")

if st.checkbox("Afficher la précision du modèle"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Précision du modèle : {accuracy:.2%}")
