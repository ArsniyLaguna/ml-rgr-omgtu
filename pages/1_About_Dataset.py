import streamlit as st
import pandas as pd

st.set_page_config(page_title="О датасете", layout="wide")
st.title("Структура данных предметной области")

@st.cache_data
def load_data():
    return pd.read_csv("R3_dataset_processed.csv")

df = load_data()

st.markdown("### Общие параметры выборки")
st.write(f"* **Всего наблюдений (строк):** {df.shape[0]}")
st.write(f"* **Общее количество признаков (столбцов):** {df.shape[1]}")

st.markdown("### Первые 5 строк датасета:")
st.dataframe(df.head(5))

st.markdown("### Описательная статистика признаков:")
st.dataframe(df.describe().T)

st.markdown("---")
st.markdown("### Сравнительный анализ точности моделей на данном датасете")
st.markdown("Результаты оценки качества аппроксимации целевой переменной по коэффициенту детерминации (R²):")

metrics_data = {
    "Архитектура модели": [
        "GradientBoostingRegressor (ML2)",
        "StackingRegressor (ML5)",
        "NeuralNetwork / MLPRegressor (ML6)",
        "CatBoostRegressor (ML3)",
        "RandomForestRegressor (ML4)",
        "HuberRegressor (ML1)"
    ],
    "Метрика точности R²": [0.8576, 0.8557, 0.8546, 0.8537, 0.8428, 0.7931]
}

df_metrics = pd.DataFrame(metrics_data)
st.table(df_metrics)
