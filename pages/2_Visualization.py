import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Визуализация", layout="wide")
st.title("Результаты графического анализа данных")

df = pd.read_csv("R3_dataset_processed.csv")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**1. Распределение целевого признака (Гистограмма с KDE)**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['Torque [Nm]'], kde=True, color='skyblue', ax=ax1)
    ax1.set_xlabel("Крутящий момент [Н·м]")
    ax1.set_ylabel("Частота")
    st.pyplot(fig1)

with col2:
    st.markdown("**2. Взаимосвязь скорости вращения и момента (Диаграмма рассеяния)**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df.sample(500, random_state=42), x='Rotational speed [rpm]', y='Torque [Nm]', color='coral', alpha=0.7, ax=ax2)
    ax2.set_xlabel("Скорость вращения [об/мин]")
    ax2.set_ylabel("Крутящий момент [Н·м]")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**3. Матрица линейной корреляции признаков (Тепловая карта)**")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax3, cbar=False)
    st.pyplot(fig3)

with col4:
    st.markdown("**4. Анализ момента при отказах оборудования (Диаграмма Ящик с усами)**")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Machine failure', y='Torque [Nm]', palette='Set2', ax=ax4)
    ax4.set_xlabel("Технический сбой (0 - нет, 1 - да)")
    ax4.set_ylabel("Крутящий момент [Н·м]")
    st.pyplot(fig4)