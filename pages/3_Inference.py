import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Прогноз", layout="wide")
st.title("Вывод (инференс) предобученных моделей")
st.markdown("---")

feature_names = [
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
    'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
    'Type_H', 'Type_L', 'Type_M'
]

selected_model_name = st.sidebar.selectbox(
    "Выберите архитектуру ML:",
    ["ML2_GradientBoosting", "ML5_Stacking", "ML6_NeuralNetwork", "ML3_CatBoost", "ML4_RandomForest", "ML1_Huber"]
)

metrics_db = {
    "ML2_GradientBoosting": 0.8576, "ML5_Stacking": 0.8557, "ML6_NeuralNetwork": 0.8546,
    "ML3_CatBoost": 0.8537, "ML4_RandomForest": 0.8428, "ML1_Huber": 0.7931
}
st.sidebar.metric("Точность выбранной модели (R²)", f"{metrics_db[selected_model_name]:.4f}")

# БЛОК 1: Ручной ввод с валидацией диапазонов и единицами измерения
st.markdown("### 1. Ручной ввод физических параметров работы оборудования")

col1, col2, col3, col4 = st.columns(4)

with col1:
    air_temp = st.number_input("Температура воздуха [K]", min_value=280.0, max_value=320.0, value=298.1, step=0.1)
with col2:
    proc_temp = st.number_input("Температура процесса [K]", min_value=290.0, max_value=330.0, value=308.6, step=0.1)
with col3:
    rot_speed = st.number_input("Скорость вращения [об/мин]", min_value=500, max_value=4000, value=1500, step=10)
with col4:
    tool_wear = st.slider("Износ инструмента [мин]", min_value=0, max_value=300, value=50)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Рассчитать крутящий момент", type="primary"):
    # Валидация логики физического процесса перед расчетом
    if air_temp > proc_temp:
        st.warning("Внимание: Температура окружающего воздуха превышает температуру технологического процесса.")
        
    input_row = pd.DataFrame([[
        air_temp, proc_temp, rot_speed, tool_wear, 
        0, 0, 0, 0, 0, 0, 0, 0, 0
    ]], columns=feature_names)
    
    try:
        with open(f"models/{selected_model_name}.pkl", "rb") as f:
            pipeline = pickle.load(f)
            
        prediction = pipeline.predict(input_row)[0]
        
        st.success("Расчет успешно выполнен")
        st.metric(label="Предсказанная величина крутящего момента шпинделя", value=f"{prediction:.2f} Н·м")
        
    except Exception as e:
        st.error(f"Ошибка при расчете: {e}")

# БЛОК 2: Загрузка файла *.csv для пакетного прогнозирования
st.markdown("---")
st.markdown("### 2. Пакетный расчет из файла конфигураций (*.csv)")

uploaded_file = st.file_uploader("Выберите файл в формате .csv для массового расчета", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    
    # Проверка наличия обязательных столбцов для валидации структуры
    missing_cols = [col for col in feature_names if col not in batch_df.columns]
    
    if len(missing_cols) == 0:
        st.success("Структура файла успешно валидирована.")
        
        if st.button("Выполнить расчет для таблицы данных"):
            try:
                with open(f"models/{selected_model_name}.pkl", "rb") as f:
                    pipeline = pickle.load(f)
                
                # Подача данных в модель в строгом соответствии с порядком признаков
                preds = pipeline.predict(batch_df[feature_names])
                
                result_df = batch_df.copy()
                result_df['Predicted_Torque [Nm]'] = preds
                
                st.markdown("#### Результаты прогнозирования (первые 5 строк):")
                st.dataframe(result_df.head(5))
                
                # Экспорт результатов обратно
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать полученные прогнозы в формате CSV",
                    data=csv_data,
                    file_name="predictions_output.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Ошибка при обработке таблицы: {e}")
    else:
        st.error(f"Ошибка валидации: В файле отсутствуют необходимые столбцы: {missing_cols}")