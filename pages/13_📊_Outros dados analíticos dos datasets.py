import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

value = 1
value_2 = 'Yes'
df_1 = pd.read_parquet('data/cardio_data_processed.parquet').query('cardio == @value')
df_2 = pd.read_parquet('data/cvd_cleaned.parquet').query('Heart_Disease == @value_2')

smoke_alco_no_exercise_count = df_1[(df_1['smoke'] == 1) & (df_1['alco'] == 1) & (df_1['active'] == 0)]
no_smoke_no_alco_exercise_count = df_1[(df_1['smoke'] == 0) & (df_1['alco'] == 0) & (df_1['active'] == 1)]
cholesterol_3_gluc_3_count = df_1[(df_1['cholesterol'] == 3) & (df_1['gluc'] == 3)]
cholesterol_1_gluc_1_count = df_1[(df_1['cholesterol'] == 1) & (df_1['gluc'] == 1)]
smoking_alcohol_no_exercise_count = df_2[(df_2['Smoking_History'] == 'Yes') & (df_2['Alcohol_Consumption'] > 0) & (df_2['Exercise'] == 'No')]
no_smoking_no_alcohol_exercise_count = df_2[(df_2['Smoking_History'] == 'No') & (df_2['Alcohol_Consumption'] == 0) & (df_2['Exercise'] == 'Yes')]
no_smoking_no_alcohol_exercise_count = df_2[(df_2['Smoking_History'] == 'No') & (df_2['Alcohol_Consumption'] == 0) & (df_2['Fruit_Consumption'] > 0) & (df_2['Green_Vegetables_Consumption'] > 0) & (df_2['Exercise'] == 'Yes')]



st.write('<h1>Algumas informações quantitativas dos dados utilizados:</h1>', unsafe_allow_html=True)

st.write('''Dados relativos ao Dataset: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease?select=cardio_data_processed.csv)''', unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que bebem, fumam, são sedentários e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(smoke_alco_no_exercise_count)}</span>", unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que não bebem, não fumam, praticam alguma atividade física e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(no_smoke_no_alco_exercise_count)}</span>", unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que possuem alto nível de glicose, alto nível colesterol e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(cholesterol_3_gluc_3_count)}</span>", unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que possuem baixo nível de glicose, baixo nível colesterol e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(cholesterol_1_gluc_1_count)}</span>", unsafe_allow_html=True)

st.write('''Dados relativos ao Dataset: [Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset/data)''', unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que bebem, fumam, são sedentários e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(smoking_alcohol_no_exercise_count)}</span>", unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que não bebem, não fumam, praticam alguma atividade física e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(no_smoking_no_alcohol_exercise_count)}</span>", unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que não bebem, não fumam, praticam alguma atividade física, tem inclusos frutas e vegetais em sua alimentação e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(no_smoking_no_alcohol_exercise_count)}</span>", unsafe_allow_html=True)



