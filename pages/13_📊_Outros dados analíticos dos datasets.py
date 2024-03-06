import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

df_1 = pd.read_parquet('data/cardio_data_processed.parquet')
df_2 = pd.read_parquet('data/cvd_cleaned.parquet')

smoke_alco_no_exercise_count = df_1[(df_1['smoke'] == 1) & (df_1['alco'] == 1) & (df_1['active'] == 0)]
smoke_alco_no_exercise_count = smoke_alco_no_exercise_count['cardio'].value_counts()
no_smoke_no_alco_exercise_count = df_1[(df_1['smoke'] == 0) & (df_1['alco'] == 0) & (df_1['active'] == 1)]
no_smoke_no_alco_exercise_count = no_smoke_no_alco_exercise_count['cardio'].value_counts()
cholesterol_3_gluc_3_count = df_1[(df_1['cholesterol'] == 3) & (df_1['gluc'] == 3)]
cholesterol_3_gluc_3_count = cholesterol_3_gluc_3_count['cardio'].value_counts()
cholesterol_1_gluc_1_count = df_1[(df_1['cholesterol'] == 1) & (df_1['gluc'] == 1)]
cholesterol_1_gluc_1_count = cholesterol_1_gluc_1_count['cardio'].value_counts()
smoking_alcohol_no_exercise_count = df_2[(df_2['Smoking_History'] == 'Yes') & (df_2['Alcohol_Consumption'] > 0) & (df_2['Exercise'] == 'No')]
smoking_alcohol_no_exercise_count = smoking_alcohol_no_exercise_count['Heart_Disease'].value_counts()
no_smoking_no_alcohol_exercise_count = df_2[(df_2['Smoking_History'] == 'No') & (df_2['Alcohol_Consumption'] == 0) & (df_2['Exercise'] == 'Yes')]
no_smoking_no_alcohol_exercise_count = no_smoking_no_alcohol_exercise_count['Heart_Disease'].value_counts()
no_smoking_no_alcohol_exercise_vegetables_count = df_2[(df_2['Smoking_History'] == 'No') & (df_2['Alcohol_Consumption'] == 0) & (df_2['Fruit_Consumption'] > 0) & (df_2['Green_Vegetables_Consumption'] > 0) & (df_2['Exercise'] == 'Yes')]
no_smoking_no_alcohol_exercise_vegetables_count = no_smoking_no_alcohol_exercise_vegetables_count['Heart_Disease'].value_counts()


st.write('<h1>Algumas informações quantitativas dos dados utilizados:</h1>', unsafe_allow_html=True)

st.write('''Dados relativos ao Dataset: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease?select=cardio_data_processed.csv)''', unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que bebem, fumam, são sedentários e possuem algum problema cardíaco", unsafe_allow_html=True)

plt.bar(smoke_alco_no_exercise_count.index, smoke_alco_no_exercise_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)
plt.show()

smoke_alco_no_exercise_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(smoke_alco_no_exercise_count)

st.write(f"Quantitativo de indivíduos que não bebem, não fumam, praticam alguma atividade física e possuem algum problema cardíaco", unsafe_allow_html=True)

plt.bar(no_smoke_no_alco_exercise_count.index, no_smoke_no_alco_exercise_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
plt.show()

no_smoke_no_alco_exercise_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(no_smoke_no_alco_exercise_count)

st.write(f"Quantitativo de indivíduos que possuem alto nível de glicose, alto nível colesterol e possuem algum problema cardíaco", unsafe_allow_html=True)

plt.bar(cholesterol_3_gluc_3_count.index, cholesterol_3_gluc_3_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
plt.show()

cholesterol_3_gluc_3_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(cholesterol_3_gluc_3_count)

st.write(f"Quantitativo de indivíduos que possuem baixo nível de glicose, baixo nível colesterol e possuem algum problema cardíaco", unsafe_allow_html=True)

plt.bar(cholesterol_1_gluc_1_count.index, cholesterol_1_gluc_1_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
plt.show()

cholesterol_1_gluc_1_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(cholesterol_1_gluc_1_count)

st.write('''Dados relativos ao Dataset: [Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset/data)''', unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que bebem, fumam, são sedentários e possuem algum problema cardíaco", unsafe_allow_html=True)

plt.bar(smoking_alcohol_no_exercise_count.index, smoking_alcohol_no_exercise_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
plt.show()

smoking_alcohol_no_exercise_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(smoking_alcohol_no_exercise_count)

st.write(f"Quantitativo de indivíduos que não bebem, não fumam, praticam alguma atividade física e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(no_smoking_no_alcohol_exercise_count)}</span>", unsafe_allow_html=True)

plt.bar(no_smoking_no_alcohol_exercise_count.index, no_smoking_no_alcohol_exercise_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
plt.show()

no_smoking_no_alcohol_exercise_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(no_smoking_no_alcohol_exercise_count)

st.write(f"Quantitativo de indivíduos que não bebem, não fumam, praticam alguma atividade física, tem inclusos frutas e vegetais em sua alimentação e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(no_smoking_no_alcohol_exercise_count)}</span>", unsafe_allow_html=True)

plt.bar(no_smoking_no_alcohol_exercise_vegetables_count.index, no_smoking_no_alcohol_exercise_vegetables_count.values)
plt.xlabel('Valores')
plt.ylabel('Quantidade de Ocorrências')
plt.title('Ocorrências de Valores na Coluna')
plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
plt.show()

no_smoking_no_alcohol_exercise_vegetables_count.index = ['Não cardíacos', 'Cardíacos']

st.bar_chart(no_smoking_no_alcohol_exercise_vegetables_count)

