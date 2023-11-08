import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_parquet('data/cardio_data_processed.parquet')

df_knn = df[['id', 'age_years','cholesterol', 'gluc','smoke', 'alco', 'active', 'cardio', 'bmi', 'bp_category_encoded']]

df_knn['cardio'] = LabelEncoder().fit_transform(df_knn['cardio'])

st.sidebar.header('Configurações do Gráfico')
x_column = st.sidebar.selectbox('Selecione a coluna para o eixo X', df_knn.columns)
y_column = st.sidebar.selectbox('Selecione a coluna para o eixo Y', df_knn.columns)

X = df[[x_column, y_column]]
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

prediction = knn.predict(X)

color_scale = px.colors.sequential.Viridis

st.write('**Dataframe Codificado**')
st.dataframe(df_knn)
st.write('')
st.write('----')

st.write('**Classificação do Dataframe**')
fig = px.scatter(df_knn, x = x_column, y = y_column, color = df_knn['cardio'], color_continuous_scale = color_scale, hover_name = 'id')
st.plotly_chart(fig)
st.write('----')

plt.scatter(x_column, y_column, c=prediction[0])
plt.text(x=x_column-1.7, y=y_column-0.7, s=f"new point, class: {prediction[0]}")
plt.show()