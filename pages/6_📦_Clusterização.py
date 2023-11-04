import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_parquet('data/cardio_data_processed.parquet')

#def kmeans():
df_kmeans = df[['id', 'age_years','cholesterol', 'gluc','smoke', 'alco', 'active', 'cardio', 'bmi', 'bp_category_encoded']]

df_kmeans['bp_category_encoded'] = LabelEncoder().fit_transform(df_kmeans['bp_category_encoded'])

df_kmeans = df_kmeans.rename(columns={
    'age_years': 'Idade', 'bp_category_encoded': 'Pressão Sanguínea', 
    'cholesterol': 'Colesterol', 'gluc': 'Glicose',
    'alco': 'Consumo de Álcool', 'active': 'Atividade Física', 
    'smoke': 'Tabagismo', 'bmi': 'IMC',
    'cardio': 'Cardio'})

st.sidebar.header('Configurações do Gráfico')
x_column = st.sidebar.selectbox('Selecione a coluna para o eixo X', df_kmeans.columns)
y_column = st.sidebar.selectbox('Selecione a coluna para o eixo Y', df_kmeans.columns)

X = df_kmeans[[x_column, y_column]]
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters = 4, random_state = 4294967295)
df_kmeans['cluster'] = kmeans.fit_predict(X)

color_scale = px.colors.sequential.Viridis

st.write('**Dataframe Codificado**')
st.dataframe(df_kmeans)
st.write('')
st.write('----')

st.write('**Clusterização do Dataframe**')
fig = px.scatter(df_kmeans, x = x_column, y = y_column, color = df_kmeans['cluster'], color_continuous_scale = color_scale, hover_name = 'id')
st.plotly_chart(fig)
st.write('----')