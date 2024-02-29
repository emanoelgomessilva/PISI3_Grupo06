import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_parquet('data/cvd_cleaned.parquet')

def codificar_categorico(valor):
    if valor == 'Yes':
        return 1
    elif valor == 'No':
        return 0
    elif valor == 'No, pre-diabetes or borderline diabetes':
        return 0
    elif valor == 'Yes, but female told only during pregnancy':
        return 1

def grafico_heatmap():
    st.write('**Gráfico Heatmap**')
    heatmap_colunas = ['Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis']

    df_heat = df[heatmap_colunas].applymap(codificar_categorico)

    correlation_matrix = df_heat.corr()

    fig = px.imshow(correlation_matrix, labels=dict(x="Variáveis", y="Variáveis"), x=correlation_matrix.index, y=correlation_matrix.columns)
    st.plotly_chart(fig, use_container_width=True)

st.write('''Dados relativos ao Dataset: [Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset/data)''', unsafe_allow_html=True)

grafico_heatmap()
