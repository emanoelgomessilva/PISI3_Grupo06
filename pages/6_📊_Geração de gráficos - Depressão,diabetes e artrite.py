import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_parquet('data/cvd_cleaned.parquet')

def grafico_heatmap():
    st.write('**Gráfico Heatmap**')
    heatmap_colunas = ['Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis']

    df_heat = df[heatmap_colunas].apply(pd.to_numeric, errors='coerce')

    correlation_matrix = df_heat.corr()

    fig = px.imshow(correlation_matrix, labels=dict(x="Variáveis", y="Variáveis"), x=correlation_matrix.index, y=correlation_matrix.columns)
    st.plotly_chart(fig, use_container_width=True)

grafico_heatmap()
