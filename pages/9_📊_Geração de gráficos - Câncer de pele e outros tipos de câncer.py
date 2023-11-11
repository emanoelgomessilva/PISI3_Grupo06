import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_parquet('data/cvd_cleaned.parquet')

def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    coluna = st.selectbox('Selecione a coluna', ['Câncer de pele', 'Outros tipos de câncer'])
    st.write(f'**Distribuição de {coluna}**')
    if coluna == 'Câncer de pele':
        fig_pie = px.pie(df, names = 'Skin_Cancer')
        st.plotly_chart(fig_pie)
    else:
        fig_pie = px.pie(df, names ='Other_Cancer')
        st.plotly_chart(fig_pie)

grafico_pizza()