import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_parquet('data/cardio_data_processed.parquet')

def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    coluna = st.selectbox('Selecione a coluna', ['Fumantes', 'Ativo', 'Álcool'])
    st.write(f'**Distribuição de {coluna}**')
    if coluna == 'Fumantes':
        fig_pie = px.pie(df, names = 'smoke')
        st.plotly_chart(fig_pie)
    elif coluna == 'Ativo':
        fig_pie = px.pie(df, names = 'active')
        st.plotly_chart(fig_pie)
    else:
        fig_pie = px.pie(df, names ='alco')
        st.plotly_chart(fig_pie)

grafico_pizza()