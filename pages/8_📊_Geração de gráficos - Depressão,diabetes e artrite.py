import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_parquet('data/cvd_cleaned.parquet')

def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    coluna = st.selectbox('Selecione a coluna', ['Depressão', 'Diabetes', 'Artrite'])
    st.write(f'**Distribuição de {coluna}**')
    if coluna == 'Depressão':
        fig_pie = px.pie(df, names = 'Depression')
        st.plotly_chart(fig_pie)
    elif coluna == 'Diabetes':
        fig_pie = px.pie(df, names = 'Diabetes')
        st.plotly_chart(fig_pie)
    else:
        fig_pie = px.pie(df, names ='Arthritis')
        st.plotly_chart(fig_pie)

grafico_pizza()