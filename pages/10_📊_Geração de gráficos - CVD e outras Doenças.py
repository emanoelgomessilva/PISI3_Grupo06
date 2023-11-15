import pandas as pd
import streamlit as st
import plotly.express as px

value = 'Yes'
df = pd.read_parquet('data/cvd_cleaned.parquet').query('Heart_Disease == @value')
df = df.drop_duplicates()


def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    st.write(
    '''<p>Nesses gráficos consideramos apenas pessoas que <b>POSSUEM</b> doenças cardiovasculares</p>
    ''', unsafe_allow_html=True)
    coluna = st.selectbox('Selecione a coluna', ['Depressão', 'Diabetes', 'Artrite', 'Câncer de pele'])
    st.write(f'**Distribuição de {coluna}**')
    if coluna == 'Depressão':
        fig_pie = px.pie(df, names = 'Depression')
        st.plotly_chart(fig_pie)
    elif coluna == 'Diabetes':
        fig_pie = px.pie(df, names = 'Diabetes')
        st.plotly_chart(fig_pie)
    elif coluna == 'Artrite':
        fig_pie = px.pie(df, names ='Arthritis')
        st.plotly_chart(fig_pie)
    elif coluna == 'Câncer de pele':
        fig_pie = px.pie(df, names = 'Skin_Cancer')
        st.plotly_chart(fig_pie)
grafico_pizza()