import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_csv('data/cardio_data_processed.csv')

def grafico_barra():
    st.write('**Gráfico de Barra**')
    stacked_data = df.groupby(["cholesterol", "gluc"]).size().reset_index(name="count")
    fig_stacked = px.bar(stacked_data, x = "cholesterol", y = "count", color = "gluc", barmode = "stack", title = "Níveis de Colesterol e Glicose")
    st.plotly_chart(fig_stacked)

def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    coluna = st.selectbox('Selecione a coluna', ['Colesterol', 'Glicose'])
    st.write(f'**Distribuição de {coluna}**')
    if coluna == 'Colesterol':
        fig_pie = px.pie(df, names = 'cholesterol')
        st.plotly_chart(fig_pie)
    else:
        fig_pie = px.pie(df, names = 'gluc')
        st.plotly_chart(fig_pie)

def grafico_dispersao():
    st.write('**Gráfico de Dispersão**')
    fig_scatter = px.scatter(df, x = 'ap_hi', y = 'ap_lo', title = 'Pressão Sanguínea Sistólica (ap_hi) e Diastólica (ap_lo)')
    st.plotly_chart(fig_scatter)


escolha = st.selectbox('**Selecione um gráfico para vizualizar**', ['Barra', 'Pizza', 'Dispersão'])

if escolha == 'Barra':
    st.write('')
    grafico_barra()
elif escolha == 'Pizza':
    st.write('')
    grafico_pizza()
else:
    st.write('')
    grafico_dispersao()