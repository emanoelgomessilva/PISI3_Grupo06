import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

value = 1
df = pd.read_parquet('data/cardio_data_processed.parquet').query('cardio == @value')

st.write('''Relação comparativa entre a população fumante e não fumante que possui algum problema cardíaco nos dados utilizados''', unsafe_allow_html=True)

def grafico_barra():
    
    # Contar as ocorrências dos valores na coluna
    contagem = df['smoke'].value_counts()

    # Plotar o gráfico de barras
    plt.bar(contagem.index, contagem.values)
    plt.xlabel('Valores')
    plt.ylabel('Quantidade de Ocorrências')
    plt.title('Ocorrências de Valores na Coluna')
    plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para facilitar a leitura
    plt.show()

    contagem.index = ['Não fumantes', 'Fumantes']

    # Exibir o gráfico de barras no Streamlit
    st.bar_chart(contagem)

st.write('')
grafico_barra()


