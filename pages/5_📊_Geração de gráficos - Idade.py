import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_parquet('data/cardio_data_processed.parquet')

def grafico_bubble():
    st.write('**Gráfico Bubble**')
    bins = [19, 24, 29, 34, 39, 44, 49, 54, 59, 64]
    labels = [f'{i}-{j}' for i, j in zip(bins[:-1], bins[1:])]
    df['idade'] = pd.cut(df['age_years'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')

    grouped_data = df.groupby(['idade', 'gender', 'cardio'], observed=True).size().reset_index(name='count')

    fig = px.scatter(grouped_data, x='idade', y='gender', text='idade', color='cardio', size='count', size_max=100, opacity=0.75, color_discrete_map={0: 'blue', 1: 'red'}, labels={'gender': 'Gênero', 'cardio': 'Doença Cardiovascular'})
    fig.update_layout(xaxis_title='Idade', yaxis_title='Gênero', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def grafico_boxplot():
    st.write('**Gráfico Boxplot**')
    df_plot = df[['gender', 'age_years']]

    fig = px.box(df_plot, x='gender', y='age_years')
    fig.update_layout(xaxis_title='Gênero', yaxis_title='Idade', showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

def grafico_heatmap():
    st.write('**Gráfico Heatmap**')
    heatmap_colunas = ['age_years', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']

    df_heat = df[heatmap_colunas]
    correlation_matrix = df_heat.corr()

    fig = px.imshow(correlation_matrix, labels=dict(x="Variáveis", y="Variáveis"), x=correlation_matrix.index, y=correlation_matrix.columns)
    st.plotly_chart(fig, use_container_width=True)

escolha = st.selectbox('**Selecione um gráfico para vizualizar**', ['Bubble', 'Boxplot', 'Heatmap'])

if escolha == 'Bubble':
    st.write('')
    grafico_bubble()
elif escolha == 'Boxplot':
    st.write('')
    grafico_boxplot()
else:
    st.write('')
    grafico_heatmap()