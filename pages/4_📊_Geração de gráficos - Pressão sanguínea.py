import pandas as pd
import streamlit as st
import plotly.express as px

value = 1
df = pd.read_parquet('data/cardio_data_processed.parquet').query('cardio == @value')
df = df.rename(columns={'bp_category': 'Categorias'})

def grafico_dispersao():
    st.write('**Gráfico de Dispersão**')

    fig_scatter = px.scatter(df, x = 'ap_hi', y = 'ap_lo', title = 'Pressão Sanguínea Sistólica (ap_hi) e Diastólica (ap_lo)')
    st.plotly_chart(fig_scatter)

def grafico_barra():
    st.write('**Gráfico de Barras**')
    
    df['Categorias'] = df['Categorias'].replace({
            'Hypertension Stage 1': 'Hipertensão Nível 1',
            'Hypertension Stage 2': 'Hipertensão Nível 2',
            'Elevated': 'Elevado'
        })
    
    category_counts = df['Categorias'].value_counts().reset_index()
    category_counts.columns = ['Categorias', 'count']
    category_counts = category_counts.sort_values(by = 'count', ascending = False)

    color_scale = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig_stacked = px.bar(category_counts, x = 'Categorias', y = 'count', color = 'Categorias', color_discrete_sequence=color_scale, labels = {"count": "Contagem"}, category_orders = {"Categorias": category_counts['Categorias'].tolist()})
    st.plotly_chart(fig_stacked)

def grafico_setor():
    st.write('**Gráfico de Setores**')

    df['Categorias'] = df['Categorias'].replace({
            'Hypertension Stage 1': 'Hipertensão Nível 1',
            'Hypertension Stage 2': 'Hipertensão Nível 2',
            'Elevated': 'Elevado'
        })
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig_donut = px.pie(df, names = 'Categorias',color_discrete_sequence=colors, hole = 0.3)
    

    #fig_donut.update_traces(marker = dict(colors = colors))
    st.plotly_chart(fig_donut)

st.write('''Dados relativos ao Dataset: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease?select=cardio_data_processed.csv)''', unsafe_allow_html=True)

escolha = st.selectbox('**Selecione um gráfico para vizualizar**', ['Dispersão', 'Setores', 'Barra'])

if escolha == 'Dispersão':
    st.write('')
    grafico_dispersao()
elif escolha == 'Setores':
    st.write('')
    grafico_setor()
else:
    st.write('')
    grafico_barra()