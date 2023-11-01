import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_parquet('data/cardio_data_processed.parquet')

def grafico_barra():
    cholesterol_labels = {1:'Normal', 2:'Acima do Normal', 3:'Muito Acima do Normal'}
    gluc_labels = {1:'Normal', 2:'Acima do Normal', 3:'Muito Acima do Normal'}
    
    fig_cholesterol = go.Figure()
    fig_gluc = go.Figure()

    stacked_df_cholesterol = df[df['cholesterol'] == df['cholesterol'].unique()[0]].groupby('gluc').size().reset_index(name = 'count')
    for idx, row in stacked_df_cholesterol.iterrows():
        fig_cholesterol.add_trace(go.Bar(x = [cholesterol_labels[row['gluc']]], y = [row['count']], name = f"Colesterol {cholesterol_labels[row['gluc']]}", text = [row['count']], textposition = 'inside'))

    stacked_df_gluc = df[df['gluc'] == df['gluc'].unique()[0]].groupby('cholesterol').size().reset_index(name = 'count')
    for idx, row in stacked_df_gluc.iterrows():
        fig_gluc.add_trace(go.Bar(x = [gluc_labels[row['cholesterol']]], y = [row['count']], name = f"Glicose {gluc_labels[row['cholesterol']]}", text = [row['count']], textposition = 'inside'))

    fig_cholesterol.update_layout(title = 'Níveis de Colesterol')
    fig_gluc.update_layout(title = 'Níveis de Glicose')

    st.plotly_chart(fig_cholesterol)
    st.plotly_chart(fig_gluc)

def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    coluna = st.selectbox('Selecione a coluna', ['Colesterol', 'Glicose'])
    st.write(f'**Porcentagens de {coluna}**')
    
    if coluna == 'Colesterol':
        fig_pie = px.pie(df, names = 'cholesterol')
        st.plotly_chart(fig_pie)
    else:
        fig_pie = px.pie(df, names = 'gluc')
        st.plotly_chart(fig_pie)

escolha = st.selectbox('**Selecione um gráfico para vizualizar**', ['Barra', 'Pizza'])

if escolha == 'Barra':
    st.write('')
    grafico_barra()
else:
    st.write('')
    grafico_pizza()