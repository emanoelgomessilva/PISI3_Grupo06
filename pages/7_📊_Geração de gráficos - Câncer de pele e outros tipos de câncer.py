import pandas as pd
import streamlit as st
import plotly.express as px

value = 'Yes'
df = pd.read_parquet('data/cvd_cleaned.parquet').query('Heart_Disease == @value')
df = df.drop_duplicates()

st.write('''Dados relativos ao Dataset: [Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset/data)''', unsafe_allow_html=True)

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

skin_cancer_count = df[df['Skin_Cancer'] == 'Yes']
other_cancer_count = df[df['Other_Cancer'] == 'Yes']

st.write('**Dados quantitativos com relação à indivíduos com algum tipo de câncer e algum problema cardíaco:**')
st.write(f"Quantitativo de indivíduos que possuem câncer de pele e algum problema cardíaco é de:  <span style='color:red;'>{len(skin_cancer_count)}</span>", unsafe_allow_html=True)
st.write(f"Quantitativo de indivíduos que possuem outros tipos de câncer e algum problema cardíaco é de:  <span style='color:red;'>{len(skin_cancer_count)}</span>", unsafe_allow_html=True)
