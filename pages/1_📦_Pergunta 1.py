import streamlit as st
import mysql.connector as cn

conexao = cn.connect(
    host="localhost",
    user="root",
    password="root",
    database="tabelao"
)

cursor = conexao.cursor()

consulta_sql = "SELECT * FROM convenente"
cursor.execute(consulta_sql)

resultados = cursor.fetchall()

for resultado in resultados:
    st.write(resultado)

st.write('<h1>Pergunta 1</h1>', unsafe_allow_html=True)
st.write('''Texto da pergunta 1''', unsafe_allow_html=True)

def builder_body():
    Kmeans()

def Kmeans():

    st.sidebar.header('Configurações do Gráfico')
    filtro_1 = st.sidebar.text_input('Filtro 1')
    filtro_2 = st.sidebar.text_input('Filtro 2')

builder_body()