import streamlit as st

# Ajuste da página geral
st.set_page_config(
    page_title = 'Projeto',
    page_icon = ':books:',
    layout = 'wide',
    menu_items= {
        'Get help': "https://streamlit.io",
        'Report a Bug': "https://blog.streamlit.io",
        'About': '''Este é um projeto estudantil feito por alunos da **UFRPE**.
        Acesse: bsi.ufrpe.br
        '''
        } # type: ignore
)

# Criação de um cabeçalho
st.markdown('''
# **Projeto Científico: Análise de Doenças Cardiovasculares**

Este é um trabalho científico voltado à análise de dados, usando um <i>dataset</i> de doenças cardiovasculares.

O projeto se concentra no uso de algoritmos de clusterização e classificação para entender padrões e prever a presença de doenças cardiovasculares em pacientes.            

**Membros do Projeto:**
- `Arthur` (arthur.bbsantos@ufrpe.br)
- `Emanoel` (emanoel20092009@gmail.com)
- `Fellipe` (pipo200115@gmail.com)
- `Lucas` (lucas.dan.melo@gmail.com)
- `Renan` (renanneji1994@gmail.com)
            
**Fonte**:
- [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease?select=cardio_data_processed.csv)
---
''', unsafe_allow_html=True)