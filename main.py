import streamlit as st

# Ajuste da página geral
st.set_page_config(
    page_title = 'Projeto',
    page_icon = ':books:',
    layout = 'wide',
    menu_items= {
        'Get help': "https://streamlit.io",
        'Report a Bug': "https://blog.streamlit.io",
        'About': "Este é um projeto estudantil feito por alunos da **UFRPE**."
        } # type: ignore
)

# Criação de um cabeçalho
st.markdown('''# **Projeto científico**

Por `arthur`, `emanoel`, `lipe` e `lucas`
---
''', unsafe_allow_html=True)