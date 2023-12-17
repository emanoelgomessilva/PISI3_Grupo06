import streamlit as st

# Ajuste da página geral
st.set_page_config(
    page_title = 'Convênios',
    page_icon = ':books:',
    layout = 'wide',
    menu_items= {
        'Get help': "https://streamlit.io",
        'Report a Bug': "https://blog.streamlit.io",
        'About': '''Este é um projeto estudantil feito por alunos da **UFRPE**.
        Acesse: [bsi.ufrpe.br](https://sites.google.com/view/bsi-ufrpe)
        '''
        } # type: ignore
)

# Criação de um cabeçalho
st.markdown('''
# **Modelagem de dados para a base de Convênios do governo federal**

Este é um projeto que utiliza a base de dados de Convênios do governo federal para a construção de um Data Ware House

**Membros do Projeto:**
- `Aurineque`
- `Emanoel Gomes` (emanoel20092009@gmail.com)
- `Guilherme`
- `Júlia`
- `Pedro`
---
''', unsafe_allow_html=True)