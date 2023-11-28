import pandas as pd
import streamlit as st
import plotly.express as px

value = 'Yes'
df = pd.read_parquet('data/cvd_cleaned.parquet').query('Heart_Disease == @value')
df = df.drop_duplicates()


def grafico_pizza():
    st.write('**Gráfico de Pizza**')
    st.write(
    '''<p>Nesses gráficos consideramos apenas pessoas que <b>POSSUEM</b> doenças cardiovasculares</p>
    ''', unsafe_allow_html=True)
    coluna = st.selectbox('Selecione a coluna', ['Depressão', 'Diabetes', 'Artrite', 'Câncer de pele'])
    st.write(f'**Distribuição de {coluna}**')
    if coluna == 'Depressão':
        fig_pie = px.pie(df, names = 'Depression')
        st.plotly_chart(fig_pie)
    elif coluna == 'Diabetes':
        fig_pie = px.pie(df, names = 'Diabetes')
        st.plotly_chart(fig_pie)
    elif coluna == 'Artrite':
        fig_pie = px.pie(df, names ='Arthritis')
        st.plotly_chart(fig_pie)
    elif coluna == 'Câncer de pele':
        fig_pie = px.pie(df, names = 'Skin_Cancer')
        st.plotly_chart(fig_pie)
grafico_pizza()

st.write('----')

st.write('Gráfico Empilhado')

def barra_empilhada():
    cols_to_melt = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis']
    df_selected = df[cols_to_melt]

    df_selected['Diabetes'] = df_selected['Diabetes'].replace({
        'Yes, but female told only during pregnancy': 'Yes',
        'No, pre-diabetes or borderline diabetes': 'No'
    })

    df_melted = df_selected.melt(id_vars=['Heart_Disease'], var_name='outra_doenca', value_name='status')

    df_melted['outra_doenca'] = df_melted['outra_doenca'].map({
        'Skin_Cancer': 'Câncer de Pele',
        'Other_Cancer': 'Outros Cânceres',
        'Depression': 'Depressão',
        'Diabetes': 'Diabetes',
        'Arthritis': 'Artrite'
    })

    fig_stacked_bar = px.histogram(df_melted, x='outra_doenca', color='status',
                                    labels={'outra_doenca': 'Outras Doença', 'status': 'Status'},
                                    barmode='stack')

    st.plotly_chart(fig_stacked_bar)

barra_empilhada()