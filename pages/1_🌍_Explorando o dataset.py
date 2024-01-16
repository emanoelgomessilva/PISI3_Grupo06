import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


st.write('<h1>ANÁLISE EXPLORATÓRIA</h1>'+\
    '''<p>Trazemos nessa página uma primeira visão e análise dos nossos datasets, em que constam
    a visão em tabela dos dados de estudo e o Perfil (com gráficos e informações úteis) do dataset em si </p>
    <p>Lib utilizada <b>YData Profiling</b>(antigo <i>Pandas Profiling</i> <code>v=4.6.1</code>).</p>
    ''', unsafe_allow_html=True)

def profile():
    df = pd.read_parquet('data/cardio_data_processed.parquet')
    st.dataframe(df)
    profile = ProfileReport(df, title="Cardio_profile")
    profile.to_file(f"reports/Cardio_profile.html")
    st.session_state['Cardio_profile'] = df

def print_profile():
    st.write(f'Dataset: <i>Cardio_profile</i>', unsafe_allow_html=True)
    report_file = open(f'reports/Cardio_profile.html', 'r', encoding='utf-8')
    source_code = report_file.read() 
    components.html(source_code, height=600,  scrolling=True)

def profile2():
    df = pd.read_parquet('data\cvd_cleaned.parquet')
    st.dataframe(df)
    profile = ProfileReport(df, title="CVD_profile")
    profile.to_file(f"reports/CVD_profile.html")
    st.session_state['CVD_profile'] = df

def print_profile2():
    st.write(f'Dataset: <i>CVD_profile</i>', unsafe_allow_html=True)
    report_file = open(f'reports/CVD_profile.html', 'r', encoding='utf-8')
    source_code = report_file.read() 
    components.html(source_code, height=600,  scrolling=True)


profile()
print_profile()
st.write('----')
profile2()
print_profile2()