import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import streamlit as st


st.write('<h1>ANÁLISE EXPLORATÓRIA</h1>', unsafe_allow_html=True)

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
    components.html(source_code, height=400,  scrolling=True)

profile()
print_profile()