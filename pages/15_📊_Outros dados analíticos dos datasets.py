import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

value = 1
df_1 = pd.read_parquet('data/cardio_data_processed.parquet').query('cardio == @value')
df_2 = pd.read_parquet('data/cvd_cleaned.parquet').query('Heart_Disease == @value')

smoke_alco_no_exercise_count = df_1[(df_1['smoke'] == 1) & (df_1['alco'] == 1) & (df_1['active'] == 0)]
smoke_alco_no_exercise_count = df_1[(df_1['smoke'] == 1) & (df_1['alco'] == 1) & (df_1['active'] == 0)]

st.write('''Algumas informações quantitativas dos dados utilizados:''', unsafe_allow_html=True)

st.write(f"Quantitativo de indivíduos que bebem, fumam, são sedentários e possuem algum problema cardíaco é de:  <span style='color:red;'>{len(smoke_alco_no_exercise_count)}</span>", unsafe_allow_html=True)






