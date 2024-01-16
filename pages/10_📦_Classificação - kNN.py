import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df = pd.read_parquet('data/cvd_cleaned.parquet')
df = df.drop_duplicates()

df_knn = df[['General_Health','Checkup','Exercise','Heart_Disease','Skin_Cancer','Other_Cancer','Depression','Diabetes','Arthritis','Sex','Age_Category','BMI','Smoking_History','Alcohol_Consumption','Fruit_Consumption','Green_Vegetables_Consumption','FriedPotato_Consumption']]

encoder = OneHotEncoder(sparse_output=False)

columns_to_encode = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'General_Health', 'Checkup', 'Exercise', 'Sex', 'Age_Category', 'Smoking_History']
df_knn = pd.get_dummies(df_knn, columns=columns_to_encode)

st.sidebar.header('Configurações do Gráfico')
x_column = st.sidebar.selectbox('Selecione a coluna para o eixo X', df_knn.columns)
y_column = st.sidebar.selectbox('Selecione a coluna para o eixo Y', df_knn.columns)

X = df_knn[[x_column, y_column]]
y = df_knn['Heart_Disease_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

color_scale = px.colors.sequential.Viridis

st.write('**Dataframe Codificado**')
st.dataframe(df_knn)
st.write('')
st.write('----')

st.write('**Classificação do Dataframe**')
fig = px.scatter(df_knn, x = x_column, y = y_column, color = df_knn['Heart_Disease_Yes'], color_continuous_scale = color_scale, hover_name = None)
st.plotly_chart(fig)
st.write('----')

st.write('**Precisão do experimento**')

accuracies = []

for size in range(1, 10):
    X_sample, _, y_sample, _ = train_test_split(X_test, y_test, test_size=(size/10), random_state=42)
    
    y_sample_pred = knn.predict(X_sample)
    
    accuracy = accuracy_score(y_sample, y_sample_pred)
    accuracies.append(accuracy)

fig = px.line(x=range(10, 100, 10), y = accuracies)
fig.update_xaxes(title = 'Tamanho da Amostra (%)')
fig.update_yaxes(title = 'Precisão do Modelo')

st.plotly_chart(fig)
