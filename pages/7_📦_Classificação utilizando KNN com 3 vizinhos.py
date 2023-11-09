import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

df = pd.read_parquet('data/cvd_cleaned.parquet')

df_knn = df[['General_Health','Checkup','Exercise','Heart_Disease','Skin_Cancer','Other_Cancer','Depression','Diabetes','Arthritis','Sex','Age_Category','Height_(cm)','Weight_(kg)','BMI','Smoking_History','Alcohol_Consumption','Fruit_Consumption','Green_Vegetables_Consumption','FriedPotato_Consumption']]

mapeamento_general_health = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4
}

df_knn['General_Health'] = df['General_Health'].replace(mapeamento_general_health)

mapeamento_checkup = {
    'Within the past year': 0,
    'Within the past 2 years': 1,
    'Within the past 5 years': 2,
    '5 or more years ago': 3,
    'Never': 4   
}

df_knn['Checkup'] = df['Checkup'].replace(mapeamento_checkup)

mapeamento_exercise = {
    'Yes': 1,
    'No': 0  
}

df_knn['Exercise'] = df['Exercise'].replace(mapeamento_exercise)

mapeamento_skin_cancer = {
    'Yes': 1,
    'No': 0  
}

df_knn['Skin_Cancer'] = df['Skin_Cancer'].replace(mapeamento_skin_cancer)

mapeamento_other_cancer = {
    'Yes': 1,
    'No': 0  
}

df_knn['Other_Cancer'] = df['Other_Cancer'].replace(mapeamento_other_cancer)

mapeamento_depression = {
    'Yes': 1,
    'No': 0  
}

df_knn['Depression'] = df['Depression'].replace(mapeamento_depression)

mapeamento_diabetes = {
    'Yes': 1,
    'No': 2,
    'No, pre-diabetes or borderline diabetes': 3,
    'Yes, but female told only during pregnancy': 4  
}

df_knn['Diabetes'] = df['Diabetes'].replace(mapeamento_diabetes)

mapeamento_arthritis = {
    'Yes': 1,
    'No': 0  
}

df_knn['Arthritis'] = df['Arthritis'].replace(mapeamento_arthritis)

mapeamento_sex = {
    'Male': 1,
    'Female': 0  
}

df_knn['Sex'] = df['Sex'].replace(mapeamento_sex)

mapeamento_age_category = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80+': 12,  
}

df_knn['Age_Category'] = df['Age_Category'].replace(mapeamento_age_category)

mapeamento_smoking_history = {
    'Yes': 1,
    'No': 0  
}

df_knn['Smoking_History'] = df['Smoking_History'].replace(mapeamento_smoking_history)

df_knn['Heart_Disease'] = LabelEncoder().fit_transform(df_knn['Heart_Disease'])

st.sidebar.header('Configurações do Gráfico')
x_column = st.sidebar.selectbox('Selecione a coluna para o eixo X', df_knn.columns)
y_column = st.sidebar.selectbox('Selecione a coluna para o eixo Y', df_knn.columns)

X = df_knn[[x_column, y_column]]
y = df_knn['Heart_Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

prediction = knn.predict(X)

color_scale = px.colors.sequential.Viridis

st.write('**Dataframe Codificado**')
st.dataframe(df_knn)
st.write('')
st.write('----')

st.write('**Classificação do Dataframe**')
fig = px.scatter(df_knn, x = x_column, y = y_column, color = df_knn['Heart_Disease'], color_continuous_scale = color_scale, hover_name = None)
st.plotly_chart(fig)
st.write('----')

plt.scatter(x_column, y_column, c=prediction[0])
plt.text(x=x_column-1.7, y=y_column-0.7, s=f"new point, class: {prediction[0]}")
plt.show()