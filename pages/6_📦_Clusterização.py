import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_parquet('data/cardio_data_processed.parquet')

st.write('<h1>Clusterização (<i>clustering</i>)</h1>', unsafe_allow_html=True)
st.write('''Para a geração de grupos, foi usado o método K-means, que separa o <i>dataset</i> em <i>k</i> grupos distintos.
         Como a quantidade de <i>clusters</i> é subjetiva, para que pudesse ter uma base da quantidade adequada de
         <i>clusters</i>, foi aplicado o Método do Cotovelo, em que consiste em executar o algoritmo em <i>k</i> vezes, e
         calcular a inércia (soma das distâncias quadráticas dos pontos para o centro do cluster mais próximo), a partir do
         ponto que a incércia começa a diminuir de forma mais lenta, é chamado de "cotovelo", o número ideal de <i>clusters</i>.''', unsafe_allow_html=True)

def builder_body():
    elbow_method()
    #silhouette_score()
    Kmeans()

def elbow_method():
    df_test = df[['age_years', 'bp_category', 'cholesterol', 'gluc', 'alco', 'active', 'smoke', 'bmi']]

    df_test['bp_category'] = LabelEncoder().fit_transform(df_test['bp_category'])

    X = StandardScaler().fit_transform(df_test)

    inertia_values = []

    progress_bar = st.progress(0)

    for k in range(2, 11):
        kmeans = KMeans(n_clusters = k, random_state = 4294967295)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
        progress_bar.progress((k / 10))

    kneedle = KneeLocator(range(2, 11), inertia_values, curve = 'convex', direction = 'decreasing')
    ideal_k = kneedle.elbow

    st.write('**Método do Cotovelo**')
    fig_inertia = px.line(x=range(2, 11), y = inertia_values)
    fig_inertia.update_xaxes(title = 'Número de Clusters (K)')
    fig_inertia.update_yaxes(title = 'Inércia')

    st.plotly_chart(fig_inertia)
    st.write(f'''
             Com base no Método do Cotovelo, o número ideal de <i>clusters</i> é: <span style="color:red;">{ideal_k}</span>.
             
             Foi feito também o [Silhouette Score](https://lh3.googleusercontent.com/pw/ADCreHfwH1UJzJwSO__g2QbYlk-1gbSxSN4ckMJXnqWPnFiFyOXsLEpultG2E8Qi9JSL65m-jk-V3wLwuVi91lWdtEZnAj8rzR8-l_tALtQc8QPxt5l6wonJP52up267EKOHDQtpYSLINQlRSgZWiL081a9lHIJgUpmBDEbVuhqW8qnBlYQN6yBtyRYU8kt9Y_QJTfDQKGYvimMNaRWwaT80uYDhtee8IoH_4gV5xWRGJ5KYZJ39JQx6qNKWI2KHWG0sLr02E2XTK-IDVhm7SJ_9bD7uWqS0gxx-RDT-fCVAyEAP4KaHyDSaICvYtWJjKRRJkiyUXxrafKSgEI5JBPZUk3ADUM_TCXr80rHRjVwgcTdaXlpQKLcvFdBY_JsPRH-4I4ZaACTier6qurSvQ-L5-zpd3X5v8FbrHhJv9VPsaNuVvPKt9D1E3KfJleLG38h5EaXTqTj74HLLwBS_PvQMXu1BnkoGSg7oQZzXFkiwrzZMxbo8gviKBlnzya46JCJl30RQCjaCgTb_AMXZHJIwNVuhysC8gNmnKyES_7yTL82agtEAJ6etoXmBilUIR0Usq_uwIVHm4xjzRwPAjzl6WU-NloQaQNB9o-uNWE7KvCPhrK9yVw6KKUY0nyFaADO_SV40swIpfchMBtu42LSS7APR_BfYF-esYr9qNBJQ33y5Uk37Nzbhgskod1FOmiCF-v71feQ_kgAYs05LAG32k-8Ktc4kl40sleHYfRkGvOcNhhL04vq4JZ2yRS32obKONR3ZcisoBhwRC9erJUHjIzvHnqLvIPpShPdLU-SJxyXdOq15lIir5SOSnPtp91sqpGL9yoHIfrLwnxC2RKAjUaLgtoaHdalNxsoXGHa-VWVrojIb1Mx1WP5hD-7BwzOxYgn3l2CzxOiajF5v5fPytAwd6A=w1315-h559-s-no?authuser=0), 
             para avaliar a qualidade dos <i>clusters</i>. Utilizando esse método, é feito o cálculo do <i>score</i> para cada número de <i>cluster</i>,
             o <i>cluster</i> com maior <i>score</i> é o ideal, que nesse caso são <span style="color:red;">3</span> <i>clusters</i>.''', 
             unsafe_allow_html=True)

    st.write('----')

def silhouette_score():
    df_silhouette = df[['age_years', 'bp_category', 'cholesterol', 'gluc', 'alco', 'active', 'smoke', 'bmi']]

    df_silhouette['bp_category'] = LabelEncoder().fit_transform(df_silhouette['bp_category'])
    X = StandardScaler().fit_transform(df_silhouette)

    silhouette_scores = []

    progress_bar = st.progress(0)

    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters = n_clusters, random_state = 4294967295)
        cluster_labels = kmeans.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        progress = (n_clusters - 2 + 1) / (10 - 2 + 1)
        progress_bar.progress(progress)

    ideal_num_clusters = np.argmax(silhouette_scores) + 2

    fig = px.line(x=range(2, 11), y = silhouette_scores)
    fig.update_xaxes(title = 'Número de Clusters (K)')
    fig.update_yaxes(title = 'Pontuação da Silhueta')

    st.plotly_chart(fig)

    st.write(f"Com base no gráfico de silhueta, o número ideal de clusters é: {ideal_num_clusters}")

def Kmeans():
    df_kmeans = df[['id', 'age_years','cholesterol', 'gluc','smoke', 'alco', 'active', 'cardio', 'bmi', 'bp_category_encoded']]

    df_kmeans['bp_category_encoded'] = LabelEncoder().fit_transform(df_kmeans['bp_category_encoded'])

    df_kmeans = df_kmeans.rename(columns={
        'age_years': 'Idade', 'bp_category_encoded': 'Pressão Sanguínea', 
        'cholesterol': 'Colesterol', 'gluc': 'Glicose',
        'alco': 'Consumo de Álcool', 'active': 'Atividade Física', 
        'smoke': 'Tabagismo', 'bmi': 'IMC',
        'cardio': 'Cardio'})

    st.sidebar.header('Configurações do Gráfico')
    num_clusters = st.sidebar.slider('Número de Clusters', min_value = 2, max_value = 6, value = 3)
    x_column = st.sidebar.selectbox('Selecione a coluna para o eixo X', df_kmeans.columns)
    y_column = st.sidebar.selectbox('Selecione a coluna para o eixo Y', df_kmeans.columns)
    
    X = df_kmeans[[x_column, y_column]]
    X = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters = num_clusters, random_state = 4294967295)
    df_kmeans['cluster'] = kmeans.fit_predict(X)

    color_scale = px.colors.sequential.Viridis

    st.write('**Dataframe Codificado**')
    st.dataframe(df_kmeans)
    st.write('')
    st.write('----')

    st.write('**Clusterização do Dataframe**')
    fig = px.scatter(df_kmeans, x = x_column, y = y_column, color = df_kmeans['cluster'], color_continuous_scale = color_scale, hover_name = 'id')
    st.plotly_chart(fig)
    st.write('----')

builder_body()