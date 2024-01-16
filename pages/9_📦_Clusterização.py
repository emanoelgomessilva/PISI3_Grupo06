import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

df = pd.read_parquet('data/cardio_data_processed.parquet')

st.write('<h1>Clusterização (<i>clustering</i>)</h1>', unsafe_allow_html=True)
st.write('''Para a geração de grupos, foi usado o método K-means, que separa o <i>dataset</i> em <i>k</i> grupos distintos.
         Como a quantidade de <i>clusters</i> é subjetiva, para que pudesse ter uma base da quantidade adequada de
         <i>clusters</i>, foi aplicado o Método do Cotovelo, em que consiste em executar o algoritmo em <i>k</i> vezes, e
         calcular a inércia (soma das distâncias quadráticas dos pontos para o centro do <i>cluster</i> mais próximo), a partir do
         ponto que a incércia começa a diminuir de forma mais lenta, é chamado de "cotovelo", o número ideal de <i>clusters</i>.''', unsafe_allow_html=True)

def builder_body():
    elbow_method()
    #calculate_silhouette_score()
    Kmeans()

def elbow_method():
    df_test = df[['age_years', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi', 'bp_category_encoded']]

    encoder = OneHotEncoder(sparse_output=False)
    df_test_encoded = pd.DataFrame(encoder.fit_transform(df_test[['bp_category_encoded']]), columns=encoder.get_feature_names_out(['bp_category_encoded']))
    df_test_encoded.columns = df_test_encoded.columns.str.replace('bp_category_encoded_', '')
    df_test = pd.concat([df_test, df_test_encoded], axis=1)
    df_test.drop(['bp_category_encoded'], axis=1, inplace=True)

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
             
             Foi feito também o [Silhouette Score](https://drive.google.com/file/d/1sFk-x4ryyEtVfsUlDXayrhEHvf-mStSW/view?usp=sharing), 
             para avaliar a qualidade dos <i>clusters</i>. Utilizando esse método, é feito o cálculo do <i>score</i> para cada número de <i>cluster</i>,
             o <i>cluster</i> com maior <i>score</i> é o ideal, que nesse caso são <span style="color:red;">6</span> <i>clusters</i>, batendo
             com a mesma quantidade do método do cotovelo.''', 
             unsafe_allow_html=True)

    st.write('----')

def calculate_silhouette_score():
    df_silhouette = df[['age_years', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi', 'bp_category_encoded']]

    encoder = OneHotEncoder(sparse_output=False)
    df_silhouette_encoded = pd.DataFrame(encoder.fit_transform(df_silhouette[['bp_category_encoded']]), columns=encoder.get_feature_names_out(['bp_category_encoded']))
    df_silhouette_encoded.columns = df_silhouette_encoded.columns.str.replace('bp_category_encoded_', '')
    df_silhouette = pd.concat([df_silhouette, df_silhouette_encoded], axis=1)
    df_silhouette.drop(['bp_category_encoded'], axis=1, inplace=True)
    
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

    st.write(f"Com base no gráfico de silhueta, o número ideal de clusters é: <span style='color:red;'>{ideal_num_clusters}</span>", unsafe_allow_html=True)

def Kmeans():
    df_kmeans = df[['age_years', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bmi', 'bp_category_encoded']]
    
    encoder = OneHotEncoder(sparse_output=False)
    df_kmeans_encoded = pd.DataFrame(encoder.fit_transform(df_kmeans[['bp_category_encoded']]), columns=encoder.get_feature_names_out(['bp_category_encoded']))
    df_kmeans_encoded.columns = df_kmeans_encoded.columns.str.replace('bp_category_encoded_', '')
    df_kmeans = pd.concat([df_kmeans, df_kmeans_encoded], axis=1)
    df_kmeans.drop(['bp_category_encoded'], axis=1, inplace=True)

    normalization_option = st.selectbox('Escolha o tipo de normalização:', ['Sem Normalização', 'Normalização Padrão', 'Normalização MinMax'])

    if normalization_option == 'Sem Normalização':
        df_kmeans_standardized = df_kmeans.copy()
    elif normalization_option == 'Normalização Padrão':
        scaler = StandardScaler()
        df_kmeans_standardized = scaler.fit_transform(df_kmeans)
    elif normalization_option == 'Normalização MinMax':
        scaler = MinMaxScaler()
        df_kmeans_standardized = scaler.fit_transform(df_kmeans)

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_kmeans_standardized), columns=['PC1', 'PC2'])

    kmeans = KMeans(n_clusters=6, random_state=42)
    df_pca['cluster'] = kmeans.fit_predict(df_kmeans_standardized)
    df_pca['cluster'] = df_pca['cluster'].astype(str)

    df_kmeans_pca = pd.concat([df_kmeans, df_pca], axis=1)

    df_kmeans_pca = df_kmeans_pca.rename(columns={
        'age_years': 'Idade', 'Elevated': 'Elevado', 'Hypertension Stage 1': 'Hipertensão Nível 1',
        'Hypertension Stage 2': 'Hipertensão Nível 2', 
        'cholesterol': 'Colesterol', 'gluc': 'Glicose',
        'alco': 'Consumo de Álcool', 'active': 'Atividade Física', 
        'smoke': 'Tabagismo', 'bmi': 'IMC',
        'cardio': 'Cardio'})

    st.write('**Dataframe Codificado**')
    st.dataframe(df_kmeans_pca)
    st.write('')
    st.write('----')

    color_scale = ['#0000ff','#800080','#ffd700', '#ff0000', '#008000', '#8B4513']

    fig = px.scatter(
        df_kmeans_pca,
        x='PC1',
        y='PC2',
        color='cluster',
        color_discrete_sequence=color_scale,
        labels={'cluster': 'Cluster'},
        title='Clusters Gerados pelo K-Means',
    )

    st.plotly_chart(fig)


builder_body()