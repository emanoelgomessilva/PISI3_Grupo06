import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.write('<h1>Classificação</h1>', unsafe_allow_html=True)
st.write('''Para essa classificação, é usado dois algoritmos supervisionados: Random Forest e Árvore de Decisão
         (<i>Decision Tree</i>). As colunas usadas são colunas referentes a existência ou ausência de outras enfermidades,
         como câncer de pele, depressão, artrite, diabetes e de mais cânceres, a fim de acertar as ocorrências da coluna
         que indica presença de doença cardiovascular - chamado também de coluna <i>target</i>. Após essa predição, é 
         mostrado as porcentagens de acerto das amostras de treino e de teste.''', unsafe_allow_html=True)

def classification():
    df = pd.read_parquet('data/cvd_cleaned.parquet')
    df = df.drop_duplicates()

    df['Heart_Disease'] = LabelEncoder().fit_transform(df['Heart_Disease'])
    df['Skin_Cancer'] = LabelEncoder().fit_transform(df['Skin_Cancer'])
    df['Other_Cancer'] = LabelEncoder().fit_transform(df['Other_Cancer'])
    df['Depression'] = LabelEncoder().fit_transform(df['Depression'])
    df['Diabetes'] = LabelEncoder().fit_transform(df['Diabetes'])
    df['Arthritis'] = LabelEncoder().fit_transform(df['Arthritis'])

    df = df.rename(columns={
        'Skin_Cancer': 'Câncer de Pele',
        'Other_Cancer': 'Outros Cânceres',
        'Depression': 'Depressão',
        'Arthritis': 'Artrite'})

    selectable_features = ['Câncer de Pele', 'Outros Cânceres', 'Depressão', 'Diabetes', 'Artrite']

    selected_features = st.multiselect(
        "Características",
        selectable_features,
        default=['Depressão', 'Diabetes']
    )

    algorithm = st.selectbox("Escolha o Algoritmo", ["Random Forest", "Árvore de Decisão"])

    features_and_target = selected_features + ['Heart_Disease']
    df_selected = df[features_and_target]

    X = df_selected.drop('Heart_Disease', axis=1)
    Y = df_selected['Heart_Disease']

    X_scaled = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        Y,
        test_size=0.4,
        random_state=42
    )

    if algorithm == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    st.write(f"A porcentagem de acerto para o treino foi: {accuracy_train:.2%}")
    st.write(f"A porcentagem de acerto para o teste foi: {accuracy_test:.2%}")

classification()