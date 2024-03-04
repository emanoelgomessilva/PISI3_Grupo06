import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.write('<h1>Classificação - Gradient Boosting</h1>', unsafe_allow_html=True)
st.write('''Para essa classificação foi usado o algoritmo Gradient Boosting. As colunas usadas são colunas referentes a 
         existência ou ausência de outras enfermidades, como câncer de pele, depressão, artrite, diabetes e 
         demais cânceres, a fim de acertar as ocorrências da coluna que indica presença de doença cardiovascular - 
         chamado também de coluna <i>target</i>. Após essa predição, é mostrado as porcentagens de acerto das 
         amostras de treino e de teste.''', unsafe_allow_html=True)

def classification():
    df = pd.read_parquet('data/cvd_cleaned.parquet')
    df = df.drop_duplicates()

    columns_to_encode = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis']
    df = pd.get_dummies(df, columns=columns_to_encode)

    df = df.rename(columns={
        'Skin_Cancer_Yes': 'Câncer de Pele',
        'Other_Cancer_Yes': 'Outros Cânceres',
        'Depression_Yes': 'Depressão',
        'Arthritis_Yes': 'Artrite',
        'Skin_Cancer_No': 'Sem Câncer de Pele',
        'Other_Cancer_No': 'Sem Qualquer Cânceres',
        'Depression_No': 'Sem Depressão',
        'Arthritis_No': 'Sem Artrite',
        'Diabetes_Yes': 'Diabetes'})
    
    st.write('**Dataframe Codificado**')
    st.dataframe(df)
    st.write('')
    st.write('----')

    selectable_features = ['Câncer de Pele', 'Outros Cânceres', 'Depressão', 'Diabetes', 'Artrite']

    selected_features = st.multiselect(
        "Características",
        selectable_features,
        default=['Depressão', 'Diabetes']
    )

    if len(selected_features) < 2:
        st.error("Selecione pelo menos duas características.")

        features_and_target = selected_features + ['Heart_Disease_Yes']
        df_selected = df[features_and_target]

        X = df_selected.drop('Heart_Disease_Yes', axis=1)
        Y = df_selected['Heart_Disease_Yes']

        X_scaled = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            Y,
            test_size=0.2,
            random_state=42
        )

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        st.write(f"A porcentagem de acerto para o treino foi: <span style='color:red;'>{accuracy_train:.2%}</span>", unsafe_allow_html = True)
        st.write(f"A porcentagem de acerto para o teste foi: <span style='color:red;'>{accuracy_test:.2%}</span>", unsafe_allow_html = True)

        st.write('----')

        st.write("**Métricas de Classificação:**")
        st.table(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T)

        st.write('----')

        conf_matrix = confusion_matrix(y_test, y_test_pred)

        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=['Real Negativo', 'Real Positivo'],
            columns=['Previsto Negativo', 'Previsto Positivo']
        )

        st.write("**Matriz de Confusão**")
        st.write('''É uma tabela que resume o desempenho de um modelo de classificação, destacando 
                 Verdadeiros Positivos (VP), Falsos Positivos (FP), Falsos Negativos (FN) e Verdadeiros Negativos (VN). 
                 Essa tabela fornece uma visão detalhada dos acertos e erros do modelo, sendo importante 
                 para avaliar sua eficácia e identificar áreas de melhoria.''')
        st.table(conf_matrix_df)

        st.write('----')

classification()