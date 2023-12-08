import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.write('<h1>Classificação</h1>', unsafe_allow_html=True)
st.write('''Para essa classificação, é usado alguns algoritmos supervisionados: K-Nearest Neighbors (kNN), Suport Vector Machine (SVM),
         Random Forest e Árvore de Decisão (<i>Decision Tree</i>). As colunas usadas são colunas referentes a 
         existência ou ausência de outras enfermidades, como câncer de pele, depressão, artrite, diabetes e de 
         mais cânceres, a fim de acertar as ocorrências da coluna que indica presença de doença cardiovascular - 
         chamado também de coluna <i>target</i>. Após essa predição, é mostrado as porcentagens de acerto das 
         amostras de treino e de teste.''', unsafe_allow_html=True)

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

    if len(selected_features) < 2:
        st.error("Selecione pelo menos duas características.")
    else:
        algorithm = st.selectbox("Escolha o Algoritmo", ["Random Forest", "Árvore de Decisão", "kNN", "SVM"])

        features_and_target = selected_features + ['Heart_Disease']
        df_selected = df[features_and_target]

        X = df_selected.drop('Heart_Disease', axis=1)
        Y = df_selected['Heart_Disease']

        X_scaled = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            Y,
            test_size=0.2,
            random_state=42
        )

        if algorithm == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif algorithm == "Árvore de Decisão":
            model = DecisionTreeClassifier(random_state=42)
        elif algorithm == "kNN":
            model = KNeighborsClassifier()
        elif algorithm == "SVM":
            model = SVC(kernel='linear', random_state=42)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        st.write(f"A porcentagem de acerto para o treino foi: <span style='color:red;'>{accuracy_train:.2%}</span>", unsafe_allow_html = True)
        st.write(f"A porcentagem de acerto para o teste foi: <span style='color:red;'>{accuracy_test:.2%}</span>", unsafe_allow_html = True)

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

        if algorithm == "Random Forest" or algorithm == "Árvore de Decisão":
            model.fit(X_train, y_train)
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Característica': X.columns,
                'Importância': feature_importances
            })

            feature_importance_df = feature_importance_df.sort_values(by='Importância', ascending=False)

            st.write("**<i>Feature importance</i>**", unsafe_allow_html=True)
            st.write('''Essa métrica quantifica a influência de cada característica na capacidade 
                        preditiva do modelo, permitindo identificar quais variáveis têm maior impacto na 
                        predição do resultado. <i>Feature importance</i> é crucial para compreender o papel de 
                        cada variável no desempenho do modelo e pode auxiliar na seleção e otimização das 
                        características para melhorar a eficácia da predição.''', unsafe_allow_html=True)
            st.table(feature_importance_df)
        elif algorithm == "SVM":
            model.fit(X_train, y_train)
            coefficients = model.coef_[0]
            feature_importance = pd.DataFrame({
                'Característica': X.columns,
                'Importância': coefficients
            })

            feature_importance = feature_importance.sort_values(by='Importância', ascending=False)

            st.write("**<i>Feature importance</i>**", unsafe_allow_html=True)
            st.write('''Essa métrica quantifica a influência de cada característica na capacidade 
                        preditiva do modelo, permitindo identificar quais variáveis têm maior impacto na 
                        predição do resultado. <i>Feature importance</i> é crucial para compreender o papel de 
                        cada variável no desempenho do modelo e pode auxiliar na seleção e otimização das 
                        características para melhorar a eficácia da predição.''', unsafe_allow_html=True)
            st.table(feature_importance)
        elif algorithm == "kNN":
            st.write('''Não foi possível gerar a <i>feature importance</i> do algoritmo kNN''', unsafe_allow_html=True)

        

classification()