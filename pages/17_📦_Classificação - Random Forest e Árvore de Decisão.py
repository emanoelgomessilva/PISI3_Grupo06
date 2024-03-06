import pandas as pd
import streamlit as st
from sklearn.svm import SVC
import plotly.figure_factory as ff
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.write('<h1>Classificação</h1>', unsafe_allow_html=True)
st.write('''Para essa classificação, é usado alguns algoritmos supervisionados: K-Nearest Neighbors (kNN), Suport Vector Machine (SVM),
         Random Forest e Árvore de Decisão (<i>Decision Tree</i>). As colunas usadas são colunas referentes a 
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

    selectable_features = ['Câncer de Pele', 'Outros Cânceres', 'Depressão', 'Diabetes', 'Artrite', 'Sem Câncer de Pele', 'Sem Qualquer Cânceres','Sem Depressão', 'Sem Artrite']

    selected_features = st.multiselect(
        "Características",
        selectable_features,
        default=['Depressão', 'Diabetes']
    )

    if len(selected_features) < 2:
        st.error("Selecione pelo menos duas características.")
    else:
        algorithm = st.selectbox("Escolha o Algoritmo", ["Random Forest", "Árvore de Decisão", "kNN", "SVM", "Regressão Logística"])

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

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        if algorithm == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif algorithm == "Árvore de Decisão":
            model = DecisionTreeClassifier(random_state=42)
        elif algorithm == "kNN":
            model = KNeighborsClassifier()
        elif algorithm == "SVM":
            model = SVC(kernel='linear', random_state=42)
        elif algorithm == "Regressão Logística":
            model = LogisticRegression(random_state=42)

        model.fit(X_train_resampled, y_train_resampled)

        y_train_pred = model.predict(X_train_resampled)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
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

        fig_conf_matrix = ff.create_annotated_heatmap(
            z=conf_matrix,
            x=['Previsto Negativo', 'Previsto Positivo'],
            y=['Real Negativo', 'Real Positivo'],
            colorscale='Blues',
            showscale=False
        )

        fig_conf_matrix.update_layout(
            title='Matriz de Confusão',
            xaxis_title='Previsto',
            yaxis_title='Real'
        )

        st.plotly_chart(fig_conf_matrix)

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
        elif algorithm == "Regressão Logística":
            coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
            coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

            st.write("**Coeficientes da Regressão Logística**")
            st.table(coef_df)
        elif algorithm == "kNN":
            st.write('''Não foi possível gerar a <i>feature importance</i> do algoritmo kNN.''', unsafe_allow_html=True)

        

classification()