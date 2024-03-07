import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

selected_features = ['Depressão', 'Diabetes', 'Câncer de Pele', 'Outros Cânceres']

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

def train_and_evaluate_model(model):
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

models = {
    'Gradient Boosting' :GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    #'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42)   
}

results_list = []
confusion_matrix_list = []

for model_name, model in models.items():
    st.write(f"Métricas do algoritmo: {model_name}", unsafe_allow_html=True)
    train_and_evaluate_model(model)