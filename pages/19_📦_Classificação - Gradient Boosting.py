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

selected_features = ['Depressão', 'Diabetes', 'Câncer de Pele', 'Outros Cânceres', 'Artrite', 'Sem Câncer de Pele', 'Sem Qualquer Cânceres','Sem Depressão', 'Sem Artrite']

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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

models = {
    'Gradient Boosting' :GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42)   
}

results_list = []
confusion_matrix_list = []

for model_name, model in models.items():
    accuracy, precision, recall, f1, conf_matrix = train_and_evaluate_model(model)
    results_list.append({
        'Modelo': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

for model_name, model in models.items():
    accuracy, precision, recall, f1, conf_matrix = train_and_evaluate_model(model)
    confusion_matrix_list.append({
        'Modelo': model_name,
        'Confusion Matrix': conf_matrix
    })

results = pd.DataFrame(results_list)
confusion_matrix_df = pd.DataFrame(confusion_matrix_list)

st.write("## Resultados de treinamento do Gradient Boosting e dos algoritmos utilizados anteriormente")
st.write(results)

for model_name, model in models.items():
    accuracy, precision, recall, f1, conf_matrix = train_and_evaluate_model(model)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=['Real Negativo', 'Real Positivo'],
        columns=['Previsto Negativo', 'Previsto Positivo']
    )

    st.write(f"**Matriz de Confusão: {model_name}**")
    st.table(conf_matrix_df)

    st.write('----')