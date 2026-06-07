import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. Leitura do Arquivo
dados = pd.read_csv('diabetes.csv')
# 2. Separação dos atributos e classe
dados_atributos = dados.drop(columns=["Outcome"])
dados_classe = dados['Outcome']

# 3. Preencher valores nulos com a média
dados_atributos_limpos = dados_atributos.fillna(dados_atributos.mean())

# 4. Normalização
scaler = MinMaxScaler()
dados_num_normalizados = scaler.fit_transform(dados_atributos_limpos)
df_atributos_tratados = pd.DataFrame(dados_num_normalizados, columns=dados_atributos.columns)

# 5. Pipeline + cross validations + comparação dos modelos

print("\n Iniciando processo de comparação \n")

# Modelos do professor: Random Forest e Support Vector Machine | escolha: K-Nearest Neighbors (KNN)
modelos = {
    "Random Forests": RandomForestClassifier(random_state=42),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
}

notas_finais = {}

for nome_modelo, algoritmo in modelos.items():
    print(f"Avaliando: {nome_modelo}")
    
    #  O Pipeline (SMOTE aplicado apenas no treino de cada fold)
    pipeline_seguro = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classificador', algoritmo)
    ])
    
    # Cross Validation 
    resultados = cross_validate(
        pipeline_seguro, 
        df_atributos_tratados, 
        dados_classe, 
        scoring='accuracy', 
        cv=10, 
        n_jobs=-1
    )
    
    acuracia_media = resultados['test_score'].mean() * 100
    notas_finais[nome_modelo] = acuracia_media
    print(f"Acurácia: {acuracia_media:.2f}%\n")


print("Comparação modelos: \n")
notas_ordenadas = dict(sorted(notas_finais.items(), key=lambda item: item[1], reverse=True))

for modelo, nota in notas_ordenadas.items():
    print(f"{modelo}: {nota:.2f}%")

vencedor = list(notas_ordenadas.keys())[0]
print(f"\nMelhor modelo: {vencedor}.\n")

# 6. Treinamento do moledo
modelo_final = RandomForestClassifier(random_state=42)
balanceador_final = SMOTE(random_state=42)
X_final_balanceado, y_final_balanceado = balanceador_final.fit_resample(df_atributos_tratados, dados_classe)
modelo_final.fit(X_final_balanceado, y_final_balanceado)

# 7. Salva o modelo
pickle.dump(modelo_final, open('modelo_diabetes_rf.pkl', 'wb'))
print("Modelo salvo.\n")

# 8. Salva o scaler para a inferencia
pickle.dump(scaler, open('scaler_diabetes.pkl', 'wb'))
print("Scaler salvo.\n")

# 9. Salva a lista de colunas na ordem exata que o modelo aprendeu
pickle.dump(dados_atributos.columns, open('colunas_diabetes.pkl', 'wb'))

print("Tudo certo até aqui.\n")


