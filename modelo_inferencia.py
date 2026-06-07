import pandas as pd
import pickle

# Abrir o modelo salvo no treinamento

modelo = pickle.load(open('modelo_diabetes_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler_diabetes.pkl', 'rb'))
colunas_treino = pickle.load(open('colunas_diabetes.pkl', 'rb'))

# salvar nova inferencia
nova_paciente = [0,137,40,35,168,43.1,2.288,33]
# paciente: 0,137,40,35,168,43.1,2.288,33,1

# Transformamos a lista em um DataFrame do Pandas (garantindo que as colunas têm os nomes certos)
df_nova_paciente = pd.DataFrame([nova_paciente], columns=colunas_treino)

# Tratamento dos dados da nova paciente
df_normalizado = pd.DataFrame(scaler.transform(df_nova_paciente), columns=colunas_treino)

# Inferencia 
resultado = modelo.predict(df_normalizado)


# Resultado
# Classe 1 = Resultado Positivo (A paciente é diabética)
# Classe 0 = Resultado Negativo (A paciente não é diabética)
print("Resultado: ")
if resultado[0] == 1:
    print("A paciente se enquadra na: Classe 1 (Diabética)")
else:
    print("A paciente se enquadra na: Classe 0 (Não Diabética)")