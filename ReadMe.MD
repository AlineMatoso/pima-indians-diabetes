Link para a base: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Treine e avalie a acurácia de 3 modelos
Random Forests
Support Vector Machine
Um classificado a sua escolha
Compare a acurácia detalhadamente e indique qual modelo é mais adequado para entrar em produção

### Colunas

Dados atributos[8] = Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
Dados Classe[1] = outcome (essa coluna é resultado)

Todos são numericos

1. Pregnancies - numero de vezes gravida
2. Glucose - Concentração plasmática de glicose a 2 horas em um teste oral de tolerância à glicose
3. BloodPressure - Pressão arterial diastólica (mm Hg)
4. SkinThickness - Espessura da dobra da pele do tríceps (mm)
5. Insulin - Insulina sérica de 2 horas (mu U/ml)
6. BMI - Índice de massa corporal (peso em kg/(altura em m)^2)
7. DiabetesPedigreeFunction - Função do pedigree de diabetes
8. Age - Idade (anos)
9. Outcome - Variável de classe (0 ou 1) 268 de 768 são 1, as outras são 0


### Ordem do projeto

1. Separação da base (classe e atributo)
2. One-Hot Enconding ou normalização
3. Pipeline
4. Cross validations
5. Treinamento
6. Salvar o modelo
7. Inferência



----------------------------------------------------------------------------------------------
### 🧠 Entendendo o Código: Comparação Automatizada de Modelos

Para realizar o *benchmark* (teste comparativo) entre os três algoritmos, o código foi estruturado como um "campeonato automatizado" dividido em 4 blocos principais:

#### 1. A Inscrição dos Competidores (Os Dicionários)

modelos = {
    "Random Forests": RandomForestClassifier(random_state=42),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
}
notas_finais = {}



* **O Dicionário (`modelos`):** Funciona como a lista de inscrição do torneio. A "chave" (texto) é o nome do modelo, e o "valor" é a implementação da biblioteca `scikit-learn` (o "robô") instanciada e pronta para competir. O `random_state=42` garante que eles ajam da mesma forma em testes repetidos.
* **O Dicionário vazio (`notas_finais`):** É o nosso caderno de anotações onde guardaremos a pontuação final de cada modelo após as provas.

#### 2. A Arena de Batalha (O Loop e o Pipeline)

for nome_modelo, algoritmo in modelos.items():
    # O Pipeline (SMOTE aplicado apenas no treino de cada fold)
    pipeline_seguro = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classificador', algoritmo)
    ])



* **O Loop `for`:** A função `.items()` pega um modelo por vez no dicionário. O código testa o primeiro, depois o segundo, e por fim o terceiro, de forma automatizada sem precisarmos repetir o código.
* **O Pipeline (A regra do jogo limpo):** Em vez de passar o algoritmo puro para ser testado, criamos um "combo" de duas etapas: balanceamento (`smote`) + previsão (`classificador`). Isso garante que o SMOTE só seja aplicado nos dados de estudo (treino), prevenindo o *Data Leakage* (vazamento de dados para as fatias de teste).

#### 3. O Exame Rigoroso (Cross Validation)


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



* **A Prova (`cross_validate`):** O algoritmo divide o dataset em 10 fatias (`cv=10`) e aplica 10 provas independentes para cada modelo.
* **Performance (`n_jobs=-1`):** Um comando que avisa ao processador do computador para usar todos os seus núcleos simultaneamente, processando os testes o mais rápido possível.
* **O Boletim:** O resultado retorna 10 notas. O código calcula a média aritmética (`mean()`), multiplica por 100 para transformar em porcentagem, e salva no nosso caderno de `notas_finais`.

#### 4. O Pódio (Ordenação e Vencedor)


notas_ordenadas = dict(sorted(notas_finais.items(), key=lambda item: item[1], reverse=True))
vencedor = list(notas_ordenadas.keys())[0]



* **A Ordenação (`sorted`):** Pega as notas registradas e as organiza. A mágica acontece na função invisível `lambda item: item[1]`, que manda o Python ignorar o nome do modelo (o `item[0]`) e ordenar exclusivamente com base no valor numérico da nota (o `item[1]`).
* **Ordem Decrescente:** O `reverse=True` garante que a maior nota fique no topo da lista.
* **A Coroa:** A última linha transforma o dicionário ordenado numa lista simples e captura o item de índice `[0]` (o primeiro da fila), coroando o grande vencedor do campeonato.

