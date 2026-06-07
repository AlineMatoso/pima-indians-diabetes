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

