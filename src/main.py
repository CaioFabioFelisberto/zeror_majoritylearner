import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

df = pd.read_csv('data/credit_baseline.csv')

print(df.head())

X = df[['Renda', 'Score_Credito']]
y = df['Aprovado']

# 2. Treinar o modelo MajorityLearner (ZeroR)
majority_model = DummyClassifier(strategy="most_frequent")
majority_model.fit(X, y)

# 3. Predição e Avaliação
y_pred = majority_model.predict(X)
acc = accuracy_score(y, y_pred)
most_frequent_class = y.value_counts().idxmax()

# 4. Exibir resultados
print("=== RESULTADO DO MAJORITY LEARNER ===")
print(f"Classe Majoritária: {most_frequent_class}")
print(f"Acurácia Baseline: {acc:.2%}")
