import os
import pandas as pd
from imodels import RuleFitClassifier

def main():
    # 1. Garantir que a pasta de dados existe e carregar o CSV
    csv_path = 'data/play_tennis.csv'
    
    if not os.path.exists(csv_path):
        os.makedirs('data', exist_ok=True)
        data_dict = {
            'Aparencia': ['Ensolarado', 'Ensolarado', 'Nublado', 'Chuvoso', 'Chuvoso', 
                          'Chuvoso', 'Nublado', 'Ensolarado', 'Ensolarado', 'Chuvoso', 
                          'Ensolarado', 'Nublado', 'Nublado', 'Chuvoso'],
            'Temperatura': ['Quente', 'Quente', 'Quente', 'Ameno', 'Frio', 
                            'Frio', 'Frio', 'Ameno', 'Frio', 'Ameno', 
                            'Ameno', 'Ameno', 'Quente', 'Ameno'],
            'Umidade': ['Alta', 'Alta', 'Alta', 'Alta', 'Normal', 
                        'Normal', 'Normal', 'Alta', 'Normal', 'Normal', 
                        'Normal', 'Alta', 'Normal', 'Alta'],
            'Vento': ['Fraco', 'Forte', 'Fraco', 'Fraco', 'Fraco', 
                      'Forte', 'Forte', 'Fraco', 'Fraco', 'Fraco', 
                      'Forte', 'Forte', 'Fraco', 'Forte'],
            'Jogar': ['Nao', 'Nao', 'Sim', 'Sim', 'Sim', 
                      'Nao', 'Sim', 'Nao', 'Sim', 'Sim', 
                      'Sim', 'Sim', 'Sim', 'Nao']
        }
        pd.DataFrame(data_dict).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    # 2. Prepara os dados (Convertendo booleans explicitamente para inteiros)
    X = pd.get_dummies(df.drop(columns=['Jogar']), drop_first=False).astype(int)
    y = df['Jogar'].map({'Sim': 1, 'Nao': 0}) # Binarizando a classe alvo

    # 3. Treinar o modelo de Indução de Regras (RuleFit)
    model = RuleFitClassifier(max_rules=10, random_state=42)
    model.fit(X, y)

    # 4. Exibir as regras extraídas
    print("=== REGRAS INDUZIDAS COM IMODELS (RuleFit) ===")
    rules_df = model.get_rules()
    
    # Filtrando apenas as regras relevantes (coeficiente != 0)
    active_rules = rules_df[rules_df['importance'] > 0].sort_values(by='importance', ascending=False)
    
    for _, row in active_rules.iterrows():
        print(f"Regra: {row['rule']} | Importância: {row['importance']:.4f}")

if __name__ == "__main__":
    main()