# Zeror

Projeto de experimentação com modelos de referência e modelos interpretáveis para classificação, com foco em baseline de maioria e regras induzidas.

## Objetivo

Este repositório explora dois cenários:

- um baseline ZeroR (Majority Learner), usado como referência para avaliar a qualidade de modelos mais complexos;
- um modelo interpretável baseado em regras, usando `imodels` com `RuleFitClassifier`, para extrair regras que expliquem o comportamento do classificador.

## Estrutura do projeto

```text
zeror/
├── data/
│   ├── credit_baseline.csv
│   └── play_tennis.csv
├── src/
│   ├── main.py
│   └── imodelstest.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Descrição dos arquivos

### `src/main.py`

Executa um baseline de classificação usando `DummyClassifier(strategy="most_frequent")` sobre o dataset `data/credit_baseline.csv`.

Ele:

- carrega os dados;
- separa variáveis explicativas e alvo;
- treina o modelo majoritário;
- calcula a acurácia no próprio conjunto de treino;
- imprime a classe majoritária e a métrica de baseline.

### `src/imodelstest.py`

Cria o arquivo `data/play_tennis.csv` caso ele não exista e, em seguida, treina um `RuleFitClassifier` do pacote `imodels`.

Ele:

- converte as variáveis categóricas em indicadores binários com `get_dummies`;
- transforma a classe alvo em valores binários;
- treina o modelo;
- exibe as regras extraídas e suas importâncias.

## Dados

### `data/credit_baseline.csv`

Conjunto de dados para o baseline ZeroR, com colunas como:

- `Renda`
- `Score_Credito`
- `Aprovado`

### `data/play_tennis.csv`

Conjunto de dados do clássico problema de jogar tênis, com variáveis como:

- `Aparencia`
- `Temperatura`
- `Umidade`
- `Vento`
- `Jogar`

## Requisitos

- Python 3.10+
- pip

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como executar

### 1. Baseline ZeroR

```bash
python src/main.py
```

Esse comando calcula a acurácia do modelo majoritário e mostra a classe predominante.

### 2. Modelo interpretável com regras

```bash
python src/imodelstest.py
```

Esse comando gera o dataset, treina o `RuleFitClassifier` e imprime as regras aprendidas com suas importâncias.

## Dependências principais

O projeto usa:

- `pandas`
- `numpy`
- `scikit-learn`
- `imodels`

## Observações

- O baseline ZeroR é útil para entender a referência mínima de desempenho.
- O modelo com `RuleFit` busca maior interpretabilidade, funcionando bem em problemas com variáveis categóricas e regras simples.
- O código em `src/imodelstest.py` cria o dataset automaticamente quando necessário, facilitando a execução em ambientes novos.

## Licença

Este projeto não especifica uma licença explícita no momento.
