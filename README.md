# LogRegressionCredAnalysis

---

Aplicação em Python (Streamlit) para análise e decisão de crédito. O app treina um modelo de Regressão Logística para estimar a probabilidade de um cliente ser **bom pagador** (`Prob_BOM`), converte essa probabilidade em um **Score (0–1000)** e classifica o cliente em um **Rating (A–E)**. Também oferece backtesting, métricas e visualizações para apoiar um comitê de crédito.

Esta aplicação é um exemplo do portifólio de **Marcio Cordeiro** aplicado a Machine Learning, voltado ao mercado financeiro.

## O que o app entra

- Treino automático de um classificador de Regressão Logística (scikit-learn).
- Pré-processamento: one-hot encoding de variáveis categóricas e padronização (`StandardScaler`).
- Scoring: `Prob_BOM` → `Score` → `Rating`.
- Regras de decisão parametrizáveis (cutoff e “zona de revisão”).
- Métricas do modelo e backtesting: AUC/ROC, Gini, Brier, matriz de confusão, relatório de classificação.
- Exploração do portfólio via filtros (rating, score, busca de cliente) e exportação da visão atual.

## Estrutura do projeto

- `LogRegressionCredAnalysis.py`: aplicação Streamlit (pipeline + interface).
- `BrazilianCredit.csv`: base de dados usada pelo app.
- `requirements.txt`: dependências (observação: o app também usa Streamlit/Matplotlib/Seaborn).

## Como executar

### 1) Criar e ativar ambiente virtual (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Instalar dependências

```powershell
pip install -r requirements.txt
```

### 3) Rodar o Streamlit

```powershell
streamlit run .\LogRegressionCredAnalysis.py
```

## Dataset (entrada)

O app carrega o arquivo `BrazilianCredit.csv` e espera, no mínimo:

- `nome_cliente`: identificador do cliente (o app usa como índice).
- `risco_credito`: alvo binário do modelo (esperado: `1` = **BOM**, `0` = **MAU**).

As demais colunas são tratadas como variáveis explicativas (features). O script faz `get_dummies` automaticamente nas variáveis categóricas e padroniza as numéricas.

Observação: no painel, algumas colunas são exibidas quando existem (por exemplo: `status_conta`, `historico_credito`, `poupanca`, `valor`, `idade`, `duracao_meses`, `tempo_emprego`). Se a sua base tiver outras colunas, elas também entram no modelo.

## Pipeline (o que acontece “por trás”)

O fluxo principal pode ser lido diretamente no código em `train_pipeline()` e `score_portfolio()`:

1. Separa features e alvo:
   - `X = df.drop(columns=["risco_credito"])`
   - `y = df["risco_credito"]`
2. One-hot encoding nas categóricas:
   - `X_encoded = pd.get_dummies(X, drop_first=True)`
3. Split treino/teste:
   - `train_test_split(..., test_size=0.3, random_state=42)`
4. Padronização (z-score) com `StandardScaler`:
   - `X_scaled = (X - μ) / σ` (calculado por coluna no treino)
5. Treino do modelo:
   - `LogisticRegression(max_iter=1000, class_weight="balanced")`
6. Scoring:
   - `Prob_BOM = predict_proba(...)[:, 1]`
   - `Score = round(Prob_BOM * 1000)`
   - `Rating = score_to_rating(Score)`

## Cálculos envolvidos

### 1) Regressão Logística (probabilidade)

Para cada cliente com vetor de features **x**, a regressão logística estima:

- Função linear:`z = wᵀx + b`
- Função sigmoide (transforma em probabilidade):
  `p = σ(z) = 1 / (1 + e^(−z))`

No app, essa probabilidade é acessada via `predict_proba` do scikit-learn:

- `Prob_BOM = P(y=1 | x)`

### 2) Função objetivo (treinamento)

De forma conceitual, o treino busca **w** e **b** que minimizam a perda logística (cross-entropy) em classificação binária:

`L = − (1/n) Σ [ y·log(p) + (1−y)·log(1−p) ]`

Além disso, o scikit-learn aplica regularização por padrão (penalização para evitar overfitting).

### 3) Padronização (StandardScaler)

Cada feature numérica é transformada com z-score (ajustada no treino e aplicada no teste/novos dados):

`x_pad = (x − μ_treino) / σ_treino`

Isso ajuda a regressão logística a convergir melhor e evita que variáveis em escalas muito diferentes dominem a solução.

### 4) Score (0–1000) a partir da probabilidade

O app usa uma escala simples e interpretável:

`Score = round(Prob_BOM × 1000)`

Exemplos:

- `Prob_BOM = 0.83` → `Score ≈ 830`
- `Prob_BOM = 0.12` → `Score ≈ 120`

### 5) Rating (faixas por regra de negócio)

O rating é derivado por cortes fixos de score:

- Score ≥ 850 → Rating A (Risco Mínimo)
- Score ≥ 700 → Rating B (Risco Baixo)
- Score ≥ 500 → Rating C (Risco Médio)
- Score ≥ 300 → Rating D (Risco Alto)
- Score < 300 → Rating E (Risco Crítico)

Esses limites são regras de negócio e podem ser ajustados no código.

### 6) Regra de decisão (cutoff e zona de revisão)

O painel permite configurar um cutoff em score (`cutoff_score`) e converte para probabilidade:

`cutoff_proba = cutoff_score / 1000`

Classificação prevista usada em métricas e divergências:

- Se `Prob_BOM ≥ cutoff_proba` → **Previsto = BOM**
- Caso contrário → **Previsto = MAU**

Para decisão operacional do comitê, é possível habilitar uma “zona de revisão” (faixa intermediária):

- `low = cutoff_score − largura/2`
- `high = cutoff_score + largura/2`

Então:

- Score ≥ high → **Aprovar**
- Score < low → **Recusar**
- Caso contrário → **Revisar**

### 7) Métricas usadas no backtesting

O app avalia o modelo na amostra de teste e calcula:

- **AUC (ROC AUC)**: área sob a curva ROC (quanto maior, melhor; 0.5 ~ aleatório).
- **Gini** (muito usado em crédito):`Gini = 2 × AUC − 1`
- **Brier Score** (calibração de probabilidade):`Brier = (1/n) Σ (p_i − y_i)²`
- **Matriz de confusão** (com cutoff atual): contagens de acertos/erros por classe.
- **Classification report**: precision, recall e F1-score por classe e médias.
- **Validação cruzada (5-fold)** no treino com `scoring="roc_auc"` para estimar robustez.

## Interpretabilidade (coeficientes)

Como o modelo é linear no espaço transformado, cada feature tem um coeficiente:

- Coeficiente positivo: aumenta `z` e tende a aumentar `Prob_BOM` (mantidas as demais variáveis).
- Coeficiente negativo: diminui `z` e tende a reduzir `Prob_BOM`.

O painel exibe os coeficientes mais influentes (maior magnitude absoluta).

## Pontos de atenção

- A forma de Score aqui é uma escala direta da probabilidade (0–1000). Em produção, é comum usar escalas de scorecard (PDO/odds) e calibrar probabilidades.
- O `class_weight="balanced"` ajuda quando a base tem desbalanceamento entre BOM/MAU, reponderando a função de perda.
- `random_state=42` garante reprodutibilidade do split treino/teste.

## Referência de implementação

- Entrada e treino do pipeline: `train_pipeline()` em `LogRegressionCredAnalysis.py`
- Scoring e construção de Score/Rating: `score_portfolio()` e `score_to_rating()` em `LogRegressionCredAnalysis.py`
