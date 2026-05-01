# ============================================
# 0. IMPORTAÇÕES
# ============================================
# Aqui importamos todas as bibliotecas usadas no app:
# - Visualização (matplotlib/seaborn)
# - Manipulação de dados (numpy/pandas)
# - Interface web (streamlit)
# - Modelo e métricas (scikit-learn)
import matplotlib.pyplot as plt  # Biblioteca de gráficos (Matplotlib); aqui usamos pyplot para criar figuras e eixos (fig, ax).
import numpy as np  # Biblioteca para computação numérica; usada para vetorização, np.select, np.where, etc.
import pandas as pd  # Biblioteca para dados tabulares; usada para ler CSV, manipular DataFrames e Series.
import seaborn as sns  # Biblioteca de visualização em cima do Matplotlib; usada para plotar a matriz de confusão como heatmap.
import streamlit as st  # Framework para construir a interface web do painel (widgets, abas, sidebar, etc.).
from sklearn import metrics  # Módulo com métricas de avaliação (AUC, ROC, matriz de confusão, relatório de classificação, Brier).
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística (classificador probabilístico).
from sklearn.model_selection import cross_val_score, train_test_split  # Funções para separar treino/teste e para validação cruzada.
from sklearn.preprocessing import StandardScaler  # Padronizador (z-score) para normalizar features numéricas antes do modelo.

# ============================================
# 0.1 CONFIGURAÇÃO DO APP (STREAMLIT)
# ============================================
# Define título, layout e estado inicial da barra lateral.
st.set_page_config(page_title="LogRegressionCredAnalysis", layout="wide", initial_sidebar_state="expanded")  # Define metadados e layout inicial do app Streamlit.

SIDEBAR_STYLE = """
<style>
  [data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
  }

  [data-testid="stSidebar"] * {
    color: #e5e7eb;
  }

  [data-testid="stSidebar"] a {
    color: #93c5fd;
  }

  [data-testid="stSidebar"] [data-baseweb="radio"] *,
  [data-testid="stSidebar"] [data-baseweb="checkbox"] *,
  [data-testid="stSidebar"] [data-baseweb="select"] *,
  [data-testid="stSidebar"] [data-baseweb="slider"] * {
    color: #e5e7eb;
  }

  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] textarea {
    color: #111827 !important;
  }

  [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35) !important;
  }
</style>
"""

st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

# ============================================
# 0.2 REGRAS DE NEGÓCIO (RATING)
# ============================================
# Define uma ordem fixa de categorias para uso em filtros, gráficos e relatórios.
RATING_ORDER = [  # Lista ordenada que define a hierarquia dos ratings do melhor (A) ao pior (E).
    "Rating A (Risco Mínimo)",
    "Rating B (Risco Baixo)",
    "Rating C (Risco Médio)", 
    "Rating D (Risco Alto)",
    "Rating E (Risco Crítico)",
]  # Mantemos essa ordem para gráficos, filtros e relatórios ficarem consistentes.


def score_to_rating(score: pd.Series) -> pd.Series:  # Converte um score numérico em uma categoria textual de Rating.
    condicoes = [  # Lista de condições booleanas (em ordem); a primeira condição verdadeira define o rating.
        (score >= 850),  # Se score é muito alto, classifica como Rating A.
        (score >= 700),  # Se não foi A, mas ainda é alto, classifica como Rating B.
        (score >= 500),  # Se não foi B, mas é mediano, classifica como Rating C.
        (score >= 300),  # Se não foi C, mas não é extremamente baixo, classifica como Rating D.
        (score < 300),  # Caso extremo (muito baixo), classifica como Rating E.
    ]  # Observação: os limites são arbitrários/regra de negócio e podem ser ajustados.
    escolhas = RATING_ORDER  # Valores atribuídos para cada condição, respeitando a mesma ordem do array condicoes.
    return pd.Series(  # Retorna uma Series alinhada ao índice original para facilitar merges/assign.
        np.select(condicoes, escolhas, default="Sem Classificação"),  # np.select aplica condições vetorizadas e escolhe o rótulo.
        index=score.index,  # Garante que a Series devolvida tenha o mesmo índice do score de entrada.
    )


@st.cache_data  # Cacheia o resultado (DataFrame) para não reler o CSV a cada interação do usuário.
def load_data(path: str) -> pd.DataFrame:  # Carrega o dataset bruto a partir de um arquivo CSV.
    df = pd.read_csv(path)  # Lê o CSV em um DataFrame do Pandas.
    if "nome_cliente" not in df.columns:  # Valida se a coluna-chave (identificador) existe.
        raise ValueError("Coluna obrigatória ausente: nome_cliente")  # Falha cedo com mensagem clara para facilitar debug.
    if "risco_credito" not in df.columns:  # Valida se a coluna alvo (label) existe.
        raise ValueError("Coluna obrigatória ausente: risco_credito")  # Evita treinar modelo sem o alvo.
    return df  # Devolve o DataFrame já validado.


@st.cache_resource  # Cacheia recursos "pesados" (modelo/scaler) para não retreinar a cada interação.
def train_pipeline(df: pd.DataFrame):  # Treina o pipeline: one-hot encoding + split + scaler + regressão logística.
    X = df.drop(columns=["risco_credito"])  # Features: todas as colunas, exceto o alvo.
    y = df["risco_credito"]  # Alvo binário esperado (ex.: 1=BOM, 0=MAU).

    X_encoded_all = pd.get_dummies(X, drop_first=True)  # One-hot encoding para variáveis categóricas (drop_first evita colinearidade perfeita).
    X_train, X_test, y_train, y_test = train_test_split(  # Separa dados em treino e teste para avaliar generalização.
        X_encoded_all,  # Features já codificadas.
        y,  # Alvo correspondente.
        test_size=0.3,  # Reserva 30% para teste.
        random_state=42,  # Semente fixa para reprodutibilidade.
    )

    scaler = StandardScaler()  # Cria o normalizador (média 0, desvio 1 por coluna).
    X_train_scaled = scaler.fit_transform(X_train)  # Ajusta o scaler no treino e transforma o treino.
    X_test_scaled = scaler.transform(X_test)  # Transforma o teste usando parâmetros aprendidos no treino (sem vazar informação).

    model = LogisticRegression(max_iter=1000, class_weight="balanced")  # Define o modelo; balanceamento ajuda se classes forem desbalanceadas.
    model.fit(X_train_scaled, y_train)  # Treina o modelo para estimar P(y=1|X).

    return {  # Retorna um "pacote" com tudo que o app precisa para scoring e avaliação.
        "model": model,  # Modelo treinado.
        "scaler": scaler,  # Scaler ajustado no treino.
        "X_train_columns": X_train.columns,  # Referência do conjunto de colunas após one-hot (para alinhar novos dados).
        "X_train_scaled": X_train_scaled,  # Treino transformado (para validação cruzada/relatórios).
        "X_test_scaled": X_test_scaled,  # Teste transformado (para métricas e gráficos).
        "X_test_index": X_test.index,  # Índices das linhas do teste (para resgatar clientes na base com score).
        "y_train": y_train,  # Alvos do treino.
        "y_test": y_test,  # Alvos do teste.
    }


def score_portfolio(  # Faz o scoring (probabilidade/score/rating) para um DataFrame com os mesmos campos do treino.
    df: pd.DataFrame,  # Base com colunas de features + risco_credito (alvo), para permitir comparação.
    model: LogisticRegression,  # Modelo treinado para estimar probabilidades.
    scaler: StandardScaler,  # Scaler ajustado no treino para padronizar features.
    columns_ref,  # Colunas esperadas (após one-hot) conforme o treino, para alinhar o shape.
) -> pd.DataFrame:
    X = df.drop(columns=["risco_credito"])  # Remove o alvo para formar o conjunto de features.
    X_encoded = pd.get_dummies(X, drop_first=True)  # Aplica one-hot encoding no portfolio.
    X_aligned = X_encoded.reindex(columns=columns_ref, fill_value=0)  # Alinha colunas ao treino; categorias ausentes viram 0.

    X_scaled = scaler.transform(X_aligned)  # Padroniza usando o scaler do treino.
    proba_bom = model.predict_proba(X_scaled)[:, 1]  # Extrai P(classe=1) (probabilidade de BOM).

    df_scored = df.copy()  # Copia para não alterar o DataFrame original.
    df_scored["Prob_BOM"] = proba_bom  # Guarda a probabilidade estimada de BOM.
    df_scored["Score"] = (df_scored["Prob_BOM"] * 1000).round().astype(int)  # Converte probabilidade em score 0..1000 (escala simples).
    df_scored["Rating"] = score_to_rating(df_scored["Score"])  # Mapeia score para rating conforme regras de corte.
    df_scored["Real"] = df_scored["risco_credito"].map({1: "BOM", 0: "MAU"})  # Mapeia o alvo numérico em rótulos legíveis.

    return df_scored  # Devolve o DataFrame com colunas adicionais de scoring.


def _format_value_column(df: pd.DataFrame) -> dict:  # Monta configurações de exibição para colunas específicas no st.dataframe.
    cfg = {}  # Dicionário acumulador do column_config para o Streamlit.
    if "valor" in df.columns:  # Se existir a coluna de valor monetário, formata como moeda.
        cfg["valor"] = st.column_config.NumberColumn("Valor solicit.", format="R$ %.2f")  # Define rótulo e formato brasileiro.
    if "Score" in df.columns:  # Se existir score, exibe como barra de progresso (0..1000).
        cfg["Score"] = st.column_config.ProgressColumn("Score", min_value=0, max_value=1000, format="%d")  # Ajuda leitura rápida no comitê.
    if "Prob_BOM" in df.columns:  # Se existir probabilidade, exibe como barra de progresso (0..1).
        cfg["Prob_BOM"] = st.column_config.ProgressColumn("Prob. BOM", min_value=0.0, max_value=1.0, format="%.2f")  # Facilita comparação visual.
    return cfg  # Retorna o dicionário pronto para usar em st.dataframe(..., column_config=cfg).


# ============================================
# 1. CARREGAR O DATASET
# ============================================
# Lê o CSV e prepara o DataFrame base do projeto.
# Em seguida, definimos o índice como nome do cliente para facilitar filtros e buscas.
df_raw = load_data("BrazilianCredit.csv")  # Carrega a base CSV do projeto.
df_raw = df_raw.copy()  # Cria cópia defensiva para evitar efeitos colaterais em objetos cacheados/compartilhados.
df = df_raw.set_index("nome_cliente")  # Define o nome do cliente como índice para facilitar buscas e filtros por cliente.

# ============================================
# 2. TREINAR / RECUPERAR PIPELINE (MODELO + SCALER)
# ============================================
# Executa o pré-processamento (get_dummies + padronização) e treina a Regressão Logística.
# O retorno inclui objetos e dados necessários para avaliar e fazer scoring.
pipeline = train_pipeline(df)  # Treina (ou recupera do cache) o pipeline de ML com base nos dados atuais.
model = pipeline["model"]  # Extrai o modelo treinado.
scaler = pipeline["scaler"]  # Extrai o scaler treinado.
X_train_columns = pipeline["X_train_columns"]  # Colunas de referência do treino (após one-hot).
y_train = pipeline["y_train"]  # Alvos de treino.
y_test = pipeline["y_test"]  # Alvos de teste.
X_test_scaled = pipeline["X_test_scaled"]  # Features do teste já padronizadas.
X_test_index = pipeline["X_test_index"]  # Índices das linhas que ficaram no conjunto de teste.

# ============================================
# 3. SCORING (PROBABILIDADE, SCORE E RATING)
# ============================================
# Gera Prob_BOM (probabilidade de ser BOM) e deriva Score (0..1000) e Rating.
# Também separamos uma base de teste para backtesting/validação.
y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]  # Probabilidade prevista de BOM no conjunto de teste.
df_scored_all = score_portfolio(df.reset_index(), model, scaler, X_train_columns)  # Gera score/rating para toda a carteira (inclui coluna alvo para comparação).
df_scored_all = df_scored_all.set_index("nome_cliente")  # Recoloca nome_cliente como índice após o reset_index usado no scoring.

df_scored_test = df_scored_all.loc[X_test_index].copy()  # Recorta apenas os clientes que caíram no teste (para backtesting).
df_scored_test["Prob_BOM"] = y_pred_proba_test  # Garante que a probabilidade do teste seja exatamente a calculada no X_test_scaled.
df_scored_test["Score"] = (df_scored_test["Prob_BOM"] * 1000).round().astype(int)  # Recalcula score com base na probabilidade do teste.
df_scored_test["Rating"] = score_to_rating(df_scored_test["Score"])  # Recalcula rating com base no score do teste.

# ============================================
# 4. INTERFACE (TÍTULO E CONTROLES)
# ============================================
# Define elementos de UI do Streamlit: título, descrição e filtros na barra lateral.
HEADER_STYLE = """
<style>
.app-header{
  position: sticky;
  top: 0;
  z-index: 999;
  padding: 0.35rem 0 0.35rem 0;
  margin-bottom: 0.5rem;
  background: rgba(255, 255, 255, 0.86);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}
@media (prefers-color-scheme: dark){
  .app-header{
    background: rgba(2, 6, 23, 0.55);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
}
.app-header h1{
  font-size: 1.6rem;
  line-height: 1.15;
  margin: 0;
  padding: 0;
}
.app-header p{
  margin: 0.15rem 0 0 0;
  opacity: 0.85;
}
</style>
"""

st.markdown(HEADER_STYLE, unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-header">
      <h1>LogRegressionCredAnalysis</h1>
      <p>Modelo: Regressão Logística | Objetivo: estimar probabilidade de bom pagador (Prob_BOM) e derivar Score e Rating.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:  # Inicia o bloco que renderiza widgets na barra lateral do Streamlit.
    st.header("Controles")  # Cabeçalho da sidebar para agrupar filtros e parâmetros.
    base_exibida = st.radio(  # Widget de escolha única para selecionar qual base visualizar.
        "Base",  # Rótulo do widget.
        ["Carteira completa", "Amostra de teste (backtesting)"],  # Opções disponíveis para o usuário.
        index=0,  # Opção inicial (0 = carteira completa).
    )

    cutoff_score = st.slider("Cutoff de decisão (Score)", 0, 1000, 500, step=10)  # Slider para definir o limiar de aprovação/reprovação em score.
    cutoff_proba = cutoff_score / 1000  # Converte score em probabilidade equivalente (já que score = prob*1000).

    usar_revisao = st.checkbox("Habilitar zona de revisão", value=True)  # Liga/desliga uma faixa intermediária para decisões "Revisar".
    largura_revisao = (  # Define a largura da faixa de revisão (em score) caso habilitada.
        st.slider("Largura da revisão (Score)", 0, 200, 100, step=10) if usar_revisao else 0  # Se desabilitada, largura vira 0.
    )

    ratings_sel = st.multiselect("Ratings", options=RATING_ORDER, default=RATING_ORDER)  # Filtro por rating (permite múltipla seleção).
    faixa_score = st.slider("Faixa de Score", 0, 1000, (0, 1000), step=10)  # Filtro por intervalo de score (mín, máx).
    buscar_cliente = st.text_input("Buscar cliente (nome)", value="")  # Campo de texto para filtrar por substring do nome do cliente.
    somente_divergencias = st.checkbox("Somente divergências (Previsto ≠ Real)", value=False)  # Mostra apenas casos onde previsão difere do real.

# ============================================
# 5. REGRAS DE DECISÃO E FILTROS
# ============================================
# A partir dos controles (cutoff, revisão, rating, score, busca), montamos a visão final (df_view).
df_base = df_scored_all if base_exibida == "Carteira completa" else df_scored_test  # Seleciona a base conforme a opção da sidebar.
df_view = df_base.copy()  # Copia para aplicar filtros/colunas sem alterar o original.

low = (  # Limite inferior da zona de revisão.
    max(0, int(cutoff_score - (largura_revisao / 2))) if usar_revisao and largura_revisao > 0 else cutoff_score  # Se não houver revisão, low = cutoff.
)
high = (  # Limite superior da zona de revisão.
    min(1000, int(cutoff_score + (largura_revisao / 2))) if usar_revisao and largura_revisao > 0 else cutoff_score  # Se não houver revisão, high = cutoff.
)

df_view["Previsto"] = np.where(df_view["Prob_BOM"] >= cutoff_proba, "BOM", "MAU")  # Converte probabilidade em classe prevista usando o cutoff.
if usar_revisao and largura_revisao > 0:  # Se a zona de revisão estiver habilitada e tiver largura > 0, cria 3 estados de decisão.
    df_view["Decisão"] = np.where(  # Usa regra baseada em score para Aprovar/Recusar/Revisar.
        df_view["Score"] >= high,  # Se score acima do limite superior...
        "Aprovar",  # ...aprova automaticamente.
        np.where(df_view["Score"] < low, "Recusar", "Revisar"),  # Se abaixo do limite inferior, recusa; senão, manda para revisão.
    )
else:  # Caso não exista zona de revisão, mantém decisão binária simples.
    df_view["Decisão"] = np.where(df_view["Score"] >= cutoff_score, "Aprovar", "Recusar")  # Regra binária: acima do cutoff aprova, abaixo recusa.

df_view = df_view[df_view["Rating"].isin(ratings_sel)]  # Aplica filtro de rating selecionado.
df_view = df_view[(df_view["Score"] >= faixa_score[0]) & (df_view["Score"] <= faixa_score[1])]  # Aplica filtro por faixa de score.

if buscar_cliente.strip():  # Só filtra por nome se o usuário digitou algo além de espaços.
    busca = buscar_cliente.strip().lower()  # Normaliza a busca (trim + minúsculas) para comparação case-insensitive.
    df_view = df_view[df_view.index.to_series().astype(str).str.lower().str.contains(busca)]  # Mantém somente clientes cujo nome contém o termo buscado.

if somente_divergencias:  # Se habilitado, mostra apenas erros do modelo (previsto diferente do real).
    df_view = df_view[df_view["Previsto"] != df_view["Real"]]  # Filtra divergências para análise de casos problemáticos.

# ============================================
# 6. INDICADORES (KPIs)
# ============================================
# Calcula métricas rápidas da visão atual para o comitê.
qtde = int(df_view.shape[0])  # Quantidade de clientes na visão atual (após filtros).
score_medio = float(df_view["Score"].mean()) if qtde else np.nan  # Score médio; se não houver linhas, evita erro e usa NaN.
taxa_maus = float((df_view["Real"] == "MAU").mean()) if qtde else np.nan  # Proporção de MAU na base filtrada.
taxa_aprov = float((df_view["Decisão"] == "Aprovar").mean()) if qtde else np.nan  # Proporção de decisões de aprovação na base filtrada.

col1, col2, col3, col4 = st.columns(4)  # Cria 4 colunas na UI para exibir KPIs lado a lado.
col1.metric("Clientes (visão atual)", f"{qtde:,}".replace(",", "."))  # Mostra volume de clientes com separador de milhar no padrão PT-BR.
col2.metric("Score médio", "-" if np.isnan(score_medio) else f"{score_medio:.0f}")  # Mostra score médio (ou "-" se vazio).
col3.metric("Taxa de MAU (real)", "-" if np.isnan(taxa_maus) else f"{taxa_maus:.1%}")  # Mostra taxa real de maus pagadores.
col4.metric("Taxa de aprovação", "-" if np.isnan(taxa_aprov) else f"{taxa_aprov:.1%}")  # Mostra taxa de aprovação segundo a regra atual.

# ============================================
# 7. ABAS (RESUMO, CLIENTES, MODELO, RELATÓRIO)
# ============================================
# Organiza as saídas do painel em seções: gráficos, tabela, avaliação do modelo e relatório textual.
tab_resumo, tab_clientes, tab_modelo, tab_relatorio = st.tabs(["Resumo", "Clientes", "Modelo", "Relatório"])  # Cria abas para separar visualizações.

with tab_resumo:  # Conteúdo da aba "Resumo".
    col_a, col_b = st.columns(2)  # Divide a aba em duas colunas para gráficos lado a lado.

    with col_a:  # Primeira coluna: distribuição por Rating.
        st.subheader("Distribuição por Rating")  # Subtítulo da seção.
        rating_counts = df_view["Rating"].value_counts().reindex(RATING_ORDER, fill_value=0)  # Conta clientes por rating e garante a ordem A..E.
        fig, ax = plt.subplots(figsize=(6.0, 3.0))  # Cria figura e eixo para o gráfico.
        ax.bar(list(rating_counts.index), rating_counts.to_numpy(), color="#1f77b4")  # Plota barras com contagem por rating.
        ax.set_ylabel("Clientes")  # Rótulo do eixo Y.
        ax.tick_params(axis="x", rotation=25)  # Inclina rótulos do eixo X para caberem.
        ax.grid(axis="y", alpha=0.25)  # Adiciona grade horizontal leve para leitura.
        st.pyplot(fig, use_container_width=True)  # Renderiza a figura no Streamlit usando toda a largura disponível.

    with col_b:  # Segunda coluna: distribuição de Score.
        st.subheader("Distribuição de Score")  # Subtítulo da seção.
        fig, ax = plt.subplots(figsize=(6.0, 3.0))  # Cria figura e eixo para o histograma.
        ax.hist(df_view["Score"], bins=30, color="#2ca02c", alpha=0.85)  # Histograma de scores na visão atual.
        ax.axvline(cutoff_score, color="#d62728", linestyle="--", lw=2, label=f"Cutoff {cutoff_score}")  # Linha vertical indicando o cutoff atual.
        if usar_revisao and largura_revisao > 0:  # Se existir revisão, destaca a faixa intermediária.
            ax.axvspan(low, high, color="#ff7f0e", alpha=0.15, label="Zona de revisão")  # Faixa sombreada entre low e high.
        ax.set_xlabel("Score")  # Rótulo do eixo X.
        ax.set_ylabel("Clientes")  # Rótulo do eixo Y.
        ax.grid(alpha=0.25)  # Grade leve no gráfico.
        ax.legend()  # Exibe legenda com cutoff e zona de revisão.
        st.pyplot(fig, use_container_width=True)  # Renderiza a figura na UI.

    st.subheader("Risco observado por Rating (Real)")  # Seção para medir risco "observado" (taxa de MAU) por rating.
    risco_por_rating = (  # Calcula taxa de MAU por rating usando o rótulo real.
        df_view.assign(is_mau=(df_view["Real"] == "MAU").astype(int))  # Cria coluna is_mau (1 se real=MAU, senão 0).
        .groupby("Rating", dropna=False)["is_mau"]  # Agrupa por rating e seleciona a coluna is_mau.
        .mean()  # Média de is_mau = proporção de maus (bad rate) em cada rating.
        .reindex(RATING_ORDER)  # Reordena para manter a ordem A..E.
    )
    risco_df = risco_por_rating.reset_index().rename(columns={"is_mau": "Taxa_MAU"})  # Converte para DataFrame e renomeia coluna para exibição.
    st.dataframe(  # Mostra tabela com taxa de MAU por rating.
        risco_df,  # DataFrame a ser exibido.
        use_container_width=True,  # Usa toda a largura da área.
        column_config={  # Ajusta formato da coluna de taxa.
            "Taxa_MAU": st.column_config.NumberColumn("Taxa de MAU", format="%.2f"),  # Mostra com 2 casas decimais (ex.: 0.12).
        },
    )

with tab_clientes:  # Conteúdo da aba "Clientes".
    st.subheader("Tabela para Comitê")  # Título da tabela principal.
    cols_view = [  # Ordem preferida das colunas exibidas (se existirem na base).
        "Score",  # Score calculado.
        "Rating",  # Rating derivado do score.
        "Decisão",  # Decisão gerada pela regra (Aprovar/Recusar/Revisar).
        "Prob_BOM",  # Probabilidade de bom pagador.
        "Real",  # Classe real (BOM/MAU) derivada do alvo.
        "status_conta",  # Feature original (se estiver no dataset).
        "historico_credito",  # Feature original (se estiver no dataset).
        "poupanca",  # Feature original (se estiver no dataset).
        "valor",  # Feature original (se estiver no dataset).
        "idade",  # Feature original (se estiver no dataset).
        "duracao_meses",  # Feature original (se estiver no dataset).
        "tempo_emprego",  # Feature original (se estiver no dataset).
    ]
    cols_view = [c for c in cols_view if c in df_view.columns]  # Remove colunas que não existirem no DataFrame atual.
    df_grid = (  # Prepara a grade final: seleciona colunas, ordena e deixa o índice como coluna "Cliente".
        df_view[cols_view]  # Seleciona apenas as colunas desejadas.
        .sort_values(by="Score", ascending=False)  # Ordena do maior score para o menor (prioriza melhores clientes).
        .reset_index()  # Traz o índice (nome_cliente) de volta como coluna.
        .rename(columns={"nome_cliente": "Cliente"})  # Renomeia a coluna para ficar mais amigável na UI.
    )
    st.dataframe(  # Renderiza a tabela no Streamlit.
        df_grid,  # DataFrame preparado para exibição.
        use_container_width=True,  # Usa toda a largura.
        height=540,  # Altura fixa para facilitar navegação.
        column_config=_format_value_column(df_grid),  # Aplica formatos especiais (moeda/progress bars).
    )

    st.subheader("Exportação")  # Seção para exportar a visão filtrada.
    csv = df_grid.to_csv(index=False).encode("utf-8")  # Converte DataFrame para CSV (bytes) para download.
    st.download_button(  # Botão de download do CSV com a visão atual (pós-filtros).
        "Baixar CSV (visão atual)",  # Texto do botão.
        data=csv,  # Conteúdo do arquivo.
        file_name="painel_credito_visao_atual.csv",  # Nome sugerido para salvar.
        mime="text/csv",  # Tipo MIME para o navegador identificar.
    )

with tab_modelo:  # Conteúdo da aba "Modelo".
    st.subheader("Backtesting (base de teste)")  # Seção de avaliação usando a amostra de teste.
    auc = metrics.roc_auc_score(y_test, y_pred_proba_test)  # Calcula AUC (área sob a curva ROC) no teste.
    gini = (2 * auc) - 1  # Converte AUC em Gini (métrica comum em crédito).
    brier = metrics.brier_score_loss(y_test, y_pred_proba_test)  # Mede calibração: erro quadrático médio das probabilidades.
    cv_scores = cross_val_score(  # Avalia AUC via validação cruzada no treino (estimativa de robustez).
        model,  # Estimador (regressão logística).
        pipeline["X_train_scaled"],  # Features de treino já transformadas.
        y_train,  # Alvo de treino.
        cv=5,  # 5 folds.
        scoring="roc_auc",  # Métrica de score usada na CV.
    )

    col1, col2, col3, col4 = st.columns(4)  # Cria 4 colunas para mostrar métricas lado a lado.
    col1.metric("AUC", f"{auc:.4f}")  # Exibe AUC no teste.
    col2.metric("Gini", f"{gini:.4f}")  # Exibe Gini no teste.
    col3.metric("CV AUC (média)", f"{cv_scores.mean():.4f}")  # Exibe média do AUC na validação cruzada.
    col4.metric("Brier", f"{brier:.4f}")  # Exibe Brier score.

    y_pred_bin = (y_pred_proba_test >= cutoff_proba).astype(int)  # Converte probabilidades em classe 0/1 usando o cutoff atual.
    cm = metrics.confusion_matrix(y_test, y_pred_bin)  # Calcula matriz de confusão (Real x Previsto).

    col_a, col_b = st.columns(2)  # Divide em duas colunas: matriz de confusão e curva ROC.
    with col_a:  # Coluna da matriz de confusão e relatório.
        st.markdown("**Matriz de confusão (cutoff atual)**")  # Título em negrito usando Markdown.
        fig, ax = plt.subplots(figsize=(5.4, 3.8))  # Cria figura para o heatmap.
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)  # Plota a matriz com contagens.
        ax.set_xlabel("Previsto")  # Nomeia eixo X como previsto.
        ax.set_ylabel("Real")  # Nomeia eixo Y como real.
        ax.set_xticklabels(["MAU", "BOM"])  # Define rótulos das classes no eixo X.
        ax.set_yticklabels(["MAU", "BOM"], rotation=0)  # Define rótulos no eixo Y sem rotação.
        st.pyplot(fig, use_container_width=True)  # Renderiza a figura.

        resumo = metrics.classification_report(  # Calcula precision/recall/f1 por classe e médias.
            y_test,  # Verdadeiro.
            y_pred_bin,  # Predito binário.
            output_dict=True,  # Retorna estrutura em dict para virar DataFrame.
            zero_division=0,  # Evita warnings/erros quando alguma métrica divide por zero.
        )
        resumo_df = pd.DataFrame(resumo).T  # Converte o dict em DataFrame e transpõe para ficar em formato de tabela.
        st.dataframe(resumo_df, use_container_width=True, height=320)  # Exibe o relatório como tabela.

    with col_b:  # Coluna da curva ROC.
        st.markdown("**Curva ROC**")  # Título em negrito usando Markdown.
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba_test)  # Calcula pontos da ROC (FPR vs TPR) varrendo thresholds.
        fig, ax = plt.subplots(figsize=(5.4, 3.8))  # Cria figura e eixo.
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f} | Gini = {gini:.3f}", color="#1f77b4", lw=2)  # Plota ROC do modelo.
        ax.plot([0, 1], [0, 1], color="#d62728", linestyle="--", lw=1.5)  # Plota linha aleatória (baseline).
        ax.set_xlabel("Falsos Positivos")  # Eixo X: taxa de falsos positivos.
        ax.set_ylabel("Verdadeiros Positivos")  # Eixo Y: taxa de verdadeiros positivos.
        ax.grid(alpha=0.25)  # Grade leve.
        ax.legend(loc="lower right")  # Mostra legenda no canto inferior direito.
        st.pyplot(fig, use_container_width=True)  # Renderiza a figura.

    st.subheader("Variáveis (coeficientes)")  # Seção para inspecionar interpretabilidade do modelo linear.
    coef = pd.DataFrame({"Variável": X_train_columns, "Coeficiente": model.coef_[0]})  # Monta tabela Variável x Coeficiente.
    coef["Abs"] = np.abs(coef["Coeficiente"])  # Cria coluna auxiliar com valor absoluto para ordenar por importância (magnitude).
    coef = coef.sort_values("Abs", ascending=False).drop(columns=["Abs"])  # Ordena por magnitude e remove a coluna auxiliar.
    st.dataframe(coef.head(40), use_container_width=True, height=520)  # Mostra top 40 coeficientes (mais influentes).

with tab_relatorio:  # Conteúdo da aba "Relatório".
    st.subheader("Relatório (texto)")  # Título da seção.
    linhas = len(df_raw)  # Número de linhas na base original.
    dist_alvo = df_raw["risco_credito"].value_counts(dropna=False).sort_index()  # Distribuição do alvo (quantos 0 e 1).
    rating_counts = df_scored_all["Rating"].value_counts().reindex(RATING_ORDER, fill_value=0)  # Distribuição de ratings na carteira completa.
    bad_rate = (  # Calcula taxa de MAU por rating na carteira completa.
        df_scored_all.assign(is_mau=(df_scored_all["Real"] == "MAU").astype(int))  # Coluna binária indicando MAU real.
        .groupby("Rating", dropna=False)["is_mau"]  # Agrupa por rating.
        .mean()  # Média = taxa de MAU.
        .reindex(RATING_ORDER)  # Ordena A..E.
    )

    texto = []  # Lista de linhas de texto para montar um relatório simples.
    texto.append("PAINEL DE CRÉDITO — RELATÓRIO")  # Título do relatório.
    texto.append(f"Linhas: {linhas:,}".replace(",", "."))  # Total de linhas formatado no padrão PT-BR.
    texto.append("")  # Linha em branco para separar seções.
    texto.append("Distribuição do alvo (risco_credito):")  # Cabeçalho da seção do alvo.
    texto.append(dist_alvo.to_string())  # Conteúdo textual da contagem do alvo.
    texto.append("")  # Separador.
    texto.append("Métricas (teste):")  # Cabeçalho das métricas.
    texto.append(f"AUC:  {metrics.roc_auc_score(y_test, y_pred_proba_test):.4f}")  # AUC no teste.
    texto.append(f"Gini: {(2 * metrics.roc_auc_score(y_test, y_pred_proba_test) - 1):.4f}")  # Gini no teste.
    texto.append(f"Brier: {metrics.brier_score_loss(y_test, y_pred_proba_test):.4f}")  # Brier no teste.
    texto.append("")  # Separador.
    texto.append("Distribuição por Rating (carteira):")  # Cabeçalho da distribuição por rating.
    texto.append(rating_counts.to_string())  # Conteúdo textual da contagem por rating.
    texto.append("")  # Separador.
    texto.append("Taxa de MAU (real) por Rating (carteira):")  # Cabeçalho da taxa de MAU por rating.
    texto.append(((bad_rate * 100).round(2)).to_string() + " %")  # Formata taxas em porcentagem com 2 casas.

    st.text("\n".join(texto))  # Exibe o relatório como texto monoespaçado no Streamlit.
