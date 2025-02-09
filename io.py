import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from IPython.display import display, Markdown
import os
from google.colab import drive
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from collections import Counter

# --- Configurações e Constantes ---
DEFAULT_PALETTE = "viridis"
FIGSIZE = (12, 8)

sns.set(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'cyan',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': 'gray',
    'grid.linestyle': '--',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'white',
    'figure.titlesize': 20,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# --- Funções Auxiliares ---

def mount_google_drive():
    """Monta o Google Drive."""
    drive.mount('/content/drive')

def get_drive_path(relative_path):
    """Retorna o caminho completo no Drive."""
    return os.path.join('/content/drive/MyDrive', relative_path)

def ensure_directory_exists_on_drive(relative_path):
    """Cria diretório no Drive, se não existir."""
    drive_path = get_drive_path(relative_path)
    if not os.path.exists(drive_path):
        os.makedirs(drive_path, exist_ok=True)
        print(f"Diretório criado: {drive_path}")
    return drive_path

def create_figure():
    """Cria uma figura com tamanho padrão."""
    return plt.figure(figsize=FIGSIZE)

def save_fig(fig, filename, drive_folder_path):
    """Salva a figura no Drive."""
    try:
        filepath = os.path.join(drive_folder_path, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {filepath}")
    except Exception as e:
        print(f"Erro ao salvar '{filename}': {e}")
    finally:
        plt.close(fig)

# --- Funções de Visualização ---

def plot_line(df, x_col, y_col, title, filename, drive_folder_path, hue=None, style=None):
    """Plota gráfico de linha."""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas '{x_col}' ou '{y_col}' não encontradas.")
        fig = create_figure()
        sns.lineplot(x=x_col, y=y_col, data=df, hue=hue, style=style, palette=DEFAULT_PALETTE)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        if hue and hue in df.columns:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except (ValueError, KeyError, Exception) as e:
        print(f"Erro em plot_line: {e}")

def plot_histogram(df, x_col, title, filename, drive_folder_path, hue=None, bins=30, kde=True):
    """Plota histograma."""
    try:
        if x_col not in df.columns:
            raise ValueError(f"Coluna '{x_col}' não encontrada.")
        fig = create_figure()
        sns.histplot(data=df, x=x_col, hue=hue, bins=bins, palette=DEFAULT_PALETTE, kde=kde)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        if hue and hue in df.columns:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except (ValueError, KeyError, Exception) as e:
        print(f"Erro em plot_histogram: {e}")

def plot_boxplot(df, x_col, y_col, title, filename, drive_folder_path, hue=None):
    """Plota boxplot."""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas '{x_col}' ou '{y_col}' não encontradas.")
        fig = create_figure()
        sns.boxplot(x=x_col, y=y_col, data=df, hue=hue, palette=DEFAULT_PALETTE)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        if hue and hue in df.columns:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except (ValueError, KeyError, Exception) as e:
        print(f"Erro em plot_boxplot: {e}")

def plot_violin(df, x_col, y_col, title, filename, drive_folder_path, hue=None, split=False):
    """Plota violin plot."""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Colunas '{x_col}' ou '{y_col}' não encontradas.")
        fig = create_figure()
        sns.violinplot(x=x_col, y=y_col, data=df, hue=hue, split=split, palette=DEFAULT_PALETTE)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        if hue and hue in df.columns:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except (ValueError, KeyError, Exception) as e:
        print(f"Erro em plot_violin: {e}")

def plot_kde(df, x_col, title, filename, drive_folder_path, hue=None, multiple="layer"):
    """Plota KDE plot."""
    try:
        if x_col not in df.columns:
            raise ValueError(f"Coluna '{x_col}' não encontrada.")
        fig = create_figure()
        sns.kdeplot(data=df, x=x_col, hue=hue, multiple=multiple, palette=DEFAULT_PALETTE)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        save_fig(fig, filename, drive_folder_path)
    except (ValueError, KeyError, Exception) as e:
        print(f"Erro em plot_kde: {e}")

def plot_count(df, x_col, title, filename, drive_folder_path, hue=None, order=None):
    """Plota gráfico de contagem."""
    try:
        if x_col not in df.columns:
            raise ValueError(f"Coluna '{x_col}' não encontrada.")
        fig = create_figure()
        if order is None:
            order = df[x_col].value_counts().index
        sns.countplot(data=df, x=x_col, hue=hue, palette=DEFAULT_PALETTE, order=order)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.xticks(rotation=45, ha='right')  # Rotaciona rótulos do eixo x
        if hue and hue in df.columns:
            plt.legend(title=hue.replace('_', ' '), loc='upper right')
        save_fig(fig, filename, drive_folder_path)
    except (ValueError, KeyError, Exception) as e:
        print(f"Erro em plot_count: {e}")

def plot_network(graph, title, filename, drive_folder_path, pos=None):
    """Plota um grafo de rede."""
    try:
        fig = create_figure()
        if pos is None:
            pos = nx.spring_layout(graph, seed=42)  # Layout para visualização
        nx.draw(graph, pos, with_labels=False, node_size=50, node_color="skyblue", edge_color="gray")
        plt.title(title)
        save_fig(fig, filename, drive_folder_path)
    except Exception as e:
        print(f"Erro em plot_network: {e}")

def plot_topic_word_distribution(lda_model, num_words, title, filename, drive_folder_path):
    """Visualiza a distribuição de palavras por tópico."""
    try:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True)  # Ajuste para 10 tópicos
        axes = axes.flatten()

        for i, (topic_id, topic) in enumerate(lda_model.show_topics(num_topics=10, num_words=num_words, formatted=False)):
            word_probs = sorted(topic, key=lambda x: x[1], reverse=True)
            words = [word for word, _ in word_probs]
            probs = [prob for _, prob in word_probs]

            ax = axes[i]
            ax.bar(words, probs, color=plt.cm.get_cmap(DEFAULT_PALETTE)(i / 10))  # Cor consistente
            ax.set_title(f"Tópico {topic_id + 1}")
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuste para o título
        save_fig(fig, filename, drive_folder_path)

    except Exception as e:
        print(f"Erro em plot_topic_word_distribution: {e}")

# --- Funções de Análise ---

def analyze_temporal_trends(df, year_col, count_col, title, filename, drive_folder_path):
    """Analisa tendências temporais."""
    print("\n--- Análise de Tendências Temporais ---")
    try:
        if year_col not in df.columns:
            raise ValueError(f"Coluna '{year_col}' não encontrada.")

        # Agrupa por ano e conta o número de artigos/publicações
        publications_per_year = df.groupby(year_col).size().reset_index(name=count_col)

        # Visualização da tendência temporal
        plot_line(publications_per_year, year_col, count_col, title, filename, drive_folder_path)

    except (ValueError, Exception) as e:
        print(f"Erro em analyze_temporal_trends: {e}")

def analyze_citation_network(df, id_col, citations_col):
    """Analisa a rede de citações."""
    print("\n--- Análise de Redes de Citação ---")
    try:
        if id_col not in df.columns or citations_col not in df.columns:
            raise ValueError(f"Colunas '{id_col}' ou '{citations_col}' não encontradas.")

        # Cria o grafo direcionado a partir das listas de citações
        graph = nx.DiGraph()
        for _, row in df.iterrows():
            article_id = row[id_col]
            cited_ids_str = str(row[citations_col])  # Converte para string
            if cited_ids_str and cited_ids_str.lower() != 'nan':  # Verifica se não é vazio ou NaN
                cited_ids = [int(x.strip()) for x in cited_ids_str.strip("[]").split(',') if x.strip()]
                for cited_id in cited_ids:
                    graph.add_edge(article_id, cited_id)

        # Calcula métricas de centralidade
        in_degree_centrality = nx.in_degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)

        # Adiciona as métricas ao DataFrame
        df['in_degree_centrality'] = df[id_col].map(in_degree_centrality)
        df['betweenness_centrality'] = df[id_col].map(betweenness_centrality)

        return graph, df  # Retorna o grafo e o DataFrame

    except (ValueError, Exception) as e:
        print(f"Erro em analyze_citation_network: {e}")
        return nx.DiGraph(), df

def analyze_topics(df, text_col, num_topics=10, num_words=10):
    """Realiza modelagem de tópicos (LDA)."""
    print("\n--- Modelagem de Tópicos (LDA) ---")
    try:
        if text_col not in df.columns:
            raise ValueError(f"Coluna '{text_col}' não encontrada.")

        # Pré-processamento básico (tokenização, remoção de stopwords, etc.)
        texts = [str(text).lower().split() for text in df[text_col]]  # Converte para string e minúsculas

        # Cria o dicionário e o corpus
        id2word = corpora.Dictionary(texts)
        corpus = [id2word.doc2bow(text) for text in texts]

        # Treina o modelo LDA
        lda_model = LdaModel(corpus=corpus,
                             id2word=id2word,
                             num_topics=num_topics,
                             random_state=42,
                             passes=15,
                             alpha='auto',
                             eta='auto')

        # Atribui tópicos aos documentos
        df['topic'] = [max(lda_model[doc], key=lambda item: item[1])[0] for doc in corpus]

        return lda_model, df

    except (ValueError, Exception) as e:
        print(f"Erro em analyze_topics: {e}")
        return None, df

# --- Função Principal (main) ---
if __name__ == "__main__":
    mount_google_drive()
    graficos_drive_path = ensure_directory_exists_on_drive('graficos')

    display(Markdown("# Análise da Genealogia das Relações entre HD e IA (Match Ritual)"))

    # --- Carregar Dados ---
    data_url = "https://docs.google.com/spreadsheets/d/1F3euM54keaVbEHJqP_M9Z6PKmM0VsfcngBhCkmMcTQ4/export?format=csv"
    try:
        df = pd.read_csv(data_url)
        # Limpeza básica e tratamento de nomes de colunas
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df.replace('none', np.nan, inplace=True)  # Padroniza 'None' como NaN

    except Exception as e:
        print(f"Erro ao carregar dados: {e}.  Abortando.")
        exit()

    # --- Análise Exploratória e Visualizações ---
    display(Markdown("## Dados Carregados"))
    print(df.info())
    print(df.head())
    print(df.describe(include='all'))

    # --- Gráficos ---

    # 1. Publicações por Ano (Linha)
    analyze_temporal_trends(df, 'year', 'num_publications', "Número de Publicações por Ano", "publications_per_year.png", graficos_drive_path)

    # 2. Distribuição de Citações (Histograma)
    plot_histogram(df, 'citations', "Distribuição do Número de Citações", "citations_histogram.png", graficos_drive_path)

    # 3. Citações por Ano (Boxplot)
    plot_boxplot(df, 'year', 'citations', "Citações por Ano", "citations_by_year_boxplot.png", graficos_drive_path)

    # 4. Citações por Campo (Violin Plot)
    plot_violin(df, 'field', 'citations', "Distribuição de Citações por Campo", "citations_by_field_violin.png", graficos_drive_path, split=True)

    # 5. Distribuição de Citações (KDE)
    plot_kde(df, 'citations', "Densidade de Citações", "citations_kde.png", graficos_drive_path)

    # 6. Contagem de Tipos de Fonte (Count Plot)
    plot_count(df, 'source_type', "Distribuição de Tipos de Fonte", "source_type_count.png", graficos_drive_path)

    # 7. Contagem de Publicações por Campo (Count Plot)
    plot_count(df, 'field', "Número de Publicações por Campo", "field_count.png", graficos_drive_path)

    # 8. Evolução das Citações ao Longo do Tempo (Linha com Hue)
    plot_line(df, 'year', 'citations', "Evolução das Citações", "citations_over_time.png", graficos_drive_path, hue='field')

    # 9. Distribuição de Tópicos (Histograma)
    if 'topic' in df.columns:  # Verifica se a coluna 'topic' existe
        plot_histogram(df, 'topic', "Distribuição de Tópicos", "topic_distribution.png", graficos_drive_path, bins=10)

    # 10. Métodos de IA por Ano (Contagem, após transformação)
    # Primeiro, precisamos transformar a coluna 'ai_methods' em algo contável
    df['ai_methods_list'] = df['ai_methods'].str.split(', ').apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
    ai_methods_counts = df.explode('ai_methods_list').groupby('year')['ai_methods_list'].value_counts().reset_index(name='count')
    ai_methods_counts.rename(columns={'level_1': 'ai_method'}, inplace=True)
    plot_line(ai_methods_counts, 'year', 'count', "Uso de Métodos de IA ao Longo do Tempo", "ai_methods_over_time.png", graficos_drive_path, hue='ai_method', style='ai_method')

    # 11. Métodos de HD por Ano (Contagem, após transformação)
    df['hdi_methods_list'] = df['hdi_methods'].str.split(', ').apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
    hdi_methods_counts = df.explode('hdi_methods_list').groupby('year')['hdi_methods_list'].value_counts().reset_index(name='count')
    hdi_methods_counts.rename(columns={'level_1': 'hdi_method'}, inplace=True)

    if not hdi_methods_counts.empty: #Verifica se o dataframe não está vazio
        plot_line(hdi_methods_counts, 'year', 'count', "Uso de Métodos de HD ao Longo do Tempo", "hdi_methods_over_time.png", graficos_drive_path, hue='hdi_method', style='hdi_method')
    else:
        print("Não há dados suficientes para plotar a evolução dos métodos de HD.")

    # 12. Análise de Rede de Citações (se houver dados suficientes)
    graph, df = analyze_citation_network(df, 'doc_id', 'references')
    if len(graph.nodes) > 0:
        plot_network(graph, "Rede de Citações", "citation_network.png", graficos_drive_path)
        plot_histogram(df, 'in_degree_centrality', "Distribuição da Centralidade de Grau de Entrada", "in_degree_centrality_hist.png", graficos_drive_path)
        plot_histogram(df, 'betweenness_centrality', "Distribuição da Centralidade de Intermediação", "betweenness_centrality_hist.png", graficos_drive_path)
    else:
        print("Não há dados suficientes para a análise de rede de citações.")

    # 13. Modelagem de Tópicos (se aplicável)
    lda_model, df = analyze_topics(df, 'keywords', num_topics=10)
    if lda_model:
        plot_topic_word_distribution(lda_model, 10, "Distribuição de Palavras por Tópico (LDA)", "topic_word_distribution.png", graficos_drive_path)

    display(Markdown("## Conclusões"))
    display(Markdown("Resultados da análise da genealogia das relações entre HD e IA, com base nos dados carregados."))