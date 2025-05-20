import stanza
import networkx as nx
from typing import Dict, List
from functools import lru_cache
from nlp_pipeline import get_stanza_pipeline

# Зависимости, указывающие на наличие клауз
CLAUSE_DEP_LABELS = {
    'advcl', 'acl', 'relcl', 'conj', 'acl:relcl', 'root', 'aux'
}


def _extract_sentences(text: str) -> List[stanza.models.common.doc.Sentence]:
    """Фильтрует предложения, содержащие хотя бы одно буквенное слово."""
    doc = get_stanza_pipeline()(text)
    return [sent for sent in doc.sentences if any(word.text.isalpha() for word in sent.words)]


def _compute_linguistic_metrics(sentences: List[stanza.models.common.doc.Sentence]) -> Dict[str, float]:
    """
    Вычисляет основные лингвистические метрики текста.
    """
    alpha_tokens = [word for sent in sentences for word in sent.words if word.text.isalpha()]
    long_tokens = [word for word in alpha_tokens if len(word.text) >= 7]

    sentence_lengths = [
        len([word for word in sent.words if word.text.isalpha()])
        for sent in sentences
    ]

    clause_counts = [
        sum(1 for word in sent.words if word.deprel in CLAUSE_DEP_LABELS)
        for sent in sentences
    ]

    num_sentences = len(sentences)
    total_words = len(alpha_tokens)
    total_long_words = len(long_tokens)

    avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences else 0
    avg_clauses_per_sentence = sum(clause_counts) / num_sentences if num_sentences else 0

    lix_score = (
        total_words / num_sentences + (total_long_words * 100 / total_words)
        if total_words and num_sentences else 0
    )
    rix_score = total_long_words / num_sentences if num_sentences else 0

    return {
        'n_sentences': num_sentences,
        'avg_sentence_length': round(avg_sentence_length, 2),
        'avg_clauses_per_sentence': round(avg_clauses_per_sentence, 2),
        'lix_score': round(lix_score, 2),
        'rix_score': round(rix_score, 2),
    }


def _build_dependency_graph(sentences: List[stanza.models.common.doc.Sentence]) -> nx.DiGraph:
    """
    Формирует ориентированный граф синтаксических зависимостей.
    """
    graph = nx.DiGraph()
    token_index = 0

    for sent in sentences:
        word_map = {word.id: token_index + i for i, word in enumerate(sent.words)}
        for word in sent.words:
            if word.upos not in {"PUNCT", "SPACE"}:
                graph.add_node(word_map[word.id])
                if word.head != 0:
                    head_index = word_map.get(word.head)
                    if head_index is not None:
                        graph.add_edge(head_index, word_map[word.id])
        token_index += len(sent.words)

    return graph


def _compute_graph_metrics(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Вычисляет метрики графа: плотность и глобальную эффективность.
    """
    density = nx.density(graph)
    efficiency = nx.global_efficiency(graph.to_undirected()) if graph.number_of_nodes() > 1 else 0
    return {
        'network_density': round(density, 4),
        'network_efficiency': round(efficiency, 4)
    }


def _compute_tree_depth(sentences: List[stanza.models.common.doc.Sentence]) -> float:
    """
    Вычисляет среднюю глубину дерева синтаксических зависимостей.
    """

    def dfs(word_id, children_map):
        if word_id not in children_map:
            return 1
        return 1 + max(dfs(child, children_map) for child in children_map[word_id])

    depths = []
    for sent in sentences:
        children_map = {}
        root_id = None
        for word in sent.words:
            if word.head == 0:
                root_id = word.id
            else:
                children_map.setdefault(word.head, []).append(word.id)

        if root_id is not None:
            depth = dfs(root_id, children_map)
            depths.append(depth)

    return round(sum(depths) / len(depths), 2) if depths else 0


def analyze_text_synt(text: str) -> Dict[str, float]:
    """
    Выполняет комплексный синтаксический и графовый анализ текста, возвращая ключевые метрики.
    """
    sentences = _extract_sentences(text)
    linguistic_metrics = _compute_linguistic_metrics(sentences)
    graph = _build_dependency_graph(sentences)
    graph_metrics = _compute_graph_metrics(graph)
    tree_depth = _compute_tree_depth(sentences)

    return {
        **linguistic_metrics,
        **graph_metrics,
        'avg_tree_depth': tree_depth
    }
