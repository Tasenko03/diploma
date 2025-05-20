import torch
import numpy as np
import nltk
import re
from collections import Counter
from scipy.stats import entropy
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import logging
from typing import List, Dict, Any
from functools import lru_cache
from nlp_pipeline import get_stanza_pipeline

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')

# Константы для перплексии
MIN_PPL = 10.0
MAX_PPL = 1000.0

# Связки для анализа
POSITIVE_CONNECTORS: List[str] = [
    'и', 'а также', 'к тому же', 'более того', 'также',
    'да', 'помимо этого', 'вдобавок', 'причем', 'не только',
    'следовательно', 'итак', 'таким образом'
]

NEGATIVE_CONNECTORS: List[str] = [
    'но', 'однако', 'тем не менее', 'в то время как',
    'хотя', 'зато', 'несмотря на', 'в отличие от',
    'напротив', 'в противовес', 'вопреки'
]

class TextCohesionAnalyzer:
    def __init__(self):
        """Инициализация моделей и пайплайнов для анализа текста."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1

        logger.info("Загрузка модели эмбеддингов...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        logger.info("Загрузка модели перплексии...")
        self.coherence_model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
        self.coherence_tokenizer = AutoTokenizer.from_pretrained(self.coherence_model_name)
        self.coherence_model = AutoModelForCausalLM.from_pretrained(self.coherence_model_name).to(self.device)

        logger.info("Загрузка NER пайплайна...")
        self.ner_pipeline = pipeline(
            "ner",
            model="ai-forever/ruBert-base",
            tokenizer="ai-forever/ruBert-base",
            aggregation_strategy="simple",
            device=self.device_id
        )

        # Кэш для результатов
        self._cache_embedding = {}
        self._cache_perplexity = {}
        self._cache_tfidf = {}
        self._cache_entity = {}
        self._cache_thematic = {}

    def calculate_cohesion_embedding(self, text: str) -> float:
        """Вычисляет среднюю косинусную схожесть между последовательными предложениями на основе эмбеддингов."""
        if text in self._cache_embedding:
            return self._cache_embedding[text]

        sentences = [s for s in sent_tokenize(text, language='russian') if s.strip()]
        if len(sentences) < 2:
            self._cache_embedding[text] = 0.0
            return 0.0
        try:
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
            sim_scores = [util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(embeddings)-1)]
            result = sum(sim_scores) / len(sim_scores)
            self._cache_embedding[text] = result
            return result
        except Exception as e:
            logger.error(f"Ошибка при расчете семантической когезии: {e}")
            self._cache_embedding[text] = 0.0
            return 0.0

    def calculate_perplexity(self, text: str, stride: int = None) -> float:
        """Вычисляет перплексию текста с использованием языковой модели для оценки когерентности."""
        if text in self._cache_perplexity:
            return self._cache_perplexity[text]

        try:
            if not text.strip():
                self._cache_perplexity[text] = float('inf')
                return float('inf')

            max_len = self.coherence_model.config.max_position_embeddings
            stride = stride or max_len // 2
            encodings = self.coherence_tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
            input_ids = encodings.input_ids[0].to(self.device)
            attn_mask = encodings.attention_mask[0].to(self.device)

            if input_ids.size(0) < 2:
                self._cache_perplexity[text] = float('inf')
                return float('inf')

            nlls, total_tokens = [], 0
            for i in range(0, input_ids.size(0), stride):
                begin = max(i + stride - max_len, 0)
                end = min(i + stride, input_ids.size(0))
                input_slice = input_ids[begin:end]
                mask_slice = attn_mask[begin:end]
                labels = input_slice.clone()
                labels[:i - begin] = -100

                if (labels != -100).sum().item() == 0:
                    continue

                with torch.no_grad():
                    outputs = self.coherence_model(input_slice.unsqueeze(0), attention_mask=mask_slice.unsqueeze(0), labels=labels.unsqueeze(0))
                    nlls.append(outputs.loss * (labels != -100).sum())
                    total_tokens += (labels != -100).sum().item()

            if total_tokens == 0:
                self._cache_perplexity[text] = float('inf')
                return float('inf')

            avg_nll = torch.stack(nlls).sum() / total_tokens
            result = torch.exp(avg_nll).item()
            self._cache_perplexity[text] = result
            return result
        except Exception as e:
            logger.error(f"Ошибка при расчете перплексии: {e}")
            self._cache_perplexity[text] = float('inf')
            return float('inf')

    def normalize_perplexity(self, ppl: float) -> float:
        """Нормализует значение перплексии в диапазоне [0,1] для дальнейшего использования."""
        if ppl == float('inf'):
            return 1.0
        ppl = min(max(ppl, MIN_PPL), MAX_PPL)
        log_ppl = np.log(ppl + 1e-9)
        log_min = np.log(MIN_PPL + 1e-9)
        log_max = np.log(MAX_PPL + 1e-9)
        return max(0.0, min(1.0, (log_ppl - log_min) / (log_max - log_min)))

    def calculate_tfidf_cohesion(self, text: str) -> float:
        """Вычисляет когезию текста на основе косинусной схожести TF-IDF векторов последовательных предложений."""
        if text in self._cache_tfidf:
            return self._cache_tfidf[text]

        sentences = [s for s in sent_tokenize(text, language='russian') if s.strip()]
        if len(sentences) < 2:
            self._cache_tfidf[text] = 0.0
            return 0.0
        try:
            vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
            tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
            sim_scores = [
                np.dot(tfidf_matrix[i], tfidf_matrix[i+1]) / (np.linalg.norm(tfidf_matrix[i]) * np.linalg.norm(tfidf_matrix[i+1]))
                for i in range(len(tfidf_matrix) - 1)
                if np.linalg.norm(tfidf_matrix[i]) > 1e-9 and np.linalg.norm(tfidf_matrix[i+1]) > 1e-9
            ]
            result = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
            self._cache_tfidf[text] = result
            return result
        except Exception as e:
            logger.error(f"Ошибка при расчете TF-IDF когезии: {e}")
            self._cache_tfidf[text] = 0.0
            return 0.0

    def calculate_entity_coherence(self, text: str, method: str = 'entropy') -> float:
        """Оценивает когерентность текста на основе распределения именованных сущностей (NER).

        Параметры:
        method — метод оценки: 'entropy' (энтропия) или 'repetition' (повторяемость).
        """
        cache_key = (text, method)
        if cache_key in self._cache_entity:
            return self._cache_entity[cache_key]

        if not text.strip():
            self._cache_entity[cache_key] = 0.0
            return 0.0
        try:
            entities = self.ner_pipeline(text)
            entity_texts = [e['word'].lower() for e in entities]
            if not entity_texts:
                self._cache_entity[cache_key] = 0.0
                return 0.0

            counter = Counter(entity_texts)
            total = sum(counter.values())

            if method == 'entropy':
                probs = np.array(list(counter.values())) / total
                ent = entropy(probs)
                max_ent = np.log(len(probs)) if len(probs) > 1 else 1.0
                result = max(0.0, min(1.0, 1.0 - (ent / max_ent)))
            elif method == 'repetition':
                repeats = sum(c - 1 for c in counter.values() if c > 1)
                result = repeats / total
            else:
                raise ValueError("Метод должен быть 'entropy' или 'repetition'")
            self._cache_entity[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Ошибка при расчете когерентности по NER: {e}")
            self._cache_entity[cache_key] = 0.0
            return 0.0

    def calculate_global_thematic_coherence(self, text: str, segment_type: str = 'paragraph') -> float:
        """Вычисляет тематическую когерентность текста через среднюю схожесть сегментов с общим текстом.

        Параметры:
        segment_type — тип сегментации: 'paragraph' (параграфы) или 'sentence' (предложения).
        """
        cache_key = (text, segment_type)
        if cache_key in self._cache_thematic:
            return self._cache_thematic[cache_key]

        if not text.strip():
            self._cache_thematic[cache_key] = 0.0
            return 0.0
        try:
            segments = text.split('\n\n') if segment_type == 'paragraph' else sent_tokenize(text, language='russian')
            segments = [s.strip() for s in segments if s.strip()]
            if not segments:
                self._cache_thematic[cache_key] = 0.0
                return 0.0

            text_emb = self.embedding_model.encode(text, convert_to_tensor=True)
            segment_embs = self.embedding_model.encode(segments, convert_to_tensor=True)
            sim_scores = [util.pytorch_cos_sim(seg.unsqueeze(0), text_emb.unsqueeze(0)).item() for seg in segment_embs]
            result = sum(sim_scores) / len(sim_scores)
            self._cache_thematic[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Ошибка при расчете тематической когерентности: {e}")
            self._cache_thematic[cache_key] = 0.0
            return 0.0

    def calculate_cohesion_score(self, text: str, tfidf_weight: float = 0.4, embedding_weight: float = 0.3, entity_weight: float = 0.3) -> float:
        """Комбинирует различные метрики когезии в общий скор с учетом весов каждого компонента."""
        tfidf = self.calculate_tfidf_cohesion(text)
        embedding = self.calculate_cohesion_embedding(text)
        entity = self.calculate_entity_coherence(text)
        total_weight = tfidf_weight + embedding_weight + entity_weight
        return float(round((tfidf_weight * tfidf + embedding_weight * embedding + entity_weight * entity) / total_weight, 4))

    def calculate_coherence_score(self, text: str, perplexity_weight: float = 0.5, thematic_weight: float = 0.5) -> float:
        """Вычисляет общий скор когерентности, объединяя перплексию и тематическую когерентность."""
        ppl = self.calculate_perplexity(text)
        coherence_from_ppl = 1.0 - self.normalize_perplexity(ppl)
        thematic = self.calculate_global_thematic_coherence(text)
        total_weight = perplexity_weight + thematic_weight
        return float(round((perplexity_weight * coherence_from_ppl + thematic_weight * thematic) / total_weight, 4))


class ConnectorsAnalyzer:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Выполняет нормализацию текста для повышения точности последующего анализа связок.
        Приводит текст к нижнему регистру и заменяет множественные пробелы и переносы строк одним пробелом.
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    @lru_cache(maxsize=128)
    def count_connectors(text: str, connectors_tuple: tuple) -> int:
        """
        Подсчитывает количество вхождений заданных связок в тексте с учётом экранирования и поддержки многословных выражений.
        """
        text = ConnectorsAnalyzer.preprocess_text(text)
        count = 0
        for connector in connectors_tuple:
            escaped_connector = re.escape(connector)
            pattern = rf'\b{escaped_connector}\b'
            found = re.findall(pattern, text)
            count += len(found)
        return count

    @staticmethod
    def analyze_connectors(text: str) -> Dict[str, Any]:
        """
        Анализирует текст на наличие позитивных и негативных связок, вычисляет их количество, плотности и соотношения.

        Args:
            text (str): Текст для анализа.

        Returns:
            Dict[str, Any]: Словарь с вычисленными метриками:
                - positive_connectors_count (int): Количество позитивных связок.
                - negative_connectors_count (int): Количество негативных связок.
                - positive_density_per_100_words (float): Плотность позитивных связок на 100 слов.
                - negative_density_per_100_words (float): Плотность негативных связок на 100 слов.
                - connectors_density_ratio_pos_to_neg (float): Отношение плотности позитивных к негативным связкам.
                - positive_ratio_of_total_connectors (float): Доля позитивных связок среди всех связок.
                - negative_ratio_of_total_connectors (float): Доля негативных связок среди всех связок.
                """
        text_clean = ConnectorsAnalyzer.preprocess_text(text)
        word_list = word_tokenize(text_clean, language='russian')
        total_words = len(word_list) if word_list else 1
        pos_count = ConnectorsAnalyzer.count_connectors(text_clean, tuple(POSITIVE_CONNECTORS))
        neg_count = ConnectorsAnalyzer.count_connectors(text_clean, tuple(NEGATIVE_CONNECTORS))

        pos_density = (pos_count / total_words) * 100
        neg_density = (neg_count / total_words) * 100

        density_ratio = pos_density / neg_density if neg_density > 0 else float('inf')

        total_connectors = pos_count + neg_count
        pos_ratio = pos_count / total_connectors if total_connectors > 0 else 0.0
        neg_ratio = neg_count / total_connectors if total_connectors > 0 else 0.0

        return {
            'positive_connectors_count': pos_count,
            'negative_connectors_count': neg_count,
            'positive_density_per_100_words': round(pos_density, 4),
            'negative_density_per_100_words': round(neg_density, 4),
            'connectors_density_ratio_pos_to_neg': round(density_ratio, 4),
            'positive_ratio_of_total_connectors': round(pos_ratio, 4),
            'negative_ratio_of_total_connectors': round(neg_ratio, 4),
        }

class CoreferenceResolver:
    """
    Класс для разрешения кореферентности в тексте и подсчёта совпадений между местоимениями и представителями.
    """

    def __init__(self) -> None:
        self.nlp = get_stanza_pipeline()

    def lemmatize(self, text: str) -> List[str]:
        """
        Лемматизация текста с использованием pymorphy3.
        """
        return [lemmatize_word(word) for word in text.split()]

    def resolve_and_score(self, text: str) -> float:
        """
        Разрешает кореферентные цепочки и вычисляет нормализованный счёт совпадений лемм местоимений и представителей.
        Возвращает абсолютное значение и нормализованное значение (от 0 до 1).
        """
        doc = self.nlp(text)
        matches = []

        for chain in doc.coref:
            rep_lemma = " ".join(self.lemmatize(chain.representative_text))

            for mention in chain.mentions:
                sentence = next((s for s in doc.sentences if s.index == mention.sentence), None)
                if sentence is None:
                    continue

                mention_lemmas = []
                for token in sentence.tokens:
                    for word in token.words:
                        if word.lemma in {'он', 'она', 'они'} and mention.start_word <= word.id <= mention.end_word:
                            mention_lemmas.append(word.lemma)

                if mention_lemmas:
                    matches.append((" ".join(mention_lemmas), rep_lemma))

        score = sum(1 for pron_lemma, rep_lemma in matches if pron_lemma == rep_lemma)
        total_sentences = len(doc.sentences)
        score_normalized = round(score / total_sentences, 2) if total_sentences > 0 else 0.0
        return score_normalized


class TextFeaturesExtractor:
    def __init__(self):
        self.cohesion_analyzer = TextCohesionAnalyzer()
        self.connectors_analyzer = ConnectorsAnalyzer()
        self.coref_resolver = CoreferenceResolver()

    def analyze_text_features(self, text: str) -> Dict[str, Any]:
        """
        Возвращает словарь со всеми фичами текста:
        - Метрики когезии (TF-IDF, эмбеддинги, NER)
        - Метрики когерентности (перплексия, тематическая когерентность)
        - Метрики по связкам (позитивные, негативные и их соотношения)
        Все количественные показатели нормализованы по длине текста.
        """
        features = {}

        words = word_tokenize(text)
        word_count = len(words)
        word_count = word_count if word_count > 0 else 1  # во избежание деления на 0

        # Когезия
        cohesion_score = self.cohesion_analyzer.calculate_cohesion_score(text)
        features['combined_cohesion_score'] = cohesion_score 

        # Когерентность
        coherence_score = self.cohesion_analyzer.calculate_coherence_score(text)
        features['combined_coherence_score'] = coherence_score 

        connectors_metrics = self.connectors_analyzer.analyze_connectors(text)
        features.update(connectors_metrics)
        
        # Разрешение кореферентности
        coref_score = self.coref_resolver.resolve_and_score(text)
        features['coreference_resolution_score'] = coref_score

        return features
