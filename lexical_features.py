import re
import requests
from collections import Counter
from typing import List, Tuple, Dict
from functools import lru_cache

import pymorphy3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lexical_diversity import lex_div as ld
from nlp_pipeline import get_stanza_pipeline

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Глобальный объект морфоанализа
MORPH = pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=None)
def lemmatize_word(word: str) -> str:
    """
    Лемматизирует слово с использованием pymorphy3 и кэширует результат.
    """
    return MORPH.parse(word)[0].normal_form


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str):
    """
    Загружает и кэширует токенизатор HuggingFace.
    """
    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=None)
def load_model(model_name: str):
    """
    Загружает и кэширует модель HuggingFace.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name)


class NeologismCounter:
    """
    Класс для подсчёта потенциальных неологизмов и проверки их наличия в Викисловаре.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.neologisms: Counter[str] = Counter()
        self._wiktionary_cache: Dict[str, bool] = {}

    @staticmethod
    def _preprocess(text: str) -> List[str]:
        """
        Нормализация текста: приведение к нижнему регистру, удаление знаков препинания и токенизация.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        return [token for token in text.split() if token]

    def count_neologisms(self) -> int:
        """
        Подсчитывает потенциальные неологизмы — слова длиной более 2 символов, отсутствующие в словаре Pymorphy.
        """
        tokens = self._preprocess(self.text)
        self.neologisms.clear()
        for token in tokens:
            if len(token) > 2 and not MORPH.word_is_known(token):
                self.neologisms[token] += 1
        return sum(self.neologisms.values())

    def _is_in_wiktionary(self, word: str) -> bool:
        """
        Проверка слова на наличие в Викисловаре с использованием кэша.
        """
        normal_form = lemmatize_word(word)
        if normal_form in self._wiktionary_cache:
            return self._wiktionary_cache[normal_form]

        url = "https://ru.wiktionary.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": normal_form
        }

        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            exists = next(iter(pages)) != "-1"
        except (requests.RequestException, ValueError):
            exists = False  # В случае ошибки считаем слово отсутствующим

        self._wiktionary_cache[normal_form] = exists
        return exists

    def get_potential_neologisms(self) -> List[Tuple[str, int]]:
        """
        Возвращает список потенциальных неологизмов с частотами.
        """
        if not self.neologisms:
            self.count_neologisms()
        return self.neologisms.most_common()

    def get_confirmed_neologisms(self) -> List[str]:
        """
        Возвращает список неологизмов, отсутствующих в Викисловаре.
        """
        if not self.neologisms:
            self.count_neologisms()
        return [word for word in self.neologisms if not self._is_in_wiktionary(word)]

    def neologisms_per_100_words(self) -> float:
        """
        Возвращает количество подтверждённых неологизмов на 100 слов текста.
        """
        tokens = self._preprocess(self.text)
        if not tokens:
            return 0.0

        if not self.neologisms:
            self.count_neologisms()

        confirmed_count = sum(1 for word in self.neologisms if not self._is_in_wiktionary(word))
        return round((confirmed_count / len(tokens)) * 100, 2)


def calculate_mtld(text: str) -> float:
    """
    Вычисляет показатель MTLD (Measure of Textual Lexical Diversity) для заданного текста.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    tokens = [token for token in text.split() if token]
    return round(ld.mtld(tokens), 2)


class RussianSentimentAnalyzer:
    """
    Класс для анализа тональности русского текста с использованием предобученной модели.
    """

    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-sentiment-rusentiment") -> None:
        """
        Инициализация анализатора с загрузкой модели и токенизатора.
        """
        self.tokenizer = load_tokenizer(model_name)
        self.model = load_model(model_name)
        self.nlp = get_stanza_pipeline()

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения с помощью Stanza.
        """
        doc = self.nlp(text)
        return [sentence.text for sentence in doc.sentences if sentence.text]

    def analyze_sentiment(self, sentence: str) -> int:
        """
        Анализ тональности предложения.
        Возвращает: 0 — нейтральная, 1 — положительная, 2 — негативная.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
        return predicted

    def process_text(self, text: str) -> Dict[str, float]:
        """
        Анализ тональности всего текста с вычислением абсолютных и относительных показателей.
        """
        sentences = self.split_into_sentences(text)
        counts = {"positive": 0, "neutral": 0, "negative": 0}

        for sentence in sentences:
            idx = self.analyze_sentiment(sentence)
            if idx == 0:
                counts["neutral"] += 1
            elif idx == 1:
                counts["positive"] += 1
            elif idx == 2:
                counts["negative"] += 1

        total = len(sentences)
        return {
            "positive_ratio": round(counts["positive"] / total, 2) if total else 0.0,
            "negative_ratio": round(counts["negative"] / total, 2) if total else 0.0
        }


def analyze_text_lex(text: str) -> dict:
    """
    Выполняет комплексный лингвистический анализ текста, включая подсчёт неологизмов, 
    вычисление показателя связности MTLD, анализ тональности и разрешение кореферентности.
    """
    counter = NeologismCounter(text)
    counter.count_neologisms()
    confirmed_neologisms = counter.get_confirmed_neologisms()
    neologisms_rate = counter.neologisms_per_100_words()

    mtld_score = calculate_mtld(text)

    sentiment_analyzer = RussianSentimentAnalyzer()
    sentiment_result = sentiment_analyzer.process_text(text)
    
    result = {
        "confirmed_neologisms_per_100_words": neologisms_rate,
        "confirmed_neologisms": confirmed_neologisms,
        "mtld": mtld_score
    }
    result.update(sentiment_result)
    return result
