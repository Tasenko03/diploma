import re
from collections import Counter
import pymorphy3
import difflib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from functools import lru_cache

nltk.download('punkt')


class MorphologicalAnalyzer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Преобразует текст в нижний регистр, удаляет все символы, кроме русских букв и пробелов,
        затем разделяет текст на токены по пробелам.
        """
        text = text.lower()
        text = re.sub(r'[^а-яё\s]', '', text)
        return text.split()

    def analyze_pos(self, text: str) -> tuple[Counter, int]:
        """
        Выполняет морфологический разбор текста, подсчитывая количество частей речи.
        
        Возвращает:
            - Counter с частотами частей речи,
            - общее количество токенов.
        """
        tokens = self.tokenize(text)
        pos_counter = Counter()
        for token in tokens:
            parsed = self.morph.parse(token)
            if parsed:
                pos = parsed[0].tag.POS
                if pos:
                    pos_counter[pos] += 1
        return pos_counter, len(tokens)

    @lru_cache(maxsize=128)
    def analyze_pos_cached(self, text: str) -> tuple[Counter, int]:
        """
        Кэшированная версия анализа частей речи для повторного использования результатов
        при одинаковом тексте.
        """
        return self.analyze_pos(text)

    @staticmethod
    def morphological_complexity(pos_counter: Counter, total_words: int) -> float:
        """
        Вычисляет морфологическую сложность текста как отношение суммы глаголов,
        прилагательных и наречий к количеству существительных.
        """
        noun_count = pos_counter.get('NOUN', 0)
        verb_count = pos_counter.get('VERB', 0)
        adj_count = pos_counter.get('ADJF', 0) + pos_counter.get('ADJS', 0)
        adv_count = pos_counter.get('ADVB', 0)
        if noun_count == 0:
            return 0.0
        complexity_score = (verb_count + adj_count + adv_count) / noun_count
        return complexity_score


class LemmaPerTokenRatioCalculator:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Удаляет из текста все знаки препинания.
        """
        return re.sub(r'[^\w\s]', '', text)

    def calculate_lpr(self, text: str) -> float:
        """
        Рассчитывает отношение количества уникальных лемм к количеству токенов
        в тексте.
        """
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text, language='russian')
        lemmas = [self.morph.parse(token)[0].normal_form for token in tokens]
        num_tokens = len(tokens)
        num_lemmas = len(set(lemmas))
        if num_tokens == 0:
            return 0.0
        return num_lemmas / num_tokens

    @lru_cache(maxsize=128)
    def calculate_lpr_cached(self, text: str) -> float:
        """
        Кэшированная версия расчёта лемм на токен для повторного использования результатов.
        """
        return self.calculate_lpr(text)


class GrammarCorrector:
    def __init__(self, model_path: str):
        """
        Инициализация модели для грамматической коррекции с загрузкой токенизатора и модели.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """
        Удаляет знаки препинания из текста.
        """
        return re.sub(r"[^\w\s]", "", text)

    @staticmethod
    def get_token_char_spans(text: str, tokens: list[str]) -> list[tuple[int | None, int | None]]:
        """
        Определяет позиции начала и конца каждого токена в исходном тексте.
        
        Возвращает список кортежей (start, end) для каждого токена.
        """
        spans = []
        start = 0
        for token in tokens:
            start = text.find(token, start)
            if start == -1:
                spans.append((None, None))
                continue
            end = start + len(token)
            spans.append((start, end))
            start = end
        return spans

    @staticmethod
    def get_diff_spans(orig_tokens: list[str], corrected_tokens: list[str]) -> list[dict]:
        """
        Определяет разницу между списками токенов исходного и исправленного текста.
        
        Возвращает список изменений с типом операции и позициями в токенах.
        """
        matcher = difflib.SequenceMatcher(a=orig_tokens, b=corrected_tokens)
        spans = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                spans.append({
                    "type": tag,
                    "orig_span": (i1, i2),
                    "corr_span": (j1, j2),
                    "orig_tokens": orig_tokens[i1:i2],
                    "corr_tokens": corrected_tokens[j1:j2]
                })
        return spans

    def correct_text(self, text: str) -> dict:
        """
        Выполняет грамматическую коррекцию текста по предложениям, выявляет изменения
        и подсчитывает количество ошибок и нормализованное количество ошибок на токены.
        
        Возвращает словарь с общей статистикой и списком исправлений.
        """
        sentences = sent_tokenize(text)
        total_errors = 0
        corrections = []

        for sent_index, sentence in enumerate(sentences, 1):
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            orig_tokens = self.remove_punctuation(sentence).split()
            corr_tokens = self.remove_punctuation(corrected_text).split()

            orig_spans = self.get_token_char_spans(sentence, orig_tokens)

            diff_spans = self.get_diff_spans(orig_tokens, corr_tokens)

            for span in diff_spans:
                i1, i2 = span["orig_span"]
                orig_start = orig_spans[i1][0] if i1 < len(orig_spans) else None
                orig_end = orig_spans[i2 - 1][1] if (i2 - 1) < len(orig_spans) else None
                corrections.append({
                    "sent_index": sent_index,
                    "type": span["type"],
                    "orig_span": [orig_start, orig_end],
                    "change": f"{span['orig_tokens']} -> {span['corr_tokens']}"
                })
                total_errors += 1

        total_tokens = sum(len(word_tokenize(sentence, language='russian')) for sentence in sentences)
        normalized_errors = round(((total_errors / total_tokens) * 100), 2) if total_tokens > 0 else 0.0
        
        return {
            "total_errors": total_errors,
            "normalized_errors": normalized_errors,
            "corrections": corrections
        }

    @lru_cache(maxsize=64)
    def correct_text_cached(self, text: str) -> dict:
        """
        Кэшированная версия грамматической коррекции для повторного использования результатов.
        """
        return self.correct_text(text)


class TextAnalyzer:
    def __init__(self, grammar_model_path: str):
        """
        Интегрирует морфологический анализ, расчет отношения лемм к токенам и
        грамматическую коррекцию текста.
        """
        self.morph_analyzer = MorphologicalAnalyzer()
        self.lpr_calculator = LemmaPerTokenRatioCalculator()
        self.grammar_corrector = GrammarCorrector(grammar_model_path)

    def analyze(self, text: str) -> dict:
        """
        Выполняет комплексный анализ текста, включая морфологическую сложность,
        отношение лемм к токенам и грамматическую коррекцию.
        
        Возвращает словарь с результатами всех трёх методов.
        """
        pos_counter, total_words = self.morph_analyzer.analyze_pos_cached(text)
        complexity = self.morph_analyzer.morphological_complexity(pos_counter, total_words)
        lpr = self.lpr_calculator.calculate_lpr_cached(text)
        corrections = self.grammar_corrector.correct_text_cached(text)

        result =  {
            "morphological_complexity": round(complexity, 2),
            "lemma_per_token_ratio": round(lpr, 2)
        }

        result.update(corrections)
        return result
