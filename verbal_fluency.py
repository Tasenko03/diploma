from faster_whisper import WhisperModel
import re
from typing import List, Optional, Dict

# Глобальное хранилище кэшированных моделей
_model_cache: Dict[str, WhisperModel] = {}

def get_cached_model(model_size: str = "medium", device: str = "cuda") -> WhisperModel:
    """
    Возвращает экземпляр модели Whisper, используя кэширование.
    Если модель уже была загружена, возвращает её из кэша.

    :param model_size: Размер модели (например, "base", "medium", "large").
    :param device: Устройство ("cuda" или "cpu").
    :return: Экземпляр WhisperModel.
    """
    cache_key = f"{model_size}_{device}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = WhisperModel(model_size, device=device)
    return _model_cache[cache_key]

class SpeechFluencyAnalyzer:
    def __init__(self, model_size: str = "medium", device: str = "cuda", filled_pause_list: Optional[List[str]] = None):
        """
        Инициализация анализатора беглости речи с кэшированием модели.

        :param model_size: Размер модели Whisper.
        :param device: Устройство выполнения модели.
        :param filled_pause_list: Список заполненных пауз.
        """
        self.model = get_cached_model(model_size, device)
        self.filled_pause_list = filled_pause_list or ['эээ', 'эм', 'э-э', 'ммм', 'м-м', 'ааа', 'ну', 'блин', 'короче']

    def analyze(self, audio_path: str) -> Dict:
        """
        Выполняет транскрипцию аудиофайла и анализирует беглость речи.

        :param audio_path: Путь к аудиофайлу.
        :return: Словарь с рассчитанными метриками беглости речи.
        """
        segments, _ = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt="""Эээ... э-э... эм... ну... короче... блин... 
            эээ... ммм... м-м... ааа... эээ... эм... ммм... ааа..."""
        )

        self.metrics = {
            'total_words': 0,
            'total_duration': 0.0,
            'unfilled_pauses': [],
            'filled_pauses_count': 0,
            'filled_pauses_duration': 0.0,
            'first_word_start': None,
            'last_word_end': 0.0,
            'previous_word_end': None
        }

        for segment in segments:
            for word in segment.words:
                self._process_word(word)

        return self._calculate_metrics()

    def _process_word(self, word):
        """
        Обрабатывает отдельное слово из транскрипции, обновляя метрики речи.

        :param word: Объект слова с таймкодами начала и конца.
        """
        start = word.start
        end = word.end
        duration = end - start
        word_text = re.sub(r'[^\w\s]', '', word.word.strip().lower())

        # Учет заполненных пауз
        if word_text in self.filled_pause_list:
            self.metrics['filled_pauses_count'] += 1
            self.metrics['filled_pauses_duration'] += duration

        # Учет слов
        self.metrics['total_words'] += 1
        self.metrics['total_duration'] += duration

        # Учет начала и конца речи
        if self.metrics['first_word_start'] is None:
            self.metrics['first_word_start'] = start
        self.metrics['last_word_end'] = end

        # Учет незаполненных пауз
        if self.metrics['previous_word_end'] is not None and start > self.metrics['previous_word_end']:
            pause = start - self.metrics['previous_word_end']
            self.metrics['unfilled_pauses'].append(pause)

        self.metrics['previous_word_end'] = end

    def _calculate_metrics(self) -> Dict:
        """
        Вычисляет итоговые метрики беглости речи на основе собранных данных.

        :return: Словарь с метриками, такими как скорость речи, длительности и количество пауз.
        """
        first = self.metrics['first_word_start']
        last = self.metrics['last_word_end']
        total_audio_duration = (last - first) if first is not None else 0

        total_words = self.metrics['total_words']
        total_duration = self.metrics['total_duration']
        filled_count = self.metrics['filled_pauses_count']
        filled_duration = self.metrics['filled_pauses_duration']
        unfilled_pauses = self.metrics['unfilled_pauses']
        unfilled_total = sum(unfilled_pauses)
        unfilled_count = len(unfilled_pauses)

        speaking_rate = total_words / (total_audio_duration / 60) if total_audio_duration > 0 else 0
        articulation_rate = total_words / (total_duration / 60) if total_duration > 0 else 0

        words_per_minute = speaking_rate 

        filled_pauses_per_100_words = (filled_count / total_words * 100) if total_words else 0
        unfilled_pauses_per_100_words = (unfilled_count / total_words * 100) if total_words else 0

        filled_pauses_per_minute = (filled_count / (total_audio_duration / 60)) if total_audio_duration > 0 else 0
        unfilled_pauses_per_minute = (unfilled_count / (total_audio_duration / 60)) if total_audio_duration > 0 else 0

        return {
            'speaking_rate': float(round(speaking_rate, 2)),
            'articulation_rate': float(round(articulation_rate, 2)),
            'words_per_minute': float(round(words_per_minute, 2)),
            'unfilled_pauses_count': unfilled_count,
            'unfilled_pauses_total_duration': float(round(unfilled_total, 2)),
            'unfilled_pauses_average_duration': float(round(unfilled_total / unfilled_count, 2)) if unfilled_count else 0.0,
            'unfilled_pauses_per_100_words': float(round(unfilled_pauses_per_100_words, 2)),
            'unfilled_pauses_per_minute': float(round(unfilled_pauses_per_minute, 2)),
            'filled_pauses_count': filled_count,
            'filled_pauses_total_duration': float(round(filled_duration, 2)),
            'filled_pauses_average_duration': float(round(filled_duration / filled_count, 2)) if filled_count else 0.0,
            'filled_pauses_per_100_words': float(round(filled_pauses_per_100_words, 2)),
            'filled_pauses_per_minute': float(round(filled_pauses_per_minute, 2)),
            'pause_proportions_unfilled': float(round(unfilled_total / total_audio_duration, 4)) if total_audio_duration else 0.0,
            'pause_proportions_filled': float(round(filled_duration / total_audio_duration, 4)) if total_audio_duration else 0.0
        }
