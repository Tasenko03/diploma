import os
import whisper
import re

ffmpeg_dir = 'ffmpeg-7.0.2-amd64-static'
os.environ["PATH"] += os.pathsep + ffmpeg_dir

_model_cache = {}

class AudioTranscriptionProcessor:
    def __init__(self, audio_path: str):
        global _model_cache
        if "large-v3" not in _model_cache:
            _model_cache["large-v3"] = whisper.load_model("large-v3", device="cuda")
        self.model = _model_cache["large-v3"]
        self.audio_path = audio_path

    def _transcribe(self) -> str:
        """
        Выполняет транскрипцию аудиофайла.

        :return: Расшифрованный текст.
        :raises FileNotFoundError: Если файл не найден по указанному пути.
        """
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Файл {self.audio_path} не найден.")

        result: dict = self.model.transcribe(self.audio_path)
        return result["text"].strip()

    def clean_transcript(self) -> str:
        """
        Очищает транскрибированный текст от артефактов и приводит его к читаемому виду.

        :return: Очищенный текст.
        """
        transcript = self._transcribe()
        transcript = re.sub(r'\.{2,}', '.', transcript)
        transcript = re.sub(r'\s+', ' ', transcript)
        transcript = transcript.strip()
        transcript = re.sub(r'[«»“”]', '', transcript)
        return transcript
