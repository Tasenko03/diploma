import os
import sys
import argparse
import gdown
import pandas as pd
import torch
from verbal_fluency import SpeechFluencyAnalyzer
from spectral_characteristics import VoiceAnalyzer
from get_transcription import AudioTranscriptionProcessor
from lexical_features import analyze_text_lex
from syntactical_features import analyze_text_synt
from discursive_features import TextFeaturesExtractor
from morphological_features import TextAnalyzer
import gdown

# Ссылка на папку на Google Drive
folder_url = "https://drive.google.com/drive/folders/1--q1HXhRG69d1Dlv-j6VJmSyRG4mwgYJ"

# Папка, в которую будет загружено содержимое
output_dir = "./models/RuT5_GEC"

# Загрузка всей папки
gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)

# Инициализация морфологического анализатора
morph_model_path = './models/RuT5_GEC'
morph_analyzer = TextAnalyzer(morph_model_path)

def process_file(filepath: str, group_label: str) -> dict | None:
    """
    Обрабатывает аудиофайл, извлекая фонетические, спектральные, транскрипционные,
    лексические, синтаксические, дискурсивные и морфологические признаки.

    Args:
        filepath (str): Путь к аудиофайлу.
        group_label (str): Метка группы (например, 'patient' или 'control').

    Returns:
        dict | None: Словарь с объединёнными признаками и метками или None при ошибке.
    """
    try:
        # Фонетика
        fluency = SpeechFluencyAnalyzer().analyze(filepath)
        spectral = VoiceAnalyzer(filepath).extract_all_features()

        # Транскрипция
        transcription = AudioTranscriptionProcessor(filepath)
        text = transcription.clean_transcript()

        # Текстовые признаки
        lex = analyze_text_lex(text)
        synt = analyze_text_synt(text)
        disc = TextFeaturesExtractor().analyze_text_features(text)
        morph = morph_analyzer.analyze(text)

        # Объединяем всё в одну строку
        features = {
            'filepath': filepath,
            'group': group_label,
            **fluency,
            **spectral,
            **disc,
            **lex,
            **synt,
            **morph
        }
        return features
    except Exception as e:
        print(f"Ошибка при обработке файла {filepath}: {e}")
        return None


def process_directory(base_dir: str, output_path: str) -> None:
    """
    Обрабатывает все аудиофайлы в указанной директории, формируя и сохраняя
    объединённый датасет с признаками в CSV-файл.

    Args:
        base_dir (str): Путь к директории с аудиофайлами.

    Returns:
        None
    """
    all_data = []
    for fname in os.listdir(base_dir):
        if fname.endswith('.wav'):
            full_path = os.path.join(base_dir, fname)
            if fname.startswith('PD'):
                group = 'patient'
            elif fname.startswith('PN'):
                group = 'control'
            else:
                print(f"Неизвестная группа для файла {fname}, пропуск...")
                continue
            data = process_file(full_path, group)
            if data:
                all_data.append(data)
            torch.cuda.empty_cache() 

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    print("Готово. Датасет сохранён в", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка аудиофайлов с извлечением признаков.")
    parser.add_argument('audio_folder', type=str, help='Путь к директории с аудиофайлами')
    parser.add_argument('--output', '-o', type=str, default='full_dataset.csv',
                        help='Путь для сохранения итогового CSV (по умолчанию full_dataset.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.audio_folder):
        print(f"Ошибка: директория '{args.audio_folder}' не существует.")
        sys.exit(1)

    process_directory(args.audio_folder, args.output)
