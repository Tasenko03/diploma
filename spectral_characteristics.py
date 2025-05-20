import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call


class VoiceAnalyzer:
    def __init__(self, file_path):
        """
        Инициализация анализатора речи. Загружает и нормализует аудиофайл.

        :param file_path: Путь к аудиофайлу (.wav)
        """
        self.file_path = file_path
        self.sound = parselmouth.Sound(file_path)  # Анализ через Praat
        self.y, self.sr = librosa.load(file_path, sr=None)
        self.y = librosa.util.normalize(self.y)  # Нормализация амплитуды сигнала

    def analyze_pitch_jitter_shimmer_volume(self):
        """
        Анализ основных речевых характеристик: pitch, jitter, shimmer и громкость.

        :return: Словарь со статистиками pitch, jitter, shimmer и volume
        """
        pitch = self.sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        mean_pitch = float(round(np.mean(pitch_values), 2))
        max_pitch = float(round(np.max(pitch_values), 2))
        min_pitch = float(round(np.min(pitch_values), 2))
        std_pitch = float(round(np.std(pitch_values), 2))
        pitch_variability = float(round(np.max(pitch_values) - np.min(pitch_values), 2))

        point_process = call(self.sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = float(round(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 2))
        shimmer = float(round(call([self.sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 2))

        intensity = self.sound.to_intensity()
        intensity_values = intensity.values[0]

        mean_volume = float(round(np.mean(intensity_values), 2))
        max_volume = float(round(np.max(intensity_values), 2))
        min_volume = float(round(np.min(intensity_values), 2))
        std_volume = float(round(np.std(intensity_values), 2))
        volume_variability = float(round(max_volume - min_volume, 2))

        return {
            "mean_pitch": mean_pitch,
            "max_pitch": max_pitch,
            "min_pitch": min_pitch,
            "std_pitch": std_pitch,
            "pitch_variability": pitch_variability,
            "jitter": jitter,
            "shimmer": shimmer,
            "mean_volume": mean_volume,
            "max_volume": max_volume,
            "min_volume": min_volume,
            "std_volume": std_volume,
            "volume_variability": volume_variability
        }

    def calculate_mfcc(self, n_mfcc=13):
        """
        Расчёт MFCC (мел-кепстральных коэффициентов).

        :param n_mfcc: Количество MFCC коэффициентов
        :return: Список средних значений MFCC
        """
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc)
        return [float(val) for val in np.mean(mfccs.T, axis=0)]

    def calculate_lpc(self, order=12):
        """
        Расчёт коэффициентов линейного предсказания (LPC).

        :param order: Порядок LPC
        :return: Список средних LPC коэффициентов (без первого)
        """
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.010 * self.sr)
        frames = librosa.util.frame(self.y, frame_length=frame_length, hop_length=hop_length)
        lpc_coeffs_list = [librosa.lpc(frame, order=order)[1:] for frame in frames.T]
        return [float(val) for val in np.mean(np.array(lpc_coeffs_list), axis=0)] if lpc_coeffs_list else None

    def analyze_formants(self, time_step=0.01, max_number_of_formants=5, max_formant_freq=5500, window_length=0.025):
        """
        Анализ формант и их полос пропускания.

        :param time_step: Временной шаг
        :param max_number_of_formants: Максимум формант
        :param max_formant_freq: Верхняя граница частоты
        :param window_length: Длина окна анализа
        :return: Словарь со статистиками F1–F3 и соответствующих ширин полос
        """
        formant_object = call(self.sound, "To Formant (burg)", time_step, max_number_of_formants, max_formant_freq, window_length, 50)
        f1_values, f2_values, f3_values = [], [], []
        bw1_values, bw2_values, bw3_values = [], [], []

        for t in formant_object.ts():
            f1 = call(formant_object, "Get value at time", 1, t, 'hertz', 'Linear')
            f2 = call(formant_object, "Get value at time", 2, t, 'hertz', 'Linear')
            f3 = call(formant_object, "Get value at time", 3, t, 'hertz', 'Linear')
            bw1 = call(formant_object, "Get bandwidth at time", 1, t, 'hertz', 'Linear')
            bw2 = call(formant_object, "Get bandwidth at time", 2, t, 'hertz', 'Linear')
            bw3 = call(formant_object, "Get bandwidth at time", 3, t, 'hertz', 'Linear')

            if not np.isnan(f1): f1_values.append(f1)
            if not np.isnan(f2): f2_values.append(f2)
            if not np.isnan(f3): f3_values.append(f3)
            if not np.isnan(bw1): bw1_values.append(bw1)
            if not np.isnan(bw2): bw2_values.append(bw2)
            if not np.isnan(bw3): bw3_values.append(bw3)

        if not f1_values or not f2_values or not f3_values:
            return None

        return {
            "f1_mean": float(round(np.mean(f1_values), 2)),
            "f2_mean": float(round(np.mean(f2_values), 2)),
            "f3_mean": float(round(np.mean(f3_values), 2)),
            "bw1_mean": float(round(np.mean(bw1_values), 2)),
            "bw2_mean": float(round(np.mean(bw2_values), 2)),
            "bw3_mean": float(round(np.mean(bw3_values), 2)),
            "f1_std": float(round(np.std(f1_values), 2)),
            "f2_std": float(round(np.std(f2_values), 2)),
            "f3_std": float(round(np.std(f3_values), 2)),
            "bw1_std": float(round(np.std(bw1_values), 2)),
            "bw2_std": float(round(np.std(bw2_values), 2)),
            "bw3_std": float(round(np.std(bw3_values), 2)),
        }

    def extract_all_features(self):
        """
        Извлекает полный набор признаков: pitch, jitter, shimmer, громкость, MFCC, LPC, форманты.

        :return: Словарь с результатами всех анализов
        """
        features = {}
        pitch_volume = self.analyze_pitch_jitter_shimmer_volume()
        features.update(pitch_volume)
        features["mfcc"] = self.calculate_mfcc()
        features["lpc"] = self.calculate_lpc()
        formants = self.analyze_formants()
        features.update(formants)
        return features
