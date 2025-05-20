import pandas as pd
import joblib
import argparse
import numpy as np
from feat_extraction_final_pipline import process_file

MODEL_PATH = "models/shico_model/best_stacking_model.joblib"  

class AudioPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, audio_path: str):
        features = process_file(audio_path, group_label='unknown')
        if features is None:
            print(f"Не удалось извлечь признаки из файла {audio_path}")
            return None

        features_df = pd.DataFrame([features])
        features_df = features_df.drop(columns=['filepath', 'group'], errors='ignore')

        with open("models/shico_model/significant_vars.txt", 'r') as f:
            significant_vars = [line.strip() for line in f]
            
        # Отбираем только нужные признаки в нужном порядке
        features_df = features_df[significant_vars]

        features_df.replace([np.inf, -np.inf], np.nan, inplace=True) 
        
        y_pred = self.model.predict(features_df)
        y_proba = self.model.predict_proba(features_df)[:, 1]

        print(f"Файл: {audio_path}")
        print(f"Предсказанный класс: {y_pred[0]}")
        print(f"Вероятность положительного класса: {y_proba[0]:.4f}")

        return y_pred[0], y_proba[0]


def main():
    parser = argparse.ArgumentParser(description="Скрипт для предсказания класса аудиофайла.")
    parser.add_argument("audio_path", type=str, help="Путь к аудиофайлу")

    args = parser.parse_args()

    predictor = AudioPredictor()
    predictor.predict(args.audio_path)


if __name__ == "__main__":
    main()
