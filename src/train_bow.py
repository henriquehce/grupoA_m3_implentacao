"""
Modelo FIEL ao artigo (Salloum et al., 2024).

Bag-of-words + rede neural densa em TensorFlow/Keras, exatamente como descrito:
    Dense(128, relu) -> Dropout(0.5) -> Dense(64, relu) -> Dropout(0.5)
    -> Dense(n_classes, softmax)
    Otimizador: Adam | Perda: categorical_crossentropy
    Epocas: 200 | Batch size: 5

Uso:
    python src/train_bow.py
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia logs do TF

import numpy as np

from preprocess import build_training_data, load_intents

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "intents_univali.json"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def build_model(input_dim: int, output_dim: int):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Dense(128, input_shape=(input_dim,), activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(output_dim, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main(epochs: int = 200, batch_size: int = 5):
    intents = load_intents(DATA)
    X, y, words, classes = build_training_data(intents)

    print(f"Treino: X={X.shape}, y={y.shape}  | {len(classes)} intencoes")

    model = build_model(X.shape[1], y.shape[1])
    model.summary()

    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Salva modelo e artefatos do pre-processamento
    model.save(MODELS / "model_bow.keras")
    with open(MODELS / "bow_artifacts.pkl", "wb") as f:
        pickle.dump({"words": words, "classes": classes}, f)
    # historico para os graficos de acuracia/perda (Fig. 2 e 3 do artigo)
    with open(MODELS / "bow_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    final_acc = history.history["accuracy"][-1]
    print(f"\nModelo salvo em {MODELS / 'model_bow.keras'}")
    print(f"Acuracia final de treino: {final_acc:.4f}")


if __name__ == "__main__":
    main()
