"""
Modelo MODERNO (melhoria sobre o artigo).

Em vez de bag-of-words (que ignora ordem e significado), usamos embeddings
semanticos multilingues com sentence-transformers. Cada 'pattern' do dataset
vira um vetor denso; na inferencia comparamos a pergunta do usuario por
similaridade de cosseno e escolhemos a intencao mais proxima.

Vantagem: entende parafrases que NAO estao no dataset
(ex.: "quanto vou pagar por mes" ~ intent 'mensalidade_valores').

Uso:
    python src/train_embeddings.py
"""

from __future__ import annotations

import os

# Forca o backend PyTorch na transformers (evita conflito com Keras 3 do TF)
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pickle
from pathlib import Path

import numpy as np

from preprocess import load_intents

ROOT = Path(__file__).resolve().parents[1]
# Modelo moderno usa a base de FAQ REAL da UNIVALI (respostas especificas).
DATA = ROOT / "data" / "faq_univali.json"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# Modelo multilingue, leve e bom para PT-BR
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def main():
    intents = load_intents(DATA)
    encoder = get_encoder()

    patterns: list[str] = []
    labels: list[str] = []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            labels.append(intent["tag"])

    print(f"Codificando {len(patterns)} patterns com {MODEL_NAME} ...")
    embeddings = encoder.encode(patterns, normalize_embeddings=True, show_progress_bar=False)

    with open(MODELS / "embeddings_index.pkl", "wb") as f:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "patterns": patterns,
                "labels": labels,
                "embeddings": np.asarray(embeddings, dtype="float32"),
            },
            f,
        )

    print(f"Indice salvo em {MODELS / 'embeddings_index.pkl'}")
    print(f"Shape dos embeddings: {np.asarray(embeddings).shape}")


if __name__ == "__main__":
    main()
