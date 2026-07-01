"""
Pre-processamento de texto para o chatbot de FAQ (UNIVALI).

Replica o pipeline do artigo (Salloum et al., 2024):
    Tokenizacao -> Lematizacao -> Vetorizacao (bag-of-words)

Diferenca para o artigo: o paper usa NLTK + WordNetLemmatizer (so ingles).
Como nosso dataset e em portugues, usamos o spaCy (pt_core_news_sm) para a
lematizacao. Ha um fallback simples caso o modelo do spaCy nao esteja instalado.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# --- Carregamento opcional do spaCy (lematizador PT) ---------------------------
_NLP = None
_SPACY_OK = False
try:
    import spacy

    try:
        _NLP = spacy.load("pt_core_news_sm", disable=["ner", "parser"])
        _SPACY_OK = True
    except OSError:
        # Modelo ainda nao baixado: python -m spacy download pt_core_news_sm
        _SPACY_OK = False
except ImportError:
    _SPACY_OK = False


# Tokens a ignorar na construcao do vocabulario (pontuacao)
_IGNORE = {"?", "!", ".", ",", ";", ":", "'", '"', "(", ")", "-"}


def strip_accents(text: str) -> str:
    """Remove acentos para normalizar o vocabulario (cafe == café)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(sentence: str) -> list[str]:
    """Divide a frase em tokens minusculos (etapa 1 do pipeline)."""
    sentence = sentence.lower()
    # mantem letras (com acento), numeros; separa o resto
    return re.findall(r"[a-zà-ÿ0-9]+", sentence)


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Reduz cada token a sua forma base (etapa 2 do pipeline)."""
    if _SPACY_OK and _NLP is not None:
        doc = _NLP(" ".join(tokens))
        lemmas = [t.lemma_.lower() for t in doc if t.lemma_.strip()]
    else:
        # Fallback bem simples: usa o proprio token (sem lematizar de fato)
        lemmas = tokens
    # normaliza acentos e remove pontuacao residual
    return [strip_accents(w) for w in lemmas if w not in _IGNORE]


def clean(sentence: str) -> list[str]:
    """Pipeline completo de uma frase: tokeniza + lematiza + normaliza."""
    return lemmatize_tokens(tokenize(sentence))


def load_intents(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(intents: dict):
    """
    Percorre o dataset e monta:
      - words:    vocabulario lematizado e ordenado (features do bag-of-words)
      - classes:  lista ordenada de tags (intencoes)
      - documents: pares (tokens_lematizados, tag)
    """
    words: set[str] = set()
    classes: list[str] = []
    documents: list[tuple[list[str], str]] = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag not in classes:
            classes.append(tag)
        for pattern in intent["patterns"]:
            tokens = clean(pattern)
            words.update(tokens)
            documents.append((tokens, tag))

    words_sorted = sorted(words)
    classes_sorted = sorted(classes)
    return words_sorted, classes_sorted, documents


def bag_of_words(sentence: str, words: list[str]) -> list[int]:
    """
    Vetoriza uma frase (etapa 3): vetor do tamanho do vocabulario,
    com 1 onde a palavra aparece, 0 caso contrario.
    """
    sentence_words = set(clean(sentence))
    return [1 if w in sentence_words else 0 for w in words]


def build_training_data(intents: dict):
    """
    Gera as matrizes X (bag-of-words) e y (one-hot da intencao) para o treino.
    Retorna: X, y, words, classes
    """
    import numpy as np

    words, classes, documents = build_corpus(intents)

    X, y = [], []
    for tokens, tag in documents:
        sentence = " ".join(tokens)
        X.append([1 if w in set(tokens) else 0 for w in words])
        row = [0] * len(classes)
        row[classes.index(tag)] = 1
        y.append(row)

    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), words, classes


if __name__ == "__main__":
    # Teste rapido do pipeline
    intents = load_intents(Path(__file__).resolve().parents[1] / "data" / "intents_univali.json")
    X, y, words, classes = build_training_data(intents)
    print(f"spaCy disponivel: {_SPACY_OK}")
    print(f"Vocabulario: {len(words)} palavras")
    print(f"Intencoes:   {len(classes)} -> {classes}")
    print(f"Exemplos de treino: X={X.shape}, y={y.shape}")
    print(f"Tokens de 'quais cursos a univali tem?': {clean('quais cursos a univali tem?')}")
