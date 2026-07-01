"""
Avaliacao dos modelos — reproduz as analises do artigo (Figs. 1-4) e compara
o modelo fiel (BoW) com o moderno (embeddings).

Gera, em models/figs/:
  - confusion_matrix_bow.png        (Fig. 1 do artigo)
  - accuracy_loss_bow.png           (Fig. 2 do artigo)
  - roc_bow.png                     (Fig. 4 do artigo)
  - comparacao_robustez.png         (BoW vs embeddings em parafrases)
E imprime accuracy / precision / recall / F1 de cada modelo.

Uso:
    python src/evaluate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIGS = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

from preprocess import build_training_data, clean, load_intents  # noqa: E402

DATA = ROOT / "data" / "intents_univali.json"


def _metrics_block(y_true, y_pred, labels, titulo):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {titulo} ===")
    print(f"Acuracia : {acc*100:.2f}%")
    print(f"Precisao : {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_bow():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from tensorflow.keras.models import load_model

    intents = load_intents(DATA)
    X, y, words, classes = build_training_data(intents)
    model = load_model(MODELS / "model_bow.keras")

    probs = model.predict(X, verbose=0)
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(probs, axis=1)

    metrics = _metrics_block(y_true, y_pred, classes, "MODELO FIEL (BoW + TensorFlow)")

    # Fig. 1 — Matriz de confusao
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, xticks_rotation="vertical",
        cmap="Blues", ax=ax, colorbar=True,
    )
    ax.set_title("Matriz de Confusao — Modelo Fiel (BoW)")
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrix_bow.png", dpi=130)
    plt.close(fig)

    # Fig. 2 e 3 — Acuracia e perda por epoca
    hist_path = MODELS / "bow_history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist.get("accuracy", []), label="Acuracia")
        ax.plot(hist.get("loss", []), label="Perda")
        ax.set_xlabel("Epocas")
        ax.set_ylabel("Valor")
        ax.set_title("Acuracia e Perda por Epoca (Fig. 2 do artigo)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "accuracy_loss_bow.png", dpi=130)
        plt.close(fig)

    # Fig. 4 — Curvas ROC (uma por classe)
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, lw=1, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC por classe (Fig. 4 do artigo)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "roc_bow.png", dpi=130)
    plt.close(fig)

    print(f"Figuras salvas em {FIGS}")
    return metrics


# Comparacao de UTILIDADE da resposta final.
# Os dois modelos vivem em espacos de rotulo diferentes (o fiel classifica em
# ~18 intencoes tematicas genericas; o moderno recupera a FAQ real). Por isso a
# comparacao justa NAO e por rotulo, e sim: "a resposta entregue contem a
# informacao esperada?". Cada item: (pergunta_parafraseada, palavra-chave que
# uma resposta CORRETA deve conter).
TESTE_UTILIDADE = [
    ("de que formas eu posso ingressar na univali", "ingres"),
    ("perdi a senha da intranet, como recupero", "senha"),
    ("consigo transferir meu curso de medicina feito no exterior", "medicina"),
    ("tem algum auxilio para quem precisa de ajuda para se manter", "benef"),
    ("quero fazer um curso livre, como funciona", "curso"),
    ("como faco para reingressar na universidade", "reingres"),
    ("preciso de uma certidao de estudos", "certid"),
    ("o que e o programa universidade gratuita", "gratuit"),
]


def evaluate_utilidade():
    """Compara a utilidade da resposta final: fiel (BoW) vs moderno (embeddings)."""
    import matplotlib.pyplot as plt

    from chatbot import ChatBot

    bot_bow = ChatBot(mode="bow")
    bot_emb = ChatBot(mode="embeddings")

    acertos = {"bow": 0, "embeddings": 0}
    print("\n=== TESTE DE UTILIDADE DA RESPOSTA (parafrases) ===")
    print(f"{'pergunta':52s} {'chave':9s} {'BoW':5s} {'Emb':5s}")
    for q, chave in TESTE_UTILIDADE:
        rb = bot_bow.answer(q)["resposta"].lower()
        re_ = bot_emb.answer(q)["resposta"].lower()
        ok_b = chave in rb
        ok_e = chave in re_
        acertos["bow"] += int(ok_b)
        acertos["embeddings"] += int(ok_e)
        print(f"{q[:51]:52s} {chave:9s} {'OK' if ok_b else '--':5s} {'OK' if ok_e else '--':5s}")

    n = len(TESTE_UTILIDADE)
    pb = acertos["bow"] / n * 100
    pe = acertos["embeddings"] / n * 100
    print(f"\nRespostas uteis - Fiel (BoW)        : {acertos['bow']}/{n} ({pb:.0f}%)")
    print(f"Respostas uteis - Moderno (Embeddings): {acertos['embeddings']}/{n} ({pe:.0f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Fiel (BoW)", "Moderno (Embeddings)"], [pb, pe], color=["#1f6f78", "#2aa198"])
    ax.set_ylabel("% de respostas com a informacao esperada")
    ax.set_title("Utilidade da resposta a perguntas reais (parafrases)")
    ax.set_ylim(0, 100)
    for i, v in enumerate([pb, pe]):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGS / "comparacao_utilidade.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    evaluate_bow()
    evaluate_utilidade()
