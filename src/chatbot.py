"""
Inferencia do chatbot — modelo fiel ao artigo (Salloum et al., 2024):
TensorFlow + bag-of-words.

Ha um limiar de confianca: abaixo dele, responde com o intent 'fallback'.

Camada extra: quando o usuario digita o NOME de um curso (sozinho ou em
"curso de X", "quero fazer X"), respondemos com os dados reais daquele curso
(campus, turno, duracao, conceito MEC) a partir de data/cursos_univali.json,
em vez de cair no classificador generico ou no fallback.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import pickle
import random
import re
from pathlib import Path

import numpy as np

from preprocess import bag_of_words, load_intents, strip_accents

ROOT = Path(__file__).resolve().parents[1]
DATA_TOPICOS = ROOT / "data" / "intents_univali.json"   # modelo fiel (BoW)
DATA_CURSOS = ROOT / "data" / "cursos_univali.json"     # dados reais por curso
MODELS = ROOT / "models"

# Palavras de contexto de "curso" + conectivos que podem sobrar quando o
# usuario digita so o nome do curso (ex.: "quero fazer o curso de direito").
_CURSO_CUE = {
    "curso", "cursos", "graduacao", "faculdade", "cursar", "bacharelado",
    "licenciatura", "tecnologo", "formacao", "area", "estudar", "fazer",
    "quero", "gostaria", "sobre", "o", "a", "os", "as", "de", "do", "da",
    "no", "na", "em", "que", "e", "tem", "informacoes", "info", "me",
    "interessa", "queria", "fazer", "um", "uma", "pra", "para",
}


def _norm(s: str) -> str:
    return strip_accents(s).lower().strip()


class ChatBot:
    def __init__(self, threshold: float = 0.6):
        """threshold: confianca minima; abaixo disso -> fallback."""
        self.intents = load_intents(DATA_TOPICOS)
        self.responses = {it["tag"]: it["responses"] for it in self.intents["intents"]}
        self.threshold = threshold

        self._load_cursos()

        from tensorflow.keras.models import load_model

        self.model = load_model(MODELS / "model_bow.keras")
        with open(MODELS / "bow_artifacts.pkl", "rb") as f:
            art = pickle.load(f)
        self.words = art["words"]
        self.classes = art["classes"]

    # --- catalogo de cursos ---------------------------------------------------
    def _load_cursos(self):
        """Indexa cursos por nome normalizado, agregando campus/turno."""
        self.cursos: dict[str, dict] = {}
        if not DATA_CURSOS.exists():
            self._curso_keys: list[str] = []
            return
        with open(DATA_CURSOS, encoding="utf-8") as f:
            raw = json.load(f)
        for c in raw:
            nome = (c.get("curso") or "").strip()
            if not nome:
                continue
            key = _norm(nome)
            self.cursos.setdefault(key, {"nome": nome, "entries": []})
            self.cursos[key]["entries"].append(c)
        # nomes mais longos primeiro (ex.: "engenharia civil" antes de "engenharia")
        self._curso_keys = sorted(self.cursos, key=len, reverse=True)

    def _match_curso(self, text: str) -> dict | None:
        """Retorna o curso se o texto for essencialmente o nome de um curso.

        Alta precisao: so dispara quando, ao remover o nome do curso, o que
        sobra sao apenas palavras de contexto/conectivos. Assim 'psicologia' e
        'quero fazer direito' disparam, mas 'tenho direito ao prouni' nao.
        """
        t = _norm(text)
        for key in self._curso_keys:
            if re.search(r"\b" + re.escape(key) + r"\b", t):
                leftover = re.sub(r"\b" + re.escape(key) + r"\b", " ", t)
                extras = set(re.findall(r"[a-z0-9]+", leftover))
                if extras <= _CURSO_CUE:
                    return self.cursos[key]
        return None

    def _resposta_curso(self, curso: dict) -> str:
        entries = curso["entries"]
        nome = curso["nome"]
        e0 = entries[0]
        grau = (e0.get("grau") or "").strip()
        campi = sorted({(e.get("campus") or "").strip() for e in entries if e.get("campus")})
        # turnos podem vir combinados no proprio dado ("Matutino / Noturno");
        # separa e deduplica para nao repetir ("Noturno/Noturno").
        turnos_set: set[str] = set()
        for e in entries:
            for parte in re.split(r"[/,]", e.get("turno") or ""):
                parte = parte.strip()
                if parte:
                    turnos_set.add(parte)
        turnos = sorted(turnos_set)
        dur = (e0.get("duracao") or "").strip()
        mec = (e0.get("conceito_mec") or "").strip()

        txt = f"O curso de {nome}"
        if grau:
            txt += f" ({grau})"
        if campi:
            txt += f" e oferecido em: {', '.join(campi)}"
        det = []
        if turnos:
            det.append("turno " + "/".join(turnos))  # ex.: Matutino/Noturno
        if dur:
            det.append(f"duracao {dur}")
        if mec:
            det.append(f"conceito MEC {mec}")
        if det:
            txt += ". " + "; ".join(det)
        txt += ". Valores, grade e mais detalhes: https://portal.univali.br/graduacao. Duvidas: +55 47 9130-0269."
        return txt

    # --- predicao da intencao -------------------------------------------------
    def predict_intent(self, text: str) -> tuple[str, float]:
        vec = np.array([bag_of_words(text, self.words)], dtype="float32")
        probs = self.model.predict(vec, verbose=0)[0]
        i = int(np.argmax(probs))
        return self.classes[i], float(probs[i])

    # --- resposta final -------------------------------------------------------
    def answer(self, text: str) -> dict:
        tag, conf = self.predict_intent(text)

        # Prioridade: se o usuario digitou o nome de um curso, responde com os
        # dados reais daquele curso (evita fallback e resposta generica).
        curso = self._match_curso(text)
        if curso is not None:
            return {
                "intent": "curso_especifico",
                "confianca": round(conf, 3),
                "resposta": self._resposta_curso(curso),
            }

        if conf < self.threshold:
            tag = "fallback"
        resp = random.choice(self.responses.get(tag, self.responses["fallback"]))
        return {"intent": tag, "confianca": round(conf, 3), "resposta": resp}


def _repl():
    bot = ChatBot()
    print("Chatbot UNIVALI [BoW] — digite 'sair' para encerrar.\n")
    while True:
        try:
            text = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"sair", "exit", "quit"}:
            break
        out = bot.answer(text)
        print(f"Bot ({out['intent']} | {out['confianca']}): {out['resposta']}\n")


if __name__ == "__main__":
    _repl()
