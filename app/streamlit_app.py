"""
Interface web do chatbot de FAQ da UNIVALI.

Executar (a partir da raiz do projeto):
    .venv/Scripts/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from chatbot import ChatBot

st.set_page_config(page_title="Chatbot UNIVALI", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Carregando modelo...")
def carregar_bot() -> ChatBot:
    return ChatBot()


st.title("🎓 Chatbot FAQ — UNIVALI")
st.caption("Implementacao do artigo *Building and Evaluating a Chatbot Using a University FAQs Dataset* (Salloum et al., 2024), adaptado para a UNIVALI.")

with st.sidebar:
    st.header("Configuracoes")
    st.caption("Modelo: **Fiel ao artigo (BoW + TensorFlow)**")
    mostrar_debug = st.checkbox("Mostrar intencao e confianca", value=True)
    if st.button("Limpar conversa"):
        st.session_state.pop("mensagens", None)
        st.rerun()
    st.markdown("---")
    st.caption("As respostas marcadas com **[VERIFICAR]** ainda precisam ser confirmadas com dados oficiais da UNIVALI.")

bot = carregar_bot()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {"role": "assistant", "content": "Ola! Sou o assistente virtual da UNIVALI. Pergunte sobre cursos, mensalidades, bolsas, vestibular, matricula, biblioteca, campi e contato."}
    ]

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug"):
            st.caption(msg["debug"])

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    out = bot.answer(prompt)
    debug = None
    if mostrar_debug:
        debug = f"intenção: `{out['intent']}` · confiança: `{out['confianca']}`"

    with st.chat_message("assistant"):
        st.markdown(out["resposta"])
        if debug:
            st.caption(debug)
    st.session_state.mensagens.append(
        {"role": "assistant", "content": out["resposta"], "debug": debug}
    )
