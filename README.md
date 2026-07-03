---
title: Chatbot FAQ UNIVALI
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.1
app_file: app/streamlit_app.py
python_version: "3.12"
pinned: false
---

# Chatbot FAQ — UNIVALI

Implementação do artigo **_Building and Evaluating a Chatbot Using a University FAQs Dataset_** (Salloum, Shalan, Basiouni, Salloum & Alfaisal, 2024 — CCIS 2162, Springer), adaptada para a **UNIVALI** e em **português (PT-BR)**.

Disciplina: Inteligência Artificial 2 — UNIVALI.

## O que este projeto entrega

Um chatbot de FAQ universitário **fiel ao artigo**: classifica a pergunta em
**intenções temáticas** usando **bag-of-words + rede densa**, exatamente como no
paper (200 épocas, batch 5, Adam, softmax), e responde com o texto da intenção
prevista. O front-end é uma interface web de chat em Streamlit.

| O que faz | Dados | Stack |
|---|---|---|
| Classifica a pergunta em intenções temáticas com bag-of-words + rede densa. | `intents_univali.json` (curado) | TensorFlow / Keras |

A avaliação reproduz as figuras do artigo (matriz de confusão, curvas de
acurácia/perda, ROC) e inclui um teste de robustez a paráfrases.

> Os dados foram coletados das páginas públicas da UNIVALI (perguntas frequentes,
> intercâmbio, reingresso, auxílio permanência, universidade gratuita, cursos
> livres, etc.) com o scraper incluído.

## Estrutura

```
data/
  intents_univali.json        # intenções temáticas (BoW) — base do chatbot
  intents_univali.seed.json   # backup das intenções-semente originais
  intents_univali_draft.json  # rascunho bruto do scraper
  cursos_univali.json         # dados dos cursos (scraper)
src/
  preprocess.py               # tokenização + lematização (spaCy PT) + bag-of-words
  train_bow.py                # treina o modelo (TensorFlow)
  evaluate.py                 # métricas + matriz de confusão + ROC + robustez
  chatbot.py                  # inferência (CLI)
  scraper/scrape_cursos.py    # extrai dados de cada curso (campus, turno, duração, MEC)
  scraper/scrape_univali.py   # coleta páginas da UNIVALI (requests + Playwright p/ JS)
app/
  streamlit_app.py            # interface web de chat
models/                       # modelo treinado + figuras (gerados)
```

## Como rodar

> Pré-requisito: ambiente com **Python 3.12** (TensorFlow não suporta o Python 3.14).
> O venv fica em `.venv/`.

Se precisar recriar o ambiente do zero:

```bash
pip install uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv -r requirements.txt
python -m uv pip install --python .venv "https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.8.0/pt_core_news_sm-3.8.0-py3-none-any.whl"
```

### 1. Treinar o modelo

```bash
.venv/Scripts/python.exe src/train_bow.py          # treina o modelo (TensorFlow)
```
> A base `intents_univali.json` já vem pronta no repositório.

### 2. Avaliar e gerar os gráficos

```bash
.venv/Scripts/python.exe src/evaluate.py
# figuras em models/figs/
```

### 3. Conversar no terminal

```bash
.venv/Scripts/python.exe src/chatbot.py
```

### 4. Interface web (front)

```bash
.venv/Scripts/streamlit run app/streamlit_app.py
```

### (Opcional) Recoletar os dados da UNIVALI

Só necessário para reraspar as páginas. Descomente as dependências do scraper em
`requirements.txt`, instale e rode:

```bash
.venv/Scripts/python.exe -m playwright install chromium   # navegador p/ páginas JS
.venv/Scripts/python.exe src/scraper/scrape_univali.py
```
> O scraper tenta `requests` (rápido) e, se a página carregar conteúdo via
> JavaScript (SharePoint/SPA), renderiza com o Playwright. Respeita `robots.txt`.

## Pipeline (igual ao artigo)

1. **Tokenização** — divide o texto em tokens (regex/spaCy).
2. **Lematização** — reduz à forma base. O artigo usa NLTK/WordNet (só inglês);
   aqui usamos **spaCy `pt_core_news_sm`** por o dataset ser em português.
3. **Vetorização** — bag-of-words (contagem de ocorrências).
4. **Classificação** — rede densa sobre o bag-of-words (softmax).
5. **Resposta** — texto da intenção temática prevista. Abaixo do limiar de
   confiança, cai no `fallback`.

## Observações honestas sobre os resultados

- O modelo atinge **~100% de acurácia no próprio conjunto de treino** — o mesmo
  efeito (AUC = 1,00) reportado no artigo. Isso reflete **overfitting / dataset
  pequeno**, como discutido no slide de limitações, e **não** capacidade real de
  generalização.
- O dataset foi expandido para **36 intenções / ~500 patterns** (paráfrases naturais
  em PT-BR + tópicos novos com dados oficiais: Certidão de Estudos, app Minha Univali,
  Atividades Complementares). Isso melhorou a generalização de forma concreta:
  - Teste de robustez (paráfrases): **62% → 100%**
  - Perguntas totalmente inéditas (fora dos patterns): **~90%** de acerto de intenção
- Mais patterns = mais vocabulário e formas de perguntar. Aumentar **épocas** não
  ajudaria (o modelo já satura em ~100% no treino); o ganho real vem de **mais dados**.
- **Busca de curso por nome:** além do classificador, quando o usuário digita o nome
  de um curso (ex.: "medicina", "ciência da computação", "quero fazer psicologia"),
  o bot responde com os dados reais daquele curso (campus, turno, duração, conceito
  MEC) a partir de `data/cursos_univali.json`. Nomes parciais/ambíguos ("computação",
  "engenharia") listam as opções. Feito com alta precisão para não confundir casos
  como "tenho **direito** ao prouni".

## Próximos passos

- [ ] Substituir os textos `[VERIFICAR]` das intenções temáticas por respostas oficiais.
- [ ] Conjunto de teste separado (hold-out) para métricas honestas de generalização.
- [ ] Mais paráfrases por intenção para melhorar a robustez.
- [ ] Avaliar conformidade com LGPD ao usar dados/identidade da instituição.

## Créditos

Salloum, S.A. et al. (2024). _Building and Evaluating a Chatbot Using a University FAQs Dataset_. CCIS 2162, Springer Nature. DOI: 10.1007/978-3-031-65996-6_18

Dataset original de referência: [Chatbot Dataset — Nirali Vaghani (Kaggle)](https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset)
