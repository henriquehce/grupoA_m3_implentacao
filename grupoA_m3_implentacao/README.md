# Chatbot FAQ — UNIVALI

Implementação do artigo **_Building and Evaluating a Chatbot Using a University FAQs Dataset_** (Salloum, Shalan, Basiouni, Salloum & Alfaisal, 2024 — CCIS 2162, Springer), adaptada para a **UNIVALI** e em **português (PT-BR)**.

Disciplina: Inteligência Artificial 2 — UNIVALI.

## O que este projeto entrega

Um chatbot de FAQ universitário em **arquitetura híbrida** (duas abordagens lado a lado):

| Versão | O que faz | Dados | Stack |
|---|---|---|---|
| **Fiel ao artigo** | Classifica a pergunta em ~18 **intenções temáticas** com bag-of-words + rede densa, exatamente como no paper (200 épocas, batch 5, Adam, softmax). | `intents_univali.json` (curado) | TensorFlow / Keras |
| **Moderna (melhoria)** | **Recupera a FAQ real** mais parecida por similaridade de embeddings semânticos e responde com o **texto oficial da UNIVALI**. Entende paráfrases. | `faq_univali.json` (**~180 FAQs reais** raspadas + curadas) | sentence-transformers (PyTorch) |

A avaliação reproduz as figuras do artigo (matriz de confusão, curvas de acurácia/perda, ROC) para o modelo fiel e **compara a utilidade das respostas** das duas abordagens.

> Os dados reais foram coletados das páginas públicas de FAQ da UNIVALI (perguntas frequentes, intercâmbio, reingresso, auxílio permanência, universidade gratuita, cursos livres, etc.) com o scraper incluído.

## Estrutura

```
data/
  intents_univali.json        # intenções temáticas (modelo FIEL/BoW) — gerado por map_faqs
  intents_univali.seed.json   # backup das intenções-semente originais
  faq_univali.json            # base de FAQ REAL da UNIVALI (modelo MODERNO/embeddings)
  intents_univali_draft.json  # rascunho bruto do scraper (entra no build_faq)
  raw/                        # texto bruto das páginas raspadas
src/
  preprocess.py               # tokenização + lematização (spaCy PT) + bag-of-words
  train_bow.py                # treina o modelo FIEL (TensorFlow)
  train_embeddings.py         # constrói o índice de embeddings (modelo MODERNO)
  build_faq.py                # monta faq_univali.json (FAQs + 1 FAQ rica por curso + curados)
  map_faqs.py                 # mapeia FAQs reais → intenções temáticas (modelo FIEL)
  scraper/scrape_cursos.py    # extrai dados de cada curso (campus, turno, duração, MEC)
  evaluate.py                 # métricas + matriz de confusão + ROC + comparação
  chatbot.py                  # inferência (CLI), suporta os dois modos
  scraper/scrape_univali.py   # coleta páginas da UNIVALI (requests + Playwright p/ JS)
app/
  streamlit_app.py            # interface web de chat
models/                       # modelos treinados + figuras (gerados)
```

## Como rodar

> Pré-requisito: o ambiente já foi criado com **uv** + **Python 3.12** (TensorFlow ainda não suporta o Python 3.14 instalado na máquina). O venv fica em `.venv/`.

Se precisar recriar o ambiente do zero:

```bash
pip install uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv -r requirements.txt
python -m uv pip install --python .venv "https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.8.0/pt_core_news_sm-3.8.0-py3-none-any.whl"
.venv/Scripts/python.exe -m playwright install chromium   # navegador p/ páginas JS
```

### 1. (Opcional) Coletar/atualizar a base de FAQ real

```bash
.venv/Scripts/python.exe src/scraper/scrape_univali.py   # gera intents_univali_draft.json
.venv/Scripts/python.exe src/build_faq.py                # monta faq_univali.json (modelo moderno)
.venv/Scripts/python.exe src/map_faqs.py                 # mapeia FAQs → intenções (modelo fiel)
```
> As bases já vêm prontas no repositório; só rode isto para reraspar/reprocessar.
> O `map_faqs.py` agrupa as FAQs reais por tema (pela página de origem), descarta ruído,
> limita patterns por tema (reduz desbalanceamento) e escreve respostas de tópico curadas
> (incl. telefone da secretaria). O original fica salvo em `intents_univali.seed.json`.

### 2. Treinar os modelos

```bash
.venv/Scripts/python.exe src/train_bow.py          # modelo fiel (TensorFlow)
.venv/Scripts/python.exe src/train_embeddings.py   # modelo moderno (embeddings/FAQ real)
```

### 3. Avaliar e gerar os gráficos

```bash
.venv/Scripts/python.exe src/evaluate.py
# figuras em models/figs/
```

### 4. Conversar no terminal

```bash
.venv/Scripts/python.exe src/chatbot.py embeddings   # ou: bow
```

### 5. Interface web

```bash
.venv/Scripts/streamlit run app/streamlit_app.py
```

> Detalhe do scraper: ele tenta `requests` (rápido) e, se a página carregar conteúdo
> via JavaScript (SharePoint/SPA), renderiza com o Playwright. Respeita `robots.txt`
> e usa intervalo entre requisições. A saída (`intents_univali_draft.json`) é um
> **rascunho** — `build_faq.py` consolida em `faq_univali.json`; revise antes de usar.

## Pipeline (igual ao artigo)

1. **Tokenização** — divide o texto em tokens (regex/spaCy).
2. **Lematização** — reduz à forma base. O artigo usa NLTK/WordNet (só inglês);
   aqui usamos **spaCy `pt_core_news_sm`** por o dataset ser em português.
3. **Vetorização** — bag-of-words (contagem de ocorrências).
4. **Classificação / recuperação** — rede densa sobre o bag-of-words (versão fiel)
   **ou** similaridade de cosseno entre embeddings (versão moderna).
5. **Resposta** — versão fiel: resposta da intenção temática prevista; versão moderna:
   texto real da FAQ mais parecida. Abaixo do limiar de confiança, cai no `fallback`.

## Observações honestas sobre os resultados

- O modelo fiel atinge **~100% de acurácia no próprio conjunto de treino** — o mesmo
  efeito (AUC = 1,00) reportado no artigo. Isso reflete **overfitting / dataset pequeno**,
  como discutido no slide de limitações, e **não** capacidade real de generalização.
- No teste de **utilidade da resposta** (paráfrases de perguntas reais), o modelo moderno
  entrega a informação esperada com mais frequência que o fiel, porque responde com o
  **texto oficial da FAQ** em vez de uma resposta temática genérica.
- O retrieval por embeddings ainda erra em **tópicos sobrepostos** (ex.: "reingresso" vs
  "formas de ingresso") quando a base tem perguntas muito parecidas. Isso melhora com mais
  curadoria e com técnicas de *reranking* — fica como evolução natural do projeto.

## Próximos passos

- [ ] Revisar as 160 FAQs reais em `faq_univali.json` (precisão factual / atualização).
- [ ] Coletar as páginas institucionais sem formato de pergunta (moradia, transporte) de forma estruturada.
- [ ] Substituir os textos `[VERIFICAR]` das intenções temáticas por respostas oficiais.
- [ ] Conjunto de teste separado (hold-out) para métricas honestas de generalização.
- [ ] *Reranking* / limiar adaptativo para reduzir erros em tópicos sobrepostos.
- [ ] Avaliar conformidade com LGPD ao usar dados/identidade da instituição.

## Créditos

Salloum, S.A. et al. (2024). _Building and Evaluating a Chatbot Using a University FAQs Dataset_. CCIS 2162, Springer Nature. DOI: 10.1007/978-3-031-65996-6_18

Dataset original de referência: [Chatbot Dataset — Nirali Vaghani (Kaggle)](https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset)
