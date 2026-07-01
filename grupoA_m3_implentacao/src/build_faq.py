"""
Monta a base de conhecimento de FAQ real da UNIVALI (data/faq_univali.json),
usada pelo modelo MODERNO (embeddings/retrieval).

Junta:
  - As perguntas/respostas REAIS raspadas (data/intents_univali_draft.json)
  - As intencoes conversacionais curadas (saudacao, despedida, agradecimento,
    fallback) de data/intents_univali.json — para o bot saber cumprimentar.

Faz deduplicacao global por pergunta e descarta respostas curtas/ruido.

Uso:
    python src/build_faq.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data" / "intents_univali_draft.json"
TOPICOS = ROOT / "data" / "intents_univali.json"
CURSOS = ROOT / "data" / "cursos_univali.json"
SAIDA = ROOT / "data" / "faq_univali.json"

CONVERSACIONAIS = {"saudacao", "despedida", "agradecimento", "fallback"}

TEL = "+55 47 9130-0269"

BOLSAS = "(47) 3341-7873 / bolsas@univali.br"

# Intencoes curadas adicionadas a base de embeddings (nao vem do scraping).
# Garantem respostas limpas de alto nivel onde as FAQs raspadas sao muito especificas.
EXTRAS_CURADOS = [
    {
        "tag": "contato",
        "patterns": [
            "qual o telefone da univali", "telefone para contato", "telefone da secretaria",
            "como entro em contato", "como falar com a secretaria", "fale conosco",
            "qual o numero de contato", "quero falar com um atendente", "contato da univali",
        ],
        "responses": [
            f"Fale com a UNIVALI: Central de Relacionamento 0800 723 1300 / falecom@univali.br, "
            f"WhatsApp/telefone {TEL}, ou em https://portal.univali.br/fale-conosco."
        ],
    },
    {
        "tag": "bolsas_financiamento",
        "patterns": [
            "bolsas de estudo", "quais bolsas a univali oferece", "tem bolsa de estudo",
            "como conseguir bolsa", "tipos de bolsa", "desconto na mensalidade",
            "financiamento estudantil", "como financiar a faculdade", "tem fies",
        ],
        "responses": [
            "A UNIVALI possui programas de bolsas e financiamento (Universidade Gratuita, ProUni, "
            "UNIEDU, FIES e bolsas proprias). Opcoes e requisitos: https://portal.univali.br/bolsas. "
            f"Atendimento de bolsas: {BOLSAS}."
        ],
    },
    {
        "tag": "trancamento",
        "patterns": [
            "consigo trancar o curso", "como trancar a matricula", "quero trancar o curso",
            "posso trancar a faculdade", "trancamento de matricula", "como faco para trancar",
            "preciso parar o curso por um tempo",
        ],
        "responses": [
            "Para trancar a matricula: nos cursos presenciais, solicite pelo Portal do Aluno; nos "
            "cursos EAD, contate o Atendimento ao Estudante (0800 729 1756 / WhatsApp (47) 99249-9953), "
            "dentro do prazo do calendario academico. Bolsistas (ex.: UNIEDU) podem ter implicacoes "
            "financeiras. Reingresso possivel em ate 4 anos."
        ],
    },
    {
        "tag": "ouvidoria",
        "patterns": [
            "como faco uma reclamacao", "quero registrar uma reclamacao", "ouvidoria",
            "fazer uma sugestao", "canal de denuncia", "quero elogiar um atendimento",
        ],
        "responses": [
            "A Ouvidoria da UNIVALI e o canal para sugestoes, criticas, reclamacoes e elogios. "
            "Contato: ouvidoria@univali.br ou https://portal.univali.br/ouvidoria."
        ],
    },
    {
        "tag": "ead",
        "patterns": [
            "como funciona o ead", "tem curso a distancia", "curso online",
            "graduacao a distancia", "estudar pela internet", "como sao as provas no ead",
        ],
        "responses": [
            "A UNIVALI oferece cursos EAD e semipresenciais: conteudo disponivel 24h, com encontros "
            "presenciais no campus a cada dois meses para as provas. E possivel cursar em paralelo um "
            "EAD e um presencial (de areas diferentes). Saiba mais: https://ead.univali.br e "
            "https://ead.univali.br/central-de-ajuda."
        ],
    },
    {
        "tag": "transferencia_interna",
        "patterns": [
            "transferencia interna", "quero mudar de curso", "mudar de campus", "trocar de turno",
            "mudar de curso na univali", "como faco transferencia interna", "posso trocar de curso",
        ],
        "responses": [
            "A Transferencia Interna permite mudar de curso, campus ou turno dentro da UNIVALI, conforme "
            "edital (geralmente em maio e outubro). Solicite em https://www.univali.br/transferencias e "
            "anexe os documentos no prazo. Duvidas: WhatsApp (47) 99130-0269 / falecom@univali.br."
        ],
    },
    {
        "tag": "calendario",
        "patterns": [
            "calendario academico", "quando comecam as aulas", "quando comeca o semestre",
            "datas importantes", "quando termina o semestre", "feriados e recessos", "inicio das aulas",
        ],
        "responses": [
            "O Calendario Academico (aulas, matriculas, recessos e feriados) esta em "
            "https://www.univali.br/calendario, no Portal do Aluno e no app Minha Univali. As aulas do "
            "1o semestre de 2026 comecam em 26 de fevereiro. Atendimento ao estudante: (47) 3341-7814."
        ],
    },
    {
        "tag": "fies",
        "patterns": [
            "o que e o fies", "como funciona o fies", "quero o fies", "como contratar o fies",
            "requisitos do fies", "financiamento estudantil fies", "quem pode aderir ao fies",
        ],
        "responses": [
            "O FIES (do MEC) financia a graduacao. Requisitos: ter feito o ENEM a partir de 2010 com "
            "media >= 450 e redacao > 0, e renda familiar per capita de ate 3 salarios minimos. "
            "Contratacao em 3 etapas: inscricao no site do FIES, entrega de documentos no campus em ate "
            f"10 dias (com agendamento) e formalizacao no banco. Atendimento de bolsas: {BOLSAS}."
        ],
    },
    {
        "tag": "formatura",
        "patterns": [
            "como funciona a formatura", "colacao de grau", "tipos de formatura", "quero me formar",
            "como participar da colacao", "formatura", "quando e a colacao de grau",
        ],
        "responses": [
            "A UNIVALI oferece 3 tipos de colacao de grau: Oficial/Institucional (organizada pela "
            "universidade, sem custo, 30 a 70 formandos), Especial (a turma contrata empresa e elege "
            "comissao, reserva com 1 ano de antecedencia pelo Portal do Aluno) e Online (10 a 50 "
            "formandos). Para colar grau e preciso ter concluido todas as disciplinas e atividades, "
            "estar regular com o ENADE e com a documentacao. Mais: https://www.univali.br/formaturas."
        ],
    },
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


# Termos de menu/navegacao: resposta com varios deles e dump de menu (ruido)
_NAV = ["Colégio de Aplicação", "Doutorado", "Graduação", "Mestrado",
        "Pós-Graduação", "Univali Idiomas", "Filtros", "INSCRIÇÕES", "INTRANET"]


def eh_ruido(q: str, a: str) -> bool:
    if "Clique no botão abaixo" in a:           # CTA generico
        return True
    if "Selecione o seu Vínculo" in a:          # widget de formulario
        return True
    if sum(1 for n in _NAV if n in a) >= 3:     # dump de menu
        return True
    return False


def limpar_resposta(q: str, a: str) -> str:
    """Remove repeticao da pergunta no inicio da resposta."""
    if a.lower().startswith(q.lower()[:25]):
        idx = a.find("?")
        if 0 < idx < len(a) - 10:
            a = a[idx + 1:].strip()
    return a


def cursos_para_intents() -> list[dict]:
    """
    Gera uma FAQ rica por CURSO (agrupando os campi) a partir de
    data/cursos_univali.json, para o bot responder perguntas especificas
    tipo 'como e o curso de Ciencia da Computacao?'.
    """
    if not CURSOS.exists():
        return []
    registros = json.loads(CURSOS.read_text(encoding="utf-8"))

    # agrupa por nome de curso
    por_curso: dict[str, list[dict]] = {}
    for r in registros:
        por_curso.setdefault(r["curso"], []).append(r)

    intents = []
    for idx, (curso, regs) in enumerate(sorted(por_curso.items())):
        grau = regs[0].get("grau", "")
        # detalha cada campus
        partes = []
        for r in regs:
            det = f"{r['campus']} ({r.get('turno','')}, {r.get('duracao','')}".strip()
            if r.get("carga_horaria"):
                det += f", {r['carga_horaria']}"
            det += ")"
            partes.append(det)
        mec = next((r["conceito_mec"] for r in regs if r.get("conceito_mec")), "")
        url = regs[0].get("url", "")
        objetivo = next((r["objetivo"] for r in regs if r.get("objetivo")), "")

        resp = f"O curso de {curso}"
        if grau:
            resp += f" ({grau})"
        resp += " e oferecido em: " + "; ".join(partes) + "."
        if mec:
            resp += f" Conceito MEC: {mec}."
        if objetivo:
            resp += f" Objetivo: {objetivo}"
        if url:
            resp += f" Mais informacoes: {url}"

        patterns = [
            curso,
            f"curso de {curso}",
            f"como e o curso de {curso}",
            f"quero saber sobre {curso}",
            f"{curso} e em qual campus",
            f"qual a duracao do curso de {curso}",
            f"quanto tempo dura o curso de {curso}",
            f"{curso} qual o turno",
            f"a univali tem {curso}",
            f"quero estudar {curso}",
        ]
        intents.append({"tag": f"curso_{idx}", "patterns": patterns, "responses": [resp]})
    return intents


def main():
    intents_out = []

    # 1) Intencoes conversacionais (mantem patterns/responses originais)
    if TOPICOS.exists():
        topicos = json.loads(TOPICOS.read_text(encoding="utf-8"))
        for it in topicos["intents"]:
            if it["tag"] in CONVERSACIONAIS:
                intents_out.append(
                    {"tag": it["tag"], "patterns": it["patterns"], "responses": it["responses"]}
                )

    # 2) FAQs reais raspadas, com dedup global por pergunta
    vistos = set()
    n_faq = 0
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    for it in draft["intents"]:
        pergunta = it["patterns"][0].strip()
        resposta = limpar_resposta(pergunta, it["responses"][0].strip())
        chave = norm(pergunta)
        if chave in vistos or len(resposta) < 30 or eh_ruido(pergunta, resposta):
            continue
        vistos.add(chave)
        intents_out.append(
            {
                "tag": f"faq_{n_faq}",
                "patterns": [pergunta],
                "responses": [resposta],
                "_fonte": it.get("_fonte", ""),
            }
        )
        n_faq += 1

    # 3) intencoes curadas (contato com telefone, etc.)
    intents_out.extend(EXTRAS_CURADOS)

    # 4) uma FAQ rica por curso (dados estruturados de cursos_univali.json)
    cursos_intents = cursos_para_intents()
    intents_out.extend(cursos_intents)
    n_cursos = len(cursos_intents)

    saida = {
        "_meta": {
            "descricao": "Base de FAQ real da UNIVALI (scraping + curadoria) para o modelo de embeddings.",
            "idioma": "pt-BR",
            "aviso": "Conteudo raspado de paginas publicas da UNIVALI. Revisar periodicamente, pois pode desatualizar.",
        },
        "intents": intents_out,
    }
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Base FAQ salva em {SAIDA}")
    print(f"  - FAQs reais (apos dedup):   {n_faq}")
    print(f"  - FAQs por curso:            {n_cursos}")
    print(f"  - Conversacionais + curados: {len(intents_out) - n_faq - n_cursos}")
    print(f"  - Total de intents:          {len(intents_out)}")


if __name__ == "__main__":
    main()
