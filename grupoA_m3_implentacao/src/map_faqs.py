"""
Mapeia as FAQs REAIS (faq_univali.json) em INTENCOES TEMATICAS e reconstroi
data/intents_univali.json (usado pelo modelo FIEL/BoW).

Logica:
  - Limpa ruido (CTAs, dumps de menu) e corrige respostas que repetem a pergunta.
  - Agrupa cada FAQ por TEMA usando a pagina de origem (sinal mais confiavel
    que adivinhar). Caso especial: perguntas sobre senha/Intranet -> portal_senha.
  - Cada intencao tematica recebe:
       patterns  = perguntas REAIS daquele tema (ate MAX_PATTERNS, p/ reduzir
                   desbalanceamento de classe)
       responses = UMA resposta de topico curada (RESPOSTAS_TOPICO), pois o modelo
                   fiel e um CLASSIFICADOR de intencao (respostas especificas ficam
                   com o modelo de embeddings).
  - Mantem as intencoes conversacionais e alguns temas-semente sem dados reais.

Uso:
    python src/map_faqs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAQ = ROOT / "data" / "faq_univali.json"
SEED = ROOT / "data" / "intents_univali.seed.json"
SAIDA = ROOT / "data" / "intents_univali.json"

TEL = "+55 47 9130-0269"
BOLSAS = "(47) 3341-7873 / bolsas@univali.br"  # Coord. de Atendimento ao Estudante
MAX_PATTERNS = 25  # limita patterns por tema (reduz desbalanceamento no BoW)

# FAQs a descartar (ruido identificado na revisao)
DESCARTAR = {"faq_36", "faq_37", "faq_143"}

# pagina de origem (trecho) -> tema
FONTE_TEMA = {
    "universidade-gratuita": "universidade_gratuita",
    "intercambio": "intercambio",
    "cursos-livres": "cursos_livres",
    "reingresso": "reingresso",
    "auxilio-permanencia": "auxilio_permanencia",
    # A pagina "certidao-de-estudos-externa" contem, na verdade, FAQs de
    # transferencia externa / concessao de vaga -> agrupamos em transferencia.
    "certidao-de-estudos": "transferencia",
    "transferencia-externa": "transferencia",
    "fale-conosco": "contato",
    # temas especificos ANTES de "bolsas" (as URLs contem "bolsas-e-financiamentos")
    "programa-uniedu": "uniedu",
    "prouni": "prouni",
    "enade": "enade",
    "mestrado": "pos_graduacao",
    "segunda-graduacao": "segunda_graduacao",
    "bolsas": "bolsas_financiamento",
    "perguntas-frequentes": "ingresso",  # FAQ geral (fallback do portal)
}

# Resposta de topico (curada) para cada intencao tematica vinda dos dados reais
RESPOSTAS_TOPICO = {
    "ingresso": [
        f"A UNIVALI tem varias formas de ingresso (vestibular, ENEM, vestibular ACAFE, "
        f"transferencia e Programa Universidade Gratuita). Veja detalhes e inscricoes em "
        f"https://www.univali.br/vestibular ou fale com a secretaria: {TEL}."
    ],
    "portal_senha": [
        "Para recuperar a senha da Intranet (Portal do Aluno): acesse www.univali.br/intranet, "
        "clique em 'Esqueci Minha Senha' e informe seu login (CPF, e-mail, passaporte ou codigo). "
        f"Em caso de duvida, fale com a secretaria: {TEL}."
    ],
    "universidade_gratuita": [
        "O Programa Universidade Gratuita (Governo de SC) oferece bolsas de estudo integrais conforme "
        "criterios de renda e grupo familiar. Perguntas frequentes e inscricao: "
        f"https://portal.univali.br/universidade-gratuita/perguntas-frequentes ou secretaria: {TEL}."
    ],
    "intercambio": [
        "A UNIVALI oferece programas de intercambio nacional e internacional. Editais, requisitos e "
        f"processos: https://www.univali.br/intercambio ou fale com a secretaria: {TEL}."
    ],
    "cursos_livres": [
        "Os Cursos Livres da UNIVALI sao abertos a comunidade e emitem certificado digital com carga "
        "horaria. Oferta e inscricoes: https://portal.univali.br/cursos-livres."
    ],
    "reingresso": [
        "Para retornar/reingressar ao seu curso na UNIVALI, consulte o processo de reingresso: "
        f"https://portal.univali.br/reingresso ou fale com a secretaria: {TEL}."
    ],
    "auxilio_permanencia": [
        "O Auxilio Permanencia Univali apoia estudantes bolsistas que atendam aos criterios (CadUnico, "
        "renda por integrante, frequencia minima, entre outros). Detalhes e solicitacao: "
        "https://portal.univali.br/auxilio-permanencia."
    ],
    "certidao_estudos": [
        "A Certidao de Estudos (externa) e emitida pela secretaria mediante solicitacao. Requisitos e "
        f"prazos: https://portal.univali.br/certidao-de-estudos-externa ou secretaria: {TEL}."
    ],
    "transferencia": [
        "A UNIVALI aceita transferencia externa de outras instituicoes, com analise de aproveitamento de "
        f"disciplinas: https://portal.univali.br/transferencia-externa ou secretaria: {TEL}. "
        "Obs.: nao e permitida a transferencia do curso de Medicina cursado no exterior."
    ],
    "contato": [
        f"Fale com a UNIVALI: Central de Relacionamento 0800 723 1300 / falecom@univali.br, "
        f"WhatsApp/telefone {TEL}, ou em https://portal.univali.br/fale-conosco."
    ],
    "ouvidoria": [
        "A Ouvidoria da UNIVALI e o canal para sugestoes, criticas, reclamacoes e elogios. "
        "Contato: ouvidoria@univali.br ou https://portal.univali.br/ouvidoria."
    ],
    "segunda_graduacao": [
        "A Segunda Graduacao e para quem ja tem diploma de ensino superior e quer um novo curso, "
        "com incentivo de ate 40% de desconto. Editais e inscricao: "
        f"https://portal.univali.br/segunda-graduacao ou secretaria: {TEL}."
    ],
    "trancamento": [
        "Para trancar a matricula: nos cursos presenciais, solicite pelo Portal do Aluno; nos "
        "cursos EAD, contate o Atendimento ao Estudante (0800 729 1756 / WhatsApp (47) 99249-9953), "
        "sempre dentro do prazo do calendario academico. Atencao: bolsistas (ex.: UNIEDU) podem ter "
        "implicacoes financeiras. Voce pode pedir reingresso em ate 4 anos. Duvidas: secretaria "
        f"{TEL}."
    ],
    "transferencia_interna": [
        "A Transferencia Interna permite mudar de curso, campus ou turno dentro da UNIVALI, conforme "
        "edital (geralmente em maio e outubro). Solicite em https://www.univali.br/transferencias, "
        "anexando os documentos no prazo; o aproveitamento de disciplinas e feito pelo coordenador do "
        f"curso pretendido. Duvidas: WhatsApp {TEL} / falecom@univali.br."
    ],
    "calendario": [
        "O Calendario Academico (inicio e fim das aulas, matriculas, recessos e feriados) esta em "
        "https://www.univali.br/calendario, no Portal do Aluno e no app Minha Univali. As aulas do 1o "
        "semestre de 2026 comecam em 26 de fevereiro. Atendimento ao estudante: (47) 3341-7814."
    ],
    "fies": [
        "O FIES (Fundo de Financiamento Estudantil, do MEC) financia a graduacao. Requisitos: ter feito "
        "o ENEM a partir de 2010 com media >= 450 e nota de redacao > 0, e renda familiar per capita de "
        "ate 3 salarios minimos. Contratacao em 3 etapas: inscricao no site do FIES, entrega de "
        "documentos no campus em ate 10 dias (com agendamento) e formalizacao no banco. Atendimento de "
        f"bolsas: {BOLSAS}."
    ],
    "formatura": [
        "A UNIVALI oferece 3 tipos de colacao de grau: Oficial/Institucional (organizada pela "
        "universidade, sem custo, de 30 a 70 formandos), Especial (a turma contrata empresa e elege "
        "comissao, com reserva de data com 1 ano de antecedencia pelo Portal do Aluno) e Online "
        "(de 10 a 50 formandos). Para colar grau e preciso ter concluido todas as disciplinas e "
        "atividades, estar regular com o ENADE e com a documentacao. Mais: "
        "https://www.univali.br/formaturas."
    ],
    "bolsas_financiamento": [
        "A UNIVALI possui programas de bolsas e financiamento (Universidade Gratuita, ProUni, UNIEDU, "
        "FIES e bolsas proprias). Veja opcoes e requisitos em https://portal.univali.br/bolsas. "
        f"Atendimento de bolsas: {BOLSAS}."
    ],
    "uniedu": [
        "O Programa UNIEDU (Governo de Santa Catarina) oferece bolsas e financiamento estudantil "
        "(incluindo o Art. 171). Requisitos, editais e regras de acumulo: "
        f"https://portal.univali.br/programa-uniedu/perguntas-frequentes. Atendimento: {BOLSAS}."
    ],
    "prouni": [
        "O ProUni (Programa Universidade para Todos, do MEC) oferece bolsas integrais e parciais. "
        "Candidatos ProUni nao precisam prestar vestibular nem estar matriculados previamente. "
        f"Mais informacoes e atendimento de bolsas: {BOLSAS}."
    ],
    "enade": [
        "O ENADE avalia o desempenho dos estudantes e e componente obrigatorio do curso. "
        "Perguntas frequentes e dicas: "
        "https://www.univali.br/institucional/vrgdi/coordenadoria-de-processos-regulatorios/enade/"
        f"perguntas-frequentes-e-dicas. Duvidas: secretaria {TEL}."
    ],
    # --- temas-semente, com conteudo real (web) no lugar de [VERIFICAR] ---
    "cursos": [
        "A UNIVALI oferece mais de 40 cursos de graduacao, entre eles: Medicina, Odontologia, "
        "Enfermagem, Fisioterapia, Psicologia, Direito, Administracao, Ciencias Contabeis, "
        "Engenharia Civil, Engenharia de Computacao, Engenharia Mecanica, Engenharia Quimica, "
        "Ciencia da Computacao, Inteligencia Artificial, Arquitetura e Urbanismo, Design, "
        "Jornalismo, Publicidade e Propaganda, Gastronomia, Educacao Fisica, Pedagogia e "
        "Oceanografia. Lista completa e atualizada: https://portal.univali.br/graduacao."
    ],
    "campus_localizacao": [
        "A UNIVALI tem campi e unidades em Itajai, Balneario Camboriu, Biguacu, Tijucas, "
        "Sao Jose (Kobrasol) e Florianopolis. Enderecos completos: https://portal.univali.br "
        f"ou fale com a secretaria: {TEL}."
    ],
    "pos_graduacao": [
        "A UNIVALI oferece pos-graduacao lato sensu (especializacao/MBA) e stricto sensu "
        f"(mestrado/doutorado). Veja a oferta em https://portal.univali.br/pos ou secretaria: {TEL}."
    ],
    "ead": [
        "A UNIVALI oferece cursos EAD e semipresenciais: conteudo disponivel 24h, com encontros "
        "presenciais no campus a cada dois meses para as provas. Voce pode cursar em paralelo um "
        "EAD e um presencial (de areas diferentes). Saiba mais em https://ead.univali.br ou na "
        "Central de Ajuda EAD: https://ead.univali.br/central-de-ajuda."
    ],
    "mensalidade_valores": [
        "Os valores de mensalidade variam por curso e campus. Consulte o valor na pagina do curso "
        f"desejado em https://portal.univali.br/graduacao ou fale com a secretaria: {TEL}."
    ],
    "biblioteca": [
        "A UNIVALI conta com bibliotecas em seus campi e acervo virtual (emprestimo, renovacao e "
        f"bases digitais). Informacoes e acesso: https://www.univali.br/biblioteca ou secretaria: {TEL}."
    ],
    "matricula": [
        "A matricula e a rematricula (a cada semestre) sao feitas pelo Portal do Aluno. Documentos, "
        f"prazos e duvidas: fale com a Secretaria Academica pelo telefone {TEL}."
    ],
    "secretaria_documentos": [
        "Documentos academicos (historico, atestado de matricula, declaracoes) sao solicitados a "
        f"Secretaria Academica pelo Portal do Aluno ou pelo telefone {TEL}."
    ],
    "horario_atendimento": [
        "Os horarios de atendimento variam por setor e campus. Para confirmar o horario do setor "
        f"desejado, fale com a secretaria pelo telefone {TEL} ou em https://portal.univali.br/fale-conosco."
    ],
    "estagio_carreira": [
        "A UNIVALI apoia estagios e empregabilidade por meio de seus nucleos de carreira. "
        f"Para oportunidades e orientacao, fale com a secretaria: {TEL} ou https://www.univali.br."
    ],
}

# Patterns curados (escritos a mao) p/ temas mal cobertos pelo scraping.
# Sao ADICIONADOS aos patterns reais (e criam o tema se ele nao existir).
EXTRA_PATTERNS = {
    "contato": [
        "qual o telefone da univali", "telefone para contato", "telefone da secretaria",
        "como entro em contato", "como falar com a secretaria", "fale conosco",
        "qual o numero de contato", "email de contato", "quero falar com um atendente",
        "contato da univali",
    ],
    "reingresso": [
        "quero reingressar no meu curso", "como faco para reingressar",
        "voltar a estudar na univali", "retomar meu curso", "reingresso",
        "abandonei o curso e quero voltar",
    ],
    "portal_senha": [
        "esqueci minha senha do portal do aluno", "recuperar senha da intranet",
        "perdi a senha do portal", "nao consigo acessar a intranet",
        "como redefinir minha senha",
    ],
    "prouni": [
        "o que e o prouni", "como funciona o prouni", "tenho direito ao prouni",
        "bolsa prouni", "prouni na univali", "como conseguir prouni",
    ],
    "bolsas_financiamento": [
        "quais bolsas a univali oferece", "tem bolsa de estudo", "como conseguir bolsa",
        "desconto na mensalidade", "tipos de bolsa", "bolsa de estudos",
        "como financiar a faculdade",
    ],
    "ouvidoria": [
        "como faco uma reclamacao", "quero registrar uma reclamacao", "ouvidoria",
        "fazer uma sugestao", "canal de denuncia", "quero elogiar um atendimento",
        "registrar uma critica",
    ],
    "segunda_graduacao": [
        "quero fazer uma segunda graduacao", "ja tenho diploma e quero outro curso",
        "segunda graduacao", "fazer outra faculdade tendo diploma", "novo curso com desconto",
    ],
    "trancamento": [
        "consigo trancar o curso", "como trancar a matricula", "quero trancar o curso",
        "posso trancar a faculdade", "trancamento de matricula", "preciso parar o curso por um tempo",
        "como faco para trancar",
    ],
    "transferencia_interna": [
        "transferencia interna", "quero mudar de curso", "mudar de campus", "trocar de turno",
        "mudar de curso na univali", "como faco transferencia interna", "posso trocar de curso",
    ],
    "calendario": [
        "calendario academico", "quando comecam as aulas", "quando comeca o semestre",
        "datas importantes", "quando termina o semestre", "feriados e recessos",
        "inicio das aulas",
    ],
    "fies": [
        "o que e o fies", "como funciona o fies", "quero o fies", "como contratar o fies",
        "requisitos do fies", "financiamento estudantil fies", "quem pode aderir ao fies",
    ],
    "formatura": [
        "como funciona a formatura", "colacao de grau", "tipos de formatura", "quero me formar",
        "como participar da colacao", "formatura", "quando e a colacao de grau",
    ],
}

# Temas-semente (sem dados reais raspados) que vale manter
MANTER_SEED = {
    "cursos", "mensalidade_valores", "biblioteca", "campus_localizacao",
    "ead", "pos_graduacao", "matricula", "secretaria_documentos",
    "horario_atendimento", "estagio_carreira",
}
CONVERSACIONAIS = {"saudacao", "despedida", "agradecimento", "fallback"}


def tema_da_fonte(fonte: str) -> str:
    f = fonte.lower()
    for chave, tema in FONTE_TEMA.items():
        if chave in f:
            return tema
    return "ingresso"


def limpar_resposta(q: str, a: str) -> str:
    """Remove repeticao da pergunta no inicio da resposta."""
    if a.lower().startswith(q.lower()[:25]):
        # corta ate o primeiro '?' (fim da pergunta repetida)
        idx = a.find("?")
        if 0 < idx < len(a) - 10:
            a = a[idx + 1:].strip()
    return a


def main():
    faq = json.loads(FAQ.read_text(encoding="utf-8"))
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    seed_by_tag = {it["tag"]: it for it in seed["intents"]}

    # agrupa perguntas reais por tema
    temas: dict[str, list[str]] = {}
    relatorio: dict[str, int] = {}
    for it in faq["intents"]:
        if not it["tag"].startswith("faq_"):
            continue
        if it["tag"] in DESCARTAR:
            continue
        q = it["patterns"][0].strip()
        a = limpar_resposta(q, it["responses"][0].strip())
        if len(a) < 40:
            continue
        fonte = it.get("_fonte", "")
        tema = tema_da_fonte(fonte)
        # caso especial: senha / intranet
        if re.search(r"senha|intranet", q, re.I):
            tema = "portal_senha"
        temas.setdefault(tema, []).append(q)
        relatorio[tema] = relatorio.get(tema, 0) + 1

    # adiciona patterns curados (no inicio, p/ garantir presenca mesmo apos o cap)
    for tema, extras in EXTRA_PATTERNS.items():
        temas.setdefault(tema, [])
        temas[tema] = extras + temas[tema]

    intents_out = []

    # 1) conversacionais (do seed)
    for tag in CONVERSACIONAIS:
        if tag in seed_by_tag:
            intents_out.append(seed_by_tag[tag])

    # 2) temas com dados reais
    for tema, perguntas in sorted(temas.items()):
        # dedup e limite de patterns
        vistos, pats = set(), []
        for p in perguntas:
            k = p.lower()
            if k not in vistos:
                vistos.add(k)
                pats.append(p)
            if len(pats) >= MAX_PATTERNS:
                break
        responses = RESPOSTAS_TOPICO.get(tema)
        if responses is None and tema in seed_by_tag:
            responses = seed_by_tag[tema]["responses"]
        if responses is None:
            responses = [f"Consulte a secretaria da UNIVALI: {TEL} ou https://www.univali.br."]
        intents_out.append({"tag": tema, "patterns": pats, "responses": responses})

    # 3) temas-semente sem dados reais (respostas curadas substituem [VERIFICAR])
    for tag in MANTER_SEED:
        if tag in temas:  # ja coberto por dados reais
            continue
        if tag in seed_by_tag:
            it = dict(seed_by_tag[tag])
            if tag in RESPOSTAS_TOPICO:
                it["responses"] = RESPOSTAS_TOPICO[tag]
            intents_out.append(it)

    saida = {
        "_meta": {
            "descricao": "Intencoes tematicas da UNIVALI (modelo fiel/BoW). Patterns vindos de FAQs "
                         "reais raspadas + temas-semente; respostas de topico curadas.",
            "idioma": "pt-BR",
            "versao": "1.0-mapeado",
        },
        "intents": intents_out,
    }
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Mapeamento por tema (FAQs reais atribuidas):")
    for tema, n in sorted(relatorio.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {tema}")
    print(f"\nTotal de intencoes no arquivo final: {len(intents_out)}")
    print(f"Descartadas (ruido): {sorted(DESCARTAR)}")
    print(f"Arquivo salvo: {SAIDA}")


if __name__ == "__main__":
    main()
