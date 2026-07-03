"""
Scraper das paginas INDIVIDUAIS de cada curso de graduacao da UNIVALI.

Cobre DUAS fontes (sites separados):
  1. Portal presencial : https://portal.univali.br/graduacao
  2. EAD (a distancia) : https://ead.univali.br/cursos-graduacao

Extrai dados ESTRUTURADOS de cada curso (campus, turno, duracao, carga
horaria, conceito MEC, ENADE e objetivo) para o chatbot responder perguntas
especificas tipo "como e o curso de Ciencia da Computacao?".

Fluxo:
  1. Para cada fonte, renderiza a listagem (SPA) e coleta os links de curso.
  2. Renderiza cada pagina de curso e extrai os campos pelo padrao "ROTULO\\nvalor".
  3. Salva data/cursos_univali.json (uma entrada por curso+campus), sem duplicar.

Obs.: as paginas EAD tem estrutura diferente das presenciais; por isso a
extracao usa defaults por fonte (modalidade/campus) e fallback de nome pelo
<h1>/slug quando os rotulos nao batem. Ajuste os SOURCES se o site mudar.

Uso:
    python src/scraper/scrape_cursos.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAIDA = ROOT / "data" / "cursos_univali.json"

UA = "UNIVALI-FAQ-Bot-Academico/1.0 (projeto educacional)"

# Fontes de dados: cada uma tem a listagem, o trecho que identifica um link de
# curso, e os defaults aplicados quando a pagina nao traz o campo.
SOURCES = [
    {
        "nome": "presencial",
        "listagem": "https://portal.univali.br/graduacao",
        "marcador": "/graduacao/",          # href que representa um curso
        "min_hifens": 2,                     # /graduacao/<curso>-<escola>-<campus>-<grau>
        "default_modalidade": "",            # a pagina presencial traz MODALIDADE
        "default_campus": "",
    },
    {
        "nome": "ead",
        "listagem": "https://ead.univali.br/cursos-graduacao",
        "marcador": "/cursos-graduacao/",    # href de curso EAD (…/<slug>-ead)
        "min_hifens": 1,
        "default_modalidade": "EAD",
        "default_campus": "EAD (a distância)",
    },
]

# rotulo na pagina -> chave no nosso JSON
ROTULOS = {
    "CONCEITO MEC": "conceito_mec",
    "NOTA DO ENADE": "nota_enade",
    "CAMPUS": "campus",
    "MODALIDADE": "modalidade",
    "TURNO": "turno",
    "DURAÇÃO": "duracao",
    "CARGA HORÁRIA": "carga_horaria",
}
GRAUS = {"BACHARELADO", "LICENCIATURA", "TECNÓLOGO", "TECNOLOGO"}


def coletar_links(page, source: dict) -> list[str]:
    page.goto(source["listagem"], wait_until="networkidle", timeout=40000)
    page.wait_for_timeout(3000)
    for _ in range(8):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(400)
    base = re.match(r"https?://[^/]+", source["listagem"]).group(0)
    marcador = source["marcador"]
    hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.getAttribute('href'))")
    cursos = set()
    for h in hrefs:
        if not h or marcador not in h:
            continue
        # ignora a propria listagem (sem slug depois do marcador)
        depois = h.split(marcador, 1)[1].strip("/")
        if not depois or "/" in depois:
            continue
        if depois.count("-") < source["min_hifens"]:
            continue
        cursos.add(h if h.startswith("http") else base + h)
    return sorted(cursos)


def _nome_fallback(page, url: str, source: dict) -> str:
    """Nome do curso quando os GRAUS nao aparecem (paginas EAD)."""
    try:
        h1 = page.inner_text("h1").strip()
        if h1:
            return re.sub(r"\s*EAD\s*$", "", h1, flags=re.I).strip()
    except Exception:
        pass
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"-ead$", "", slug)
    return slug.replace("-", " ").strip().title()


def extrair_curso(page, url: str, source: dict) -> dict | None:
    # domcontentloaded e bem mais rapido que networkidle em SPA (que pode travar)
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector("text=DURAÇÃO", timeout=8000)
    except Exception:
        page.wait_for_timeout(2500)
    txt = page.inner_text("body")
    linhas = [l.strip() for l in txt.splitlines() if l.strip()]

    dados: dict[str, str] = {}
    nome = None
    grau = None
    for i, ln in enumerate(linhas):
        up = ln.upper()
        if up in GRAUS and nome is None and i + 1 < len(linhas):
            grau = ln.capitalize()
            nome = linhas[i + 1]
        if up in ROTULOS and i + 1 < len(linhas):
            chave = ROTULOS[up]
            if chave not in dados:
                dados[chave] = linhas[i + 1]

    # objetivo do curso (texto longo, opcional)
    objetivo = ""
    for i, ln in enumerate(linhas):
        if ln.lower().startswith("objetivo do curso") and i + 1 < len(linhas):
            objetivo = linhas[i + 1][:400]
            break

    if not nome:
        nome = _nome_fallback(page, url, source)
    # remove o sufixo " EAD" do nome (paginas EAD trazem "Administracao EAD")
    # para que o curso funda com a versao presencial no chatbot.
    nome = re.sub(r"\s+EAD\s*$", "", nome or "", flags=re.I).strip()
    campus = dados.get("campus") or source["default_campus"]
    modalidade = dados.get("modalidade") or source["default_modalidade"]

    if not nome or not campus:
        return None
    return {
        "curso": nome,
        "grau": grau or "",
        "campus": campus,
        "modalidade": modalidade,
        "turno": dados.get("turno", ""),
        "duracao": dados.get("duracao", ""),
        "carga_horaria": dados.get("carga_horaria", ""),
        "conceito_mec": dados.get("conceito_mec", ""),
        "nota_enade": dados.get("nota_enade", ""),
        "objetivo": objetivo,
        "url": url,
    }


def main():
    from playwright.sync_api import sync_playwright

    registros: list[dict] = []
    vistos: set[tuple[str, str]] = set()  # (curso, campus) para nao duplicar
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)

        for source in SOURCES:
            links = coletar_links(page, source)
            print(f"[{source['nome']}] {len(links)} cursos em {source['listagem']}")
            for i, url in enumerate(links, 1):
                try:
                    rec = extrair_curso(page, url, source)
                    if not rec:
                        print(f"  [{i:2d}/{len(links)}] (sem dados) {url.split('/')[-1]}")
                        continue
                    chave = (rec["curso"].strip(), rec["campus"].strip())
                    if chave in vistos:
                        continue
                    vistos.add(chave)
                    registros.append(rec)
                    print(f"  [{i:2d}/{len(links)}] {rec['curso']} - {rec['campus']} "
                          f"({rec['modalidade']}, {rec['duracao']})")
                except Exception as e:
                    print(f"  [{i:2d}/{len(links)}] ERRO {url.split('/')[-1]}: {e}")
                time.sleep(0.5)
        browser.close()

    SAIDA.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvos {len(registros)} cursos em {SAIDA}")


if __name__ == "__main__":
    main()
