"""
Scraper das paginas INDIVIDUAIS de cada curso de graduacao da UNIVALI.

Diferente do scrape_univali.py (que pega FAQs), este extrai dados ESTRUTURADOS
de cada curso (campus, turno, duracao, carga horaria, conceito MEC, ENADE e
objetivo), para que o chatbot responda perguntas especificas tipo
"como e o curso de Ciencia da Computacao?" em vez de algo generico.

Fluxo:
  1. Renderiza https://portal.univali.br/graduacao (SPA) e coleta os links
     /graduacao/<curso>-<escola>-<campus>-<grau>.
  2. Renderiza cada pagina de curso e extrai os campos pelo padrao "ROTULO\\nvalor".
  3. Salva data/cursos_univali.json (uma entrada por curso+campus).

Uso:
    python src/scraper/scrape_cursos.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAIDA = ROOT / "data" / "cursos_univali.json"

BASE = "https://portal.univali.br"
GRAD = f"{BASE}/graduacao"
UA = "UNIVALI-FAQ-Bot-Academico/1.0 (projeto educacional)"

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


def coletar_links(page) -> list[str]:
    page.goto(GRAD, wait_until="networkidle", timeout=40000)
    page.wait_for_timeout(3000)
    for _ in range(8):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(400)
    hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.getAttribute('href'))")
    cursos = sorted(
        {h for h in hrefs if h and "/graduacao/" in h and h.count("-") >= 2}
    )
    return [h if h.startswith("http") else BASE + h for h in cursos]


def extrair_curso(page, url: str) -> dict | None:
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

    if not nome or "campus" not in dados:
        return None
    return {
        "curso": nome,
        "grau": grau or "",
        "campus": dados.get("campus", ""),
        "modalidade": dados.get("modalidade", ""),
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

    registros = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)

        links = coletar_links(page)
        print(f"Encontrados {len(links)} cursos (curso+campus). Extraindo...")
        for i, url in enumerate(links, 1):
            try:
                rec = extrair_curso(page, url)
                if rec:
                    registros.append(rec)
                    print(f"  [{i:2d}/{len(links)}] {rec['curso']} - {rec['campus']} "
                          f"({rec['turno']}, {rec['duracao']})")
                else:
                    print(f"  [{i:2d}/{len(links)}] (sem dados) {url.split('/')[-1]}")
            except Exception as e:
                print(f"  [{i:2d}/{len(links)}] ERRO {url.split('/')[-1]}: {e}")
            time.sleep(0.5)
        browser.close()

    SAIDA.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvos {len(registros)} cursos em {SAIDA}")


if __name__ == "__main__":
    main()
