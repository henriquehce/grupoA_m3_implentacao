"""
Scraper focado das paginas publicas da UNIVALI.

Estrategia (hibrida, como combinado):
  1. Le uma lista de URLs-semente (SEED_URLS) — paginas de FAQ/cursos/atendimento.
  2. Respeita o robots.txt e usa intervalo entre requisicoes (educado).
  3. Extrai blocos de texto e, quando encontra padrao de FAQ (pergunta/resposta),
     monta pares candidatos.
  4. Salva:
       - data/raw/<slug>.txt              (texto bruto, para auditoria)
       - data/intents_univali_draft.json  (rascunho de intents para CURADORIA)

IMPORTANTE: a saida e um RASCUNHO. As respostas devem ser revisadas por uma
pessoa antes de irem para data/intents_univali.json (precisao factual).

Uso:
    python src/scraper/scrape_univali.py
    python src/scraper/scrape_univali.py --url https://www.univali.br/pagina
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.robotparser as robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DRAFT = ROOT / "data" / "intents_univali_draft.json"

HEADERS = {
    "User-Agent": "UNIVALI-FAQ-Bot-Academico/1.0 (projeto educacional; contato via universidade)"
}
DELAY_SEGUNDOS = 2.0  # intervalo educado entre requisicoes

# Paginas-semente da UNIVALI (FAQs e paginas de servico).
SEED_URLS = [
    "https://portal.univali.br/perguntas-frequentes",
    "https://www.univali.br/intercambio/perguntas-frequentes/Paginas/default.aspx",
    "https://portal.univali.br/fale-conosco",
    "https://portal.univali.br/reingresso/perguntas-frequentes",
    "https://ead.univali.br/central-de-ajuda",
    "https://portal.univali.br/certidao-de-estudos-externa/perguntas-frequentes",
    "https://portal.univali.br/universidade-gratuita/perguntas-frequentes",
    "https://portal.univali.br/",
    "https://www.univali.br/vida-no-campus/auxilio-permanencia/perguntas-frequentes/Paginas/default.aspx",
    "https://www.univali.br/vida-no-campus/moradia/Paginas/default.aspx",
    "https://www.univali.br/vida-no-campus/transporte/Paginas/default.aspx",
    "https://portal.univali.br/bolsas",
    "https://portal.univali.br/auxilio-permanencia",
    "https://portal.univali.br/cursos-livres",
    "https://portal.univali.br/graduacao",
    "https://portal.univali.br/formas-de-ingresso",
    "https://portal.univali.br/transferencia-externa",
    "https://portal.univali.br/bolsas?categories=4",
    "https://portal.univali.br/pos",
    # paginas de FAQ adicionais (descobertas via busca)
    "https://portal.univali.br/programa-uniedu/perguntas-frequentes",
    "https://portal.univali.br/transferencia-externa/perguntas-frequentes",
    "https://portal.univali.br/segunda-graduacao",
    # palpites de URL no padrao /<tema>/perguntas-frequentes (404 e ignorado)
    "https://portal.univali.br/segunda-graduacao/perguntas-frequentes",
    "https://portal.univali.br/formas-de-ingresso/perguntas-frequentes",
    "https://portal.univali.br/seletivo-univali/perguntas-frequentes",
    "https://portal.univali.br/vagas-remanescentes/perguntas-frequentes",
    "https://portal.univali.br/fies/perguntas-frequentes",
    "https://portal.univali.br/prouni/perguntas-frequentes",
]


def slugify(url: str) -> str:
    s = urlparse(url).path.strip("/") or "home"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "home"


def pode_acessar(url: str) -> bool:
    """Consulta o robots.txt do dominio."""
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        # Se nao conseguir ler o robots, seja conservador mas permita paginas publicas
        return True


def baixar_requests(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        # retorna bytes crus: deixamos o BeautifulSoup detectar o charset
        # (evita mojibake quando o servidor nao declara o encoding correto)
        return r.content
    except requests.RequestException as e:
        print(f"[erro] {url}: {e}")
        return None


def baixar_navegador(url: str) -> bytes | None:
    """Renderiza a pagina com navegador headless (paginas que usam JavaScript)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [aviso] playwright nao instalado; pulando renderizacao JS")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            browser.close()
            return html.encode("utf-8")
    except Exception as e:
        print(f"  [erro navegador] {url}: {e}")
        return None


def baixar(url: str, usar_js: bool = True) -> bytes | None:
    if not pode_acessar(url):
        print(f"[robots] bloqueado: {url}")
        return None
    html = baixar_requests(url)
    # Se o HTML estatico for pequeno (pagina renderizada via JS), tenta o navegador
    if usar_js and (html is None or len(extrair_faq(html)) == 0):
        render = baixar_navegador(url)
        if render is not None:
            html = render
    return html


def extrair_texto(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    texto = soup.get_text(separator="\n")
    linhas = [ln.strip() for ln in texto.splitlines() if len(ln.strip()) > 2]
    return "\n".join(linhas)


def extrair_faq(html: bytes) -> list[tuple[str, str]]:
    """
    Heuristica: procura padrao de FAQ comum (accordion / dt-dd / pergunta '?'
    seguida de paragrafo). Retorna pares (pergunta, resposta) candidatos.
    """
    soup = BeautifulSoup(html, "lxml")
    pares: list[tuple[str, str]] = []

    # 1) <dt>/<dd>
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            q = dt.get_text(strip=True)
            a = dd.get_text(" ", strip=True)
            if q and a:
                pares.append((q, a))

    # 2) cabecalho terminando em '?' seguido de texto
    for h in soup.find_all(["h2", "h3", "h4", "summary", "button"]):
        q = h.get_text(strip=True)
        if q.endswith("?") and 5 < len(q) < 200:
            nxt = h.find_next(["p", "div", "span"])
            if nxt:
                a = nxt.get_text(" ", strip=True)
                if 10 < len(a) < 800:
                    pares.append((q, a))

    return _dedup(pares)


# Marcadores de rodape/menu: a resposta termina ao encontrar um deles
_PARAR = (
    "redes sociais", "fale conosco", "central de atendimento", "copyright",
    "politica de", "política de", "tenho interesse", "inscreva-se",
    "para reclamacoes", "para reclamações", "todos os direitos",
)


def _eh_pergunta(linha: str) -> bool:
    linha = linha.strip()
    return linha.endswith("?") and 15 <= len(linha) <= 200


def extrair_faq_texto(texto: str) -> list[tuple[str, str]]:
    """
    Fallback baseado em texto puro (para FAQs renderizadas via JS, ex. SharePoint):
    uma linha que e pergunta ('...?') seguida das linhas de resposta ate a
    proxima pergunta ou um marcador de rodape.
    """
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    pares: list[tuple[str, str]] = []
    i = 0
    while i < len(linhas):
        if _eh_pergunta(linhas[i]):
            q = linhas[i]
            resp: list[str] = []
            j = i + 1
            while j < len(linhas):
                cur = linhas[j]
                if _eh_pergunta(cur):
                    break
                if any(m in cur.lower() for m in _PARAR):
                    break
                resp.append(cur)
                if sum(len(x) for x in resp) > 1200:
                    break
                j += 1
            resposta = " ".join(resp).strip()
            if len(resposta) > 30:  # ignora perguntas sem resposta de verdade
                pares.append((q, resposta))
            i = j
        else:
            i += 1
    return _dedup(pares)


def _dedup(pares: list[tuple[str, str]]) -> list[tuple[str, str]]:
    vistos = set()
    unicos = []
    for q, a in pares:
        chave = q.lower().strip()
        if chave not in vistos:
            vistos.add(chave)
            unicos.append((q, a))
    return unicos


def main(urls: list[str]):
    draft_intents = []
    for url in urls:
        print(f"[baixando] {url}")
        html = baixar(url)
        if not html:
            continue
        slug = slugify(url)

        texto = extrair_texto(html)
        (RAW / f"{slug}.txt").write_text(texto, encoding="utf-8")

        # Combina extracao por DOM (tags) + por texto puro (FAQs renderizadas via JS)
        faqs = _dedup(extrair_faq(html) + extrair_faq_texto(texto))
        print(f"  -> {len(faqs)} pares FAQ candidatos; texto bruto salvo em raw/{slug}.txt")
        for i, (q, a) in enumerate(faqs):
            draft_intents.append(
                {
                    "tag": f"{slug}_{i}",
                    "patterns": [q],
                    "responses": [a],
                    "_fonte": url,
                    "_status": "RASCUNHO - revisar antes de usar",
                }
            )
        time.sleep(DELAY_SEGUNDOS)

    DRAFT.write_text(
        json.dumps({"intents": draft_intents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRascunho com {len(draft_intents)} intents salvo em {DRAFT}")
    print("Proximo passo: revisar manualmente e mesclar em data/intents_univali.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", help="URL especifica (pode repetir)")
    args = ap.parse_args()
    main(args.url if args.url else SEED_URLS)
