import argparse
import csv
import time
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup

from config import BASE_URL, CATEGORIES, DOC_TYPES, LISTING_PATH, MANIFEST_PATH

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ansd-rag-bot/1.0)"}

# Le serveur ansd.sn ne renvoie pas la chaine de certificats intermediaire
# (erreur "unable to get local issuer certificate"). Les navigateurs la
# recuperent automatiquement (AIA fetching) mais `requests` non : on
# desactive donc la verification SSL pour ce site public en lecture seule.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY_SSL = False


def fetch_page(session: requests.Session, page: int, category: str | None, doc_type: str | None) -> BeautifulSoup:
    params = {"page": page, "term_node_tid_depth": category or "All"}
    if doc_type:
        params["field_types_de_document_value"] = doc_type
    resp = session.get(urljoin(BASE_URL, LISTING_PATH), params=params, timeout=30, verify=VERIFY_SSL)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_rows(soup: BeautifulSoup) -> list[dict]:
    table = soup.select_one("table.table")
    if not table:
        return []
    rows = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        link = cells[3].find("a")
        if not link or not link.get("href"):
            continue
        rows.append({
            "title": cells[0].get_text(strip=True),
            "extension": cells[1].get_text(strip=True).lower(),
            "size": cells[2].get_text(strip=True),
            "url": urljoin(BASE_URL, link["href"]),
        })
    return rows


def has_next_page(soup: BeautifulSoup) -> bool:
    return soup.select_one("li.pager__item--next a") is not None


def build_catalog(max_docs: int, category: str | None, doc_type: str | None, delay: float) -> list[dict]:
    """Parcourt le catalogue de l'ANSD et note titre/URL des PDF, sans les telecharger."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    manifest = []
    seen_urls = set()
    page = 0

    while len(manifest) < max_docs:
        soup = fetch_page(session, page, category, doc_type)
        rows = parse_rows(soup)
        if not rows:
            break

        for row in rows:
            if row["extension"] != "pdf" or row["url"] in seen_urls:
                continue

            seen_urls.add(row["url"])
            manifest.append(row)
            print(f"[{len(manifest)}/{max_docs}] {row['title']}")

            if len(manifest) >= max_docs:
                break

        if not has_next_page(soup):
            break
        page += 1
        time.sleep(delay)

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "extension", "size", "url"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\n{len(manifest)} documents references dans {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Catalogue les publications PDF de l'ANSD (titre + URL, sans telechargement)"
    )
    parser.add_argument("--max-docs", type=int, default=100, help="Nombre maximum de documents a referencer")
    parser.add_argument("--category", choices=list(CATEGORIES), default=None)
    parser.add_argument("--type", dest="doc_type", choices=list(DOC_TYPES), default=None)
    parser.add_argument("--delay", type=float, default=0.3, help="Delai (s) entre deux pages du catalogue")
    args = parser.parse_args()

    build_catalog(
        max_docs=args.max_docs,
        category=CATEGORIES.get(args.category) if args.category else None,
        doc_type=DOC_TYPES.get(args.doc_type) if args.doc_type else None,
        delay=args.delay,
    )
