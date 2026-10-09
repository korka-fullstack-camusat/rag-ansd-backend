import argparse
import csv
import uuid

import chromadb
import fitz
import requests
import urllib3
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ansd-rag-bot/1.0)"}
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY_SSL = False


def read_manifest() -> list[dict]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_pdf_bytes(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=60, verify=VERIFY_SSL)
    resp.raise_for_status()
    return resp.content


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((i, text))
    return pages


def build_chunks(pdf_bytes: bytes, title: str, splitter) -> list[dict]:
    chunks = []
    for page_num, text in extract_pages(pdf_bytes):
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "source": title, "page": page_num})
    return chunks


def ingest(reset: bool) -> None:
    if not MANIFEST_PATH.exists():
        print(f"{MANIFEST_PATH} introuvable. Lance d'abord scraper.py.")
        return

    rows = read_manifest()
    if not rows:
        print("Le manifeste est vide. Lance d'abord scraper.py.")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    already_indexed = {
        meta["url"] for meta in collection.get(include=["metadatas"])["metadatas"]
    } if collection.count() > 0 else set()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"Chargement du modele d'embedding {EMBEDDING_MODEL}...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

    session = requests.Session()
    session.headers.update(HEADERS)

    for row in tqdm(rows, desc="Documents"):
        if row["url"] in already_indexed:
            continue

        try:
            pdf_bytes = fetch_pdf_bytes(session, row["url"])
            chunks = build_chunks(pdf_bytes, row["title"], splitter)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"[WARN] echec traitement {row['title']}: {exc}")
            continue
        finally:
            pdf_bytes = None  # ne jamais persister le contenu du PDF sur disque

        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = [vec.tolist() for vec in embedder.embed(texts)]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": c["source"], "page": c["page"], "url": row["url"]} for c in chunks
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"\nIndexation terminee. {collection.count()} fragments dans la collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrait (a la volee, sans stockage disque), decoupe et vectorise les PDF de l'ANSD"
    )
    parser.add_argument("--reset", action="store_true", help="Recree la collection depuis zero")
    args = parser.parse_args()
    ingest(reset=args.reset)
