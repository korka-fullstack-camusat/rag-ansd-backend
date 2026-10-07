from pathlib import Path

BASE_URL = "https://www.ansd.sn"
LISTING_PATH = "/toutes-les-publications"

CATEGORIES = {
    "economie": "1",
    "demographie": "2",
    "societe": "3",
    "autres": "39",
}

DOC_TYPES = {
    "rapports": "1",
    "donnees": "2",
    "reference": "3",
}

ROOT_DIR = Path(__file__).parent
MANIFEST_PATH = ROOT_DIR / "data" / "manifest.csv"
CHROMA_DIR = ROOT_DIR / "storage" / "chroma"
COLLECTION_NAME = "ansd_publications"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 6
