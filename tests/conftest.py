"""Configuration commune des tests.

Les tests unitaires et d'integration n'appellent jamais de service externe (modele de langage,
Soynade, Redis) et n'ecrivent jamais dans la vraie base d'analytique : tout est redirige vers
un dossier temporaire ou remplace par des doublures.
"""

import os
import tempfile
from pathlib import Path

# Avant tout import des modules de l'application : cache en memoire, pas de vraie cle.
os.environ.setdefault("REDIS_URL", "")
os.environ["REDIS_URL"] = ""
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

import analytics  # noqa: E402

# Journal d'utilisation des tests dans un fichier jetable, jamais dans storage/.
_TMP = Path(tempfile.mkdtemp(prefix="ansd-tests-"))
analytics.DB_PATH = _TMP / "analytics.sqlite3"
analytics._conn = None
