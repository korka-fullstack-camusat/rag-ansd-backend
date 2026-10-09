#!/usr/bin/env sh
# Lance les tests du backend dans l'image Docker (memes dependances qu'en production).
#
#   ./run_tests.sh              tests unitaires + integration (rapides, sans reseau ni cout)
#   ./run_tests.sh unit         tests unitaires seulement
#   ./run_tests.sh integration  tests d'integration de l'API seulement
#   ./run_tests.sh e2e          tests de bout en bout contre la plateforme lancee
#                               (docker compose up -d dans le backend ET le frontend) ;
#                               vrais appels au modele : quelques centimes, ~2 minutes
#   ./run_tests.sh all          tout
# Options pytest supplementaires apres le mode, ex. : ./run_tests.sh unit -k cache -x
set -e
cd "$(dirname "$0")"
MODE="${1:-default}"
[ $# -gt 0 ] && shift

case "$MODE" in
  unit) TARGET="tests/unit" ;;
  integration) TARGET="tests/integration" ;;
  e2e) TARGET="tests/e2e -m e2e" ;;
  all) TARGET="tests -m 'e2e or not e2e'" ;;
  default) TARGET="tests/unit tests/integration" ;;
  *) echo "Mode inconnu : $MODE (unit | integration | e2e | all)"; exit 2 ;;
esac

# Le code est monte tel quel (pas besoin de reconstruire l'image apres une modification).
# Sur le reseau du docker compose : les tests e2e joignent l'API par http://backend:8000
# et le frontend par la machine hote.
docker compose run --rm --no-deps -T \
  -v "$PWD:/src" -w /src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e E2E_API_URL="${E2E_API_URL:-http://backend:8000}" \
  -e E2E_FRONTEND_URL="${E2E_FRONTEND_URL:-http://host.docker.internal:3000}" \
  backend sh -c "pip install -q --disable-pip-version-check --root-user-action=ignore -r requirements-dev.txt && python -m pytest $TARGET $*"
