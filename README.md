# API RAG Plannea

API FastAPI exposant un pipeline RAG : récupération de chunks dans un index Supabase (pgvector) avec des embeddings `intfloat/multilingual-e5-small`, puis génération de la réponse avec Mistral.

## Prérequis

- Python 3.12
- Une base Supabase contenant l'index vectoriel
- Une clé API Mistral (facultative, uniquement pour `/query`)

## Configuration

Copier `.env.example` vers `.env` et renseigner :

| Variable | Description |
|---|---|
| `SUPABASE_DB_URL` | URL de connexion Postgres de Supabase (obligatoire) |
| `MISTRAL_API_KEY` | Clé API Mistral (facultative) : sans elle, `/query` répond `503` |

## Lancer en local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

L'API est disponible sur http://127.0.0.1:8000 (documentation interactive sur `/docs`).

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### `POST /query`

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Votre question", "top_k": 3}'
```

- `question` : obligatoire, 1 à 2000 caractères
- `top_k` : facultatif, de 1 à 10 (3 par défaut)

Réponse :

```json
{
  "answer": "…",
  "sources": [{"file_name": "…", "score": 0.82, "text": "…"}]
}
```

Codes d'erreur : `422` (requête invalide), `502` (erreur de Mistral), `503` (`MISTRAL_API_KEY` non configurée).

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t api-rag .
docker run --rm -p 8080:8080 --env-file .env api-rag
```

Le modèle d'embeddings est embarqué dans l'image au build : pas de téléchargement au démarrage.

## Déploiement sur Google Cloud Run

Créer les secrets dans Secret Manager, puis :

```bash
gcloud run deploy api-rag --source . --region europe-west1 \
  --memory 2Gi --cpu-boost --min-instances 1 \
  --set-secrets MISTRAL_API_KEY=mistral-key:latest,SUPABASE_DB_URL=supabase-url:latest
```
