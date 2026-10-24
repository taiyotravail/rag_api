FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modèle d'embeddings embarqué dans l'image : aucun téléchargement au démarrage.
ENV HF_HOME=/models
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small')"

COPY main.py rag.py ./

ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
