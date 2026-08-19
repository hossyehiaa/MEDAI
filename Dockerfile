FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY server/requirements.txt ./server/requirements.txt
RUN pip install -r server/requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY . .
EXPOSE 10000
CMD ["sh","-c","uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
