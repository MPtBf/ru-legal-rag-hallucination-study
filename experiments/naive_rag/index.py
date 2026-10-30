"""Индекс наивного RAG: чанки корпуса и Chroma на multilingual-e5-small."""
import hashlib
import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)

# Имя коллекции = каталог data/index/: смена любой настройки даёт новый индекс.
COLLECTION = f"naive_rag_e5small_{config.INDEX_VERSION}"


def index_fingerprint() -> str:
    """Fingerprint all inputs that determine persisted vectors and chunk IDs."""
    digest = hashlib.sha256()
    with open(config.CORPUS_PATH, "rb") as corpus:
        for chunk in iter(lambda: corpus.read(1024 * 1024), b""):
            digest.update(chunk)
    settings = {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_revision": config.EMBEDDING_REVISION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "index_version": config.INDEX_VERSION,
        "index_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


class E5Embeddings(Embeddings):
    """Обёртка над e5: без префиксов query:/passage: качество заметно падает."""

    def __init__(self, base: Embeddings):
        self.base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents(["passage: " + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query("query: " + text)


def _load_corpus() -> list[dict]:
    rows = []
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Нарезка по строкам с упаковкой до size; сверхдлинные строки режем вручную."""
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE")

    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        while len(line) > size:
            segments.append(line[:size])
            line = line[size - overlap:]
        if line:
            segments.append(line)

    chunks = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n{segment}" if current else segment
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Перекрытие хвостом прошлого чанка: поиск не теряет границу.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{segment}" if tail else segment
    if current:
        chunks.append(current)
    return chunks


def build_chunks() -> list[Document]:
    """Чанк корпуса: заголовок статьи + кусок текста, метаданные для оценки."""
    docs = []
    for row in _load_corpus():
        title = row["title"].strip()
        parts = split_text(row["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, part in enumerate(parts):
            docs.append(
                Document(
                    # Заголовок в каждом чанке: иначе середина длинной статьи
                    # не находится по номеру и названию статьи.
                    page_content=f"{title}\n{part}",
                    metadata={
                        "article_id": row["_id"],
                        "title": title,
                        "chunk_index": i,
                        "chunk_count": len(parts),
                    },
                )
            )
    return docs


def _open_store(fingerprint: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=E5Embeddings(
            HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={"revision": config.EMBEDDING_REVISION},
                # Нормализация нужна, чтобы косинус в Chroma считался корректно.
                encode_kwargs={"normalize_embeddings": True},
            )
        ),
        persist_directory=str(config.INDEX_DIR / COLLECTION),
        # Косинус, а не L2 по умолчанию: оценки сравнимы между прогонами.
        collection_metadata={"hnsw:space": "cosine", "index_fingerprint": fingerprint},
    )


def get_store() -> Chroma:
    """Открывает индекс; при расхождении с корпусом пересобирает."""
    fingerprint = index_fingerprint()
    store = _open_store(fingerprint)
    docs = build_chunks()
    actual = store._collection.count()
    actual_fingerprint = (store._collection.metadata or {}).get("index_fingerprint")
    if actual != len(docs) or actual_fingerprint != fingerprint:
        if actual:
            logger.warning(
                "Индекс устарел (чанков %d/%d, fingerprint совпадает: %s) — пересборка",
                actual, len(docs), actual_fingerprint == fingerprint,
            )
        store.delete_collection()
        store = _open_store(fingerprint)
        ids = [
            f"{doc.metadata['article_id']}#{doc.metadata['chunk_index']}" for doc in docs
        ]
        logger.info("Индексация: %d чанков в %s", len(docs), COLLECTION)
        # Батчами: одна отправка ~1400 документов не помещается в память энкодера.
        for start in range(0, len(docs), 64):
            store.add_documents(
                docs[start : start + 64], ids=ids[start : start + 64]
            )
    return store
