"""Наивный RAG: поиск по векторной базе, затем ответ модели по фрагментам."""
import hashlib
from typing import Any, Dict, List

import config
from experiments import Answer, register_pipeline
import prompts
from experiments.naive_rag.index import get_store
from llm import create_llm


@register_pipeline("naive_rag")
class NaiveRagPipeline:
    """Векторный поиск (e5-small + Chroma) и генерация без реранкера."""

    ANSWER_PREFIX = "Ответ"

    def __init__(self, model: str):
        if model not in config.MODELS:
            raise ValueError(f"Неизвестная модель: {model}")
        self.llm = create_llm(model)
        cfg = config.MODELS[model]
        self.model = model
        self.requested_model = cfg["api_model"]
        self.answer_col = f"{self.ANSWER_PREFIX}_{cfg['label']}_NaiveRAG"
        self.retrieval_col = f"Найденные статьи_{cfg['label']}_NaiveRAG"
        self.system_prompt = prompts.RAG_SYSTEM_PROMPT
        self.prompt_version = prompts.RAG_PROMPT_VERSION
        # Условия ретривера: попадают в метаданные и сравниваются при resume.
        self.index_conditions = {
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_revision": config.EMBEDDING_REVISION,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "index_version": config.INDEX_VERSION,
            "corpus_sha256": hashlib.sha256(config.CORPUS_PATH.read_bytes()).hexdigest(),
        }
        self.store = get_store()

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Ищем больше top_k и дедуплицируем по статье: куски одной статьи
        не должны вытеснять другие статьи из контекста."""
        candidates = config.TOP_K * 3
        # Префикс query: добавляет E5Embeddings — здесь передаём голый вопрос.
        hits = self.store.similarity_search_with_relevance_scores(
            question, k=candidates
        )
        fragments: List[Dict[str, Any]] = []
        seen_articles: set = set()
        for doc, score in hits:
            article_id = doc.metadata["article_id"]
            if article_id in seen_articles:
                continue
            seen_articles.add(article_id)
            fragments.append(
                {
                    "article_id": article_id,
                    "title": doc.metadata["title"],
                    "chunk_index": doc.metadata["chunk_index"],
                    "text": doc.page_content,
                    "score": round(float(score), 4),
                }
            )
            if len(fragments) == config.TOP_K:
                break
        return fragments

    @staticmethod
    def retrieval_summary(fragments: List[Dict[str, Any]]) -> str:
        # "tk-81 (0.87); tk-80 (0.84)" — номера статей для оценки recall@k.
        return "; ".join(f"{f['article_id']} ({f['score']:.2f})" for f in fragments)

    def answer(self, question: str) -> Answer:
        fragments = self.retrieve(question)
        prompt = prompts.format_rag_prompt(question, fragments)
        text = self.llm.generate_response(
            prompt=prompt, system_prompt=prompts.RAG_SYSTEM_PROMPT
        )
        return Answer(
            text=text,
            docs=fragments,
            meta={
                "response_model": self.llm.last_model,
                "system_fingerprint": self.llm.last_fingerprint,
                "usage": self.llm.last_usage,
                "retrieval": self.retrieval_summary(fragments),
                "fragments": fragments,
            },
        )
