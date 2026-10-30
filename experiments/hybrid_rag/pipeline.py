"""Гибридный RAG: BM25 + Chroma, затем Reranker"""
import hashlib
from typing import Any, Dict, List

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import config
import prompts
from experiments import Answer, register_pipeline
from experiments.naive_rag.index import get_store, build_chunks
from llm import create_llm


@register_pipeline("hybrid_rag")
class HybridRagPipeline:
    """Гибридный поиск (BM25 + Chroma) с реранкером на CPU."""

    ANSWER_PREFIX = "Ответ"

    def __init__(self, model: str):
        if model not in config.MODELS:
            raise ValueError(f"Неизвестная модель: {model}")
        self.llm = create_llm(model)
        cfg = config.MODELS[model]
        self.model = model
        self.requested_model = cfg["api_model"]
        self.answer_col = f"{self.ANSWER_PREFIX}_{cfg['label']}_HybridRAG"
        self.retrieval_col = f"Найденные статьи_{cfg['label']}_HybridRAG"
        self.system_prompt = prompts.RAG_SYSTEM_PROMPT
        self.prompt_version = prompts.RAG_PROMPT_VERSION

        candidates_k = config.TOP_K * config.HYBRID_CANDIDATES_MULTIPLIER

        # 1. Векторный поиск (берем тот же индекс, что и в naive_rag)
        self.store = get_store()
        self.vector_retriever = self.store.as_retriever(
            search_kwargs={"k": candidates_k}
        )

        # 2. Лексический поиск - BM25 по тем же чанкам
        docs = build_chunks()
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = candidates_k

        # 3. Объединение (Ensemble)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[config.HYBRID_BM25_WEIGHT, config.HYBRID_VECTOR_WEIGHT]
        )

        # 4. Инициализируем реранкер
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name=config.RERANKER_MODEL,
            model_kwargs={"device": "cpu"}
        )

        # Условия для метаданных (чтобы resume не смешал naive и hybrid)
        self.index_conditions = {
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_revision": config.EMBEDDING_REVISION,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "index_version": config.INDEX_VERSION,
            "corpus_sha256": hashlib.sha256(config.CORPUS_PATH.read_bytes()).hexdigest(),
            # Новые параметры для гибрида
            "retriever_type": "hybrid_bm25_chroma",
            "bm25_weight": config.HYBRID_BM25_WEIGHT,
            "vector_weight": config.HYBRID_VECTOR_WEIGHT,
            "reranker_model": config.RERANKER_MODEL,
            "reranker_top_n": config.RERANKER_TOP_N,
        }

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Поиск, объединение, реранк и дедупликация по статьям."""
        # 1. Получаем кандидатов от Ensemble
        docs = self.ensemble_retriever.invoke(question)
        if not docs:
            return []

        # 2. Считаем скоры через Cross-Encoder
        # Формируем пары (вопрос, текст_документа)
        pairs = [[question, doc.page_content] for doc in docs]

        # Вызываем модель напрямую, получаем скоры. чем выше скор, тем релевантнее документ
        scores = self.reranker_model.score(pairs)

        # 3. Прикрепляем скоры к метаданным и сортируем
        for doc, score in zip(docs, scores):
            doc.metadata["relevance_score"] = float(score)

        docs_sorted = sorted(
            docs,
            key=lambda x: x.metadata["relevance_score"],
            reverse=True
        )

        # 4. Дедупликация и обрезка
        fragments: List[Dict[str, Any]] = []
        seen_articles: set = set()

        for doc in docs_sorted:
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
                    "score": doc.metadata["relevance_score"],
                }
            )

            # Ограничиваем итоговое количество статей до TOP_K
            if len(fragments) == config.TOP_K:
                break

        return fragments

    @staticmethod
    def retrieval_summary(fragments: List[Dict[str, Any]]) -> str:
        # например, "tk-81 (0.95); tk-80 (0.88)", если реранкер вернул скоры
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