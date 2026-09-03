import math
import os
import re

import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()


class VectorStore:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        )

        self.persist_directory = os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "./chroma_db",
        )

        self.collection_name = os.getenv(
            "CHROMA_COLLECTION_NAME",
            "research_papers",
        )

        self.vector_weight = float(
            os.getenv(
                "VECTOR_WEIGHT",
                "0.7",
            )
        )

        self.bm25_weight = float(
            os.getenv(
                "BM25_WEIGHT",
                "0.3",
            )
        )

        self.embedding_model = (
            SentenceTransformer(
                self.embedding_model_name
            )
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )

        self.documents_cache = []

        self._rebuild_cache()

    # ========================================================
    # TOKENIZATION
    # ========================================================

    @staticmethod
    def tokenize(text):

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

    # ========================================================
    # CACHE
    # ========================================================

    def _rebuild_cache(self):

        self.documents_cache = []

        total = self.collection.count()

        if total == 0:
            return

        data = self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = data.get(
            "documents",
            [],
        )

        metadatas = data.get(
            "metadatas",
            [],
        )

        ids = data.get(
            "ids",
            [],
        )

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            document_id = (
                ids[index]
                if index < len(ids)
                else str(index)
            )

            self.documents_cache.append(
                {
                    "id": document_id,
                    "text": document,
                    "metadata": metadata,
                }
            )

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents,
    ):

        if not documents:
            return

        texts = [
            document["text"]
            for document in documents
        ]

        ids = [
            document["id"]
            for document in documents
        ]

        metadatas = []

        for document in documents:

            metadata = {
                "source": str(
                    document.get(
                        "source",
                        "Unknown",
                    )
                ),

                "page": str(
                    document.get(
                        "page",
                        "Unknown",
                    )
                ),

                "paper_title": str(
                    document.get(
                        "paper_title",
                        "Unknown",
                    )
                ),

                "authors": str(
                    document.get(
                        "authors",
                        "Unknown",
                    )
                ),

                "year": str(
                    document.get(
                        "year",
                        "Unknown",
                    )
                ),

                "doi": str(
                    document.get(
                        "doi",
                        "Not found",
                    )
                ),
            }

            metadatas.append(
                metadata
            )

        embeddings = (
            self.embedding_model.encode(
                texts,
                show_progress_bar=False,
            ).tolist()
        )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self._rebuild_cache()

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(
        self,
        query,
        n_results=5,
    ):

        count = self.collection.count()

        if count == 0:

            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        n_results = min(
            n_results,
            count,
        )

        query_embedding = (
            self.embedding_model.encode(
                [query]
            ).tolist()
        )

        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

    # ========================================================
    # BM25
    # ========================================================

    def bm25_search(
        self,
        query,
        n_results=5,
    ):

        if not self.documents_cache:
            return []

        query_tokens = self.tokenize(
            query
        )

        if not query_tokens:
            return []

        N = len(
            self.documents_cache
        )

        tokenized_docs = []

        document_frequency = {}

        for item in self.documents_cache:

            tokens = self.tokenize(
                item["text"]
            )

            tokenized_docs.append(
                tokens
            )

            for token in set(tokens):

                document_frequency[token] = (
                    document_frequency.get(
                        token,
                        0,
                    )
                    + 1
                )

        avg_doc_length = (
            sum(
                len(tokens)
                for tokens in tokenized_docs
            )
            / max(N, 1)
        )

        k1 = 1.5
        b = 0.75

        scored_documents = []

        for index, item in enumerate(
            self.documents_cache
        ):

            tokens = tokenized_docs[index]

            if not tokens:
                continue

            term_frequency = {}

            for token in tokens:

                term_frequency[token] = (
                    term_frequency.get(
                        token,
                        0,
                    )
                    + 1
                )

            score = 0.0

            doc_length = len(tokens)

            for query_token in query_tokens:

                tf = term_frequency.get(
                    query_token,
                    0,
                )

                if tf == 0:
                    continue

                df = document_frequency.get(
                    query_token,
                    0,
                )

                idf = math.log(
                    1
                    + (
                        N
                        - df
                        + 0.5
                    )
                    / (
                        df
                        + 0.5
                    )
                )

                denominator = (
                    tf
                    + k1
                    * (
                        1
                        - b
                        + b
                        * (
                            doc_length
                            / max(
                                avg_doc_length,
                                1,
                            )
                        )
                    )
                )

                score += (
                    idf
                    * (
                        tf
                        * (k1 + 1)
                    )
                    / denominator
                )

            if score > 0:

                scored_documents.append(
                    (
                        score,
                        item,
                    )
                )

        scored_documents.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return scored_documents[
            :n_results
        ]

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        query,
        n_results=5,
    ):

        total = self.collection.count()

        if total == 0:

            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "scores": [[]],
            }

        candidate_count = min(
            max(
                n_results * 4,
                10,
            ),
            total,
        )

        vector_results = (
            self.vector_search(
                query,
                candidate_count,
            )
        )

        vector_documents = (
            vector_results.get(
                "documents",
                [[]],
            )[0]
        )

        vector_metadatas = (
            vector_results.get(
                "metadatas",
                [[]],
            )[0]
        )

        vector_distances = (
            vector_results.get(
                "distances",
                [[]],
            )[0]
        )

        bm25_results = (
            self.bm25_search(
                query,
                candidate_count,
            )
        )

        fused = {}

        # ----------------------------------------------------
        # VECTOR RANKING
        # ----------------------------------------------------

        for rank, document in enumerate(
            vector_documents,
            start=1,
        ):

            metadata = (
                vector_metadatas[rank - 1]
                if rank - 1
                < len(vector_metadatas)
                else {}
            )

            key = (
                metadata.get(
                    "source",
                    "",
                ),
                metadata.get(
                    "page",
                    "",
                ),
                document[:100],
            )

            if key not in fused:

                fused[key] = {
                    "text": document,
                    "metadata": metadata,
                    "score": 0.0,
                    "distance": (
                        vector_distances[
                            rank - 1
                        ]
                        if rank - 1
                        < len(vector_distances)
                        else None
                    ),
                }

            fused[key]["score"] += (
                self.vector_weight
                / rank
            )

        # ----------------------------------------------------
        # BM25 RANKING
        # ----------------------------------------------------

        for rank, (
            bm25_score,
            item,
        ) in enumerate(
            bm25_results,
            start=1,
        ):

            document = item["text"]

            metadata = item["metadata"]

            key = (
                metadata.get(
                    "source",
                    "",
                ),
                metadata.get(
                    "page",
                    "",
                ),
                document[:100],
            )

            if key not in fused:

                fused[key] = {
                    "text": document,
                    "metadata": metadata,
                    "score": 0.0,
                    "distance": None,
                }

            # Reciprocal rank contribution
            fused[key]["score"] += (
                self.bm25_weight
                / rank
            )

        ranked = sorted(
            fused.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        ranked = ranked[
            :n_results
        ]

        return {
            "documents": [
                [
                    item["text"]
                    for item in ranked
                ]
            ],

            "metadatas": [
                [
                    item["metadata"]
                    for item in ranked
                ]
            ],

            "distances": [
                [
                    item["distance"]
                    for item in ranked
                ]
            ],

            "scores": [
                [
                    item["score"]
                    for item in ranked
                ]
            ],
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        n_results=5,
        hybrid=True,
    ):

        if hybrid:

            return self.hybrid_search(
                query,
                n_results,
            )

        return self.vector_search(
            query,
            n_results,
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count(self):

        return self.collection.count()

    # ========================================================
    # LIST SOURCES
    # ========================================================

    def list_sources(self):

        if self.collection.count() == 0:
            return []

        data = self.collection.get(
            include=[
                "metadatas"
            ]
        )

        metadatas = data.get(
            "metadatas",
            [],
        )

        sources = set()

        for metadata in metadatas:

            source = metadata.get(
                "source"
            )

            if source:
                sources.add(source)

        return sorted(
            sources
        )

    # ========================================================
    # PAPER METADATA
    # ========================================================

    def get_source_metadata(
        self,
        source,
    ):

        data = self.collection.get(
            where={
                "source": source
            },
            include=[
                "metadatas"
            ],
        )

        metadatas = data.get(
            "metadatas",
            [],
        )

        if not metadatas:
            return {}

        return metadatas[0]

    # ========================================================
    # PAPER DOCUMENTS
    # ========================================================

    def get_source_documents(
        self,
        source,
    ):

        data = self.collection.get(
            where={
                "source": source
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = data.get(
            "documents",
            [],
        )

        metadatas = data.get(
            "metadatas",
            [],
        )

        combined = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            combined.append(
                {
                    "text": document,
                    "metadata": metadata,
                }
            )

        def page_number(item):

            page = item[
                "metadata"
            ].get(
                "page",
                "0",
            )

            try:
                return int(page)
            except Exception:
                return 0

        combined.sort(
            key=page_number
        )

        return combined

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):

        try:

            self.client.delete_collection(
                name=self.collection_name
            )

        except Exception:

            pass

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )

        self.documents_cache = []