import os
import re

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class RAGAssistant:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, vector_store):

        self.vector_store = vector_store

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        self.memory_enabled = (
            os.getenv(
                "MEMORY_ENABLED",
                "true",
            ).lower()
            == "true"
        )

        self.memory_window = int(
            os.getenv(
                "MEMORY_WINDOW",
                "6",
            )
        )

        # Maximum characters sent to the LLM in one prompt.
        self.max_prompt_chars = int(
            os.getenv(
                "MAX_PROMPT_CHARS",
                "24000",
            )
        )

        # Maximum characters used from one paper
        # during comparison.
        self.max_compare_chars = int(
            os.getenv(
                "MAX_COMPARE_CHARS",
                "9000",
            )
        )

        # Maximum characters used for summarization
        # batches.
        self.summary_batch_chars = int(
            os.getenv(
                "SUMMARY_BATCH_CHARS",
                "8000",
            )
        )

    # ========================================================
    # SAFE TEXT
    # ========================================================

    @staticmethod
    def _clean_text(text):

        if not text:
            return ""

        return str(text).strip()

    # ========================================================
    # TRUNCATE TEXT
    # ========================================================

    @staticmethod
    def _truncate(text, max_chars):

        text = text or ""

        if len(text) <= max_chars:
            return text

        return (
            text[:max_chars]
            + "\n\n[Content truncated for model limits.]"
        )

    # ========================================================
    # LLM CALL
    # ========================================================

    def _generate(
        self,
        prompt,
        max_tokens=1800,
    ):

        prompt = self._truncate(
            prompt,
            self.max_prompt_chars,
        )

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert research "
                            "paper analysis assistant. "
                            "Use only the information supplied "
                            "in the user prompt."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=0.1,

                max_tokens=max_tokens,
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def _build_context(
        self,
        documents,
        metadatas,
    ):

        context_parts = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            context_parts.append(
                f"SOURCE: {source}\n"
                f"PAGE: {page}\n\n"
                f"{document}"
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        return self._truncate(
            context,
            16000,
        )

    # ========================================================
    # CONVERSATION MEMORY
    # ========================================================

    def _build_memory(
        self,
        history,
    ):

        if not self.memory_enabled:
            return ""

        if not history:
            return ""

        recent_history = history[
            -self.memory_window:
        ]

        parts = []

        for message in recent_history:

            role = message.get(
                "role",
                "user",
            )

            content = self._truncate(
                message.get(
                    "content",
                    "",
                ),
                1200,
            )

            parts.append(
                f"{role.upper()}: {content}"
            )

        return "\n\n".join(parts)

    # ========================================================
    # ANSWER QUESTION
    # ========================================================

    def answer(
        self,
        question,
        n_results=5,
        history=None,
        hybrid=True,
    ):

        results = self.vector_store.search(
            query=question,
            n_results=n_results,
            hybrid=hybrid,
        )

        documents = results.get(
            "documents",
            [[]],
        )

        metadatas = results.get(
            "metadatas",
            [[]],
        )

        documents = (
            documents[0]
            if documents
            else []
        )

        metadatas = (
            metadatas[0]
            if metadatas
            else []
        )

        if not documents:

            return (
                "The information is not available "
                "in the uploaded research papers.",
                results,
            )

        context = self._build_context(
            documents,
            metadatas,
        )

        memory_text = self._build_memory(
            history
        )

        prompt = f"""
You are a research-paper RAG assistant.

Answer the user's question using ONLY the
research-paper context below.

RULES:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the answer is not supported by the context,
   say that it is not available in the uploaded papers.
4. Give the answer directly.
5. Explain important claims using paper evidence.
6. Mention the paper name and page number when possible.
7. If multiple papers support different statements,
   clearly identify the source.
8. Conversation history may resolve references such
   as "this paper", "it", or "that method", but factual
   claims must come from retrieved paper content.

CONVERSATION MEMORY:

{memory_text}

RESEARCH PAPER CONTEXT:

{context}

USER QUESTION:

{question}

Provide a concise research-oriented answer.
"""

        answer = self._generate(
            prompt,
            max_tokens=1800,
        )

        return answer, results

    # ========================================================
    # GET PAPER TEXT
    # ========================================================

    def _get_paper_text(
        self,
        source,
        max_chars=None,
    ):

        documents = (
            self.vector_store
            .get_source_documents(
                source
            )
        )

        if not documents:
            return ""

        parts = []

        for item in documents:

            metadata = item.get(
                "metadata",
                {},
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            text = item.get(
                "text",
                "",
            )

            parts.append(
                f"PAGE {page}\n{text}"
            )

        text = "\n\n".join(parts)

        return self._truncate(
            text,
            max_chars or self.max_compare_chars,
        )

    # ========================================================
    # SPLIT INTO BATCHES
    # ========================================================

    def _make_batches(
        self,
        text,
        max_chars,
    ):

        if not text:
            return []

        sections = text.split(
            "\n\n"
        )

        batches = []

        current = ""

        for section in sections:

            section = section.strip()

            if not section:
                continue

            if (
                len(current)
                + len(section)
                + 2
                > max_chars
            ):

                if current:
                    batches.append(
                        current
                    )

                current = section

            else:

                if current:
                    current += "\n\n"

                current += section

        if current:
            batches.append(current)

        return batches

    # ========================================================
    # SUMMARIZE SINGLE BATCH
    # ========================================================

    def _summarize_batch(
        self,
        batch,
    ):

        prompt = f"""
Analyze this section of a research paper.

Use ONLY the supplied content.

Extract:

- Research problem
- Motivation
- Objective
- Methodology
- Models or algorithms
- Dataset / experiments
- Main findings
- Results
- Limitations
- Future work
- Important technical details

Do not invent information.

If something is missing, do not guess.

PAPER SECTION:

{batch}
"""

        return self._generate(
            prompt,
            max_tokens=1200,
        )

    # ========================================================
    # SUMMARIZE PAPER
    # ========================================================

    def summarize_paper(
        self,
        source,
    ):

        documents = (
            self.vector_store
            .get_source_documents(
                source
            )
        )

        if not documents:

            return (
                "No indexed content was found "
                "for this paper."
            )

        sections = []

        for item in documents:

            metadata = item.get(
                "metadata",
                {},
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            text = item.get(
                "text",
                "",
            )

            if text.strip():

                sections.append(
                    f"PAGE {page}\n{text}"
                )

        full_text = "\n\n".join(
            sections
        )

        batches = self._make_batches(
            full_text,
            self.summary_batch_chars,
        )

        if not batches:
            return "No readable content was found."

        partial_summaries = []

        for batch in batches:

            partial = self._summarize_batch(
                batch
            )

            partial_summaries.append(
                partial
            )

        combined = "\n\n---\n\n".join(
            partial_summaries
        )

        combined = self._truncate(
            combined,
            14000,
        )

        final_prompt = f"""
Create a final structured research-paper summary.

Use ONLY the intermediate analysis below.

Format:

# Paper Summary

## 1. Research Problem

## 2. Motivation

## 3. Objective

## 4. Proposed Method

## 5. Methodology

## 6. Models / Algorithms

## 7. Dataset / Experiments

## 8. Main Results

## 9. Key Contributions

## 10. Limitations

## 11. Future Work

## 12. Overall Summary

Do not invent information.

If information is unavailable, write:

"Not clearly specified in the paper."

INTERMEDIATE PAPER ANALYSIS:

{combined}
"""

        return self._generate(
            final_prompt,
            max_tokens=2500,
        )

    # ========================================================
    # CREATE COMPACT PAPER SUMMARY
    # ========================================================

    def _create_comparison_summary(
        self,
        source,
    ):

        paper_text = self._get_paper_text(
            source,
            max_chars=self.max_compare_chars,
        )

        if not paper_text:
            return (
                f"PAPER: {source}\n"
                "No readable content found."
            )

        prompt = f"""
Create a compact factual research-paper profile.

Use ONLY the supplied paper content.

Return:

Paper:
Research Problem:
Objective:
Methodology:
Models/Algorithms:
Dataset:
Experiments:
Main Results:
Contributions:
Limitations:

Keep each field concise.

Do not invent information.
If unavailable, write "Not specified".

PAPER CONTENT:

{paper_text}
"""

        summary = self._generate(
            prompt,
            max_tokens=1200,
        )

        return (
            f"PAPER: {source}\n\n"
            f"{summary}"
        )

    # ========================================================
    # COMPARE PAPERS
    # ========================================================

    def compare_papers(
        self,
        sources,
    ):

        if len(sources) < 2:

            return (
                "At least two papers are required "
                "for comparison."
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT send complete papers together.
        # First create compact summaries.
        # ----------------------------------------------------

        paper_profiles = []

        for source in sources:

            profile = (
                self._create_comparison_summary(
                    source
                )
            )

            paper_profiles.append(
                profile
            )

        combined_profiles = "\n\n====================\n\n".join(
            paper_profiles
        )

        # Very important for TPM protection.
        combined_profiles = self._truncate(
            combined_profiles,
            12000,
        )

        prompt = f"""
Compare the research papers using ONLY the
compact paper profiles below.

Create a clear research comparison.

# Research Paper Comparison

## 1. Research Problem

Compare the problem addressed by each paper.

## 2. Objectives

Compare the objectives.

## 3. Methodology

Compare the methodologies.

## 4. Models / Algorithms

Compare the models and algorithms.

## 5. Datasets

Compare datasets and experimental data.

## 6. Experimental Setup

Compare experiments and evaluation setup.

## 7. Main Results

Compare reported results.

## 8. Key Contributions

Compare major contributions.

## 9. Limitations

Compare limitations.

## 10. Similarities

List important similarities.

## 11. Differences

List important differences.

## 12. Strengths and Weaknesses

Explain the strengths and weaknesses based ONLY
on the supplied information.

## 13. Overall Comparison

Give a concise overall comparison.

IMPORTANT:

- Do not invent facts.
- Do not use outside knowledge.
- Do not assume missing information.
- If something is unavailable, say
  "Not specified in the provided paper profile."
- Keep each paper clearly identified.

PAPER PROFILES:

{combined_profiles}
"""

        return self._generate(
            prompt,
            max_tokens=2200,
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    @staticmethod
    def _tokens(text):

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

    # ========================================================
    # TOKEN SET OVERLAP
    # ========================================================

    @staticmethod
    def _overlap(
        first,
        second,
    ):

        first_tokens = set(
            RAGAssistant._tokens(
                first
            )
        )

        second_tokens = set(
            RAGAssistant._tokens(
                second
            )
        )

        if not first_tokens:
            return 0.0

        return (
            len(
                first_tokens
                & second_tokens
            )
            / len(first_tokens)
        )

    # ========================================================
    # SENTENCE GROUNDING
    # ========================================================

    def _grounding_score(
        self,
        answer,
        context,
    ):

        answer_sentences = re.split(
            r"[.!?]+",
            answer,
        )

        answer_sentences = [
            sentence.strip()
            for sentence in answer_sentences
            if len(sentence.strip()) > 20
        ]

        if not answer_sentences:
            return 0.0

        grounded = 0

        context_lower = context.lower()

        for sentence in answer_sentences:

            words = [
                word
                for word in self._tokens(
                    sentence
                )
                if len(word) > 3
            ]

            if not words:
                continue

            matches = sum(
                1
                for word in words
                if word in context_lower
            )

            ratio = (
                matches
                / len(words)
            )

            if ratio >= 0.35:
                grounded += 1

        return (
            grounded
            / len(answer_sentences)
        )

    # ========================================================
    # RAG EVALUATION
    # ========================================================

    def evaluate_rag(
        self,
        question,
        answer,
        context,
    ):

        context_relevance = (
            self._overlap(
                question,
                context,
            )
        )

        answer_question_relevance = (
            self._overlap(
                question,
                answer,
            )
        )

        answer_grounding = (
            self._grounding_score(
                answer,
                context,
            )
        )

        average = (
            context_relevance
            + answer_question_relevance
            + answer_grounding
        ) / 3

        if average >= 0.75:

            interpretation = (
                "Strong retrieval and grounding "
                "signals. The answer appears well "
                "connected to the retrieved context."
            )

        elif average >= 0.50:

            interpretation = (
                "Moderate RAG quality. Some evidence "
                "supports the answer, but retrieval "
                "or grounding could be improved."
            )

        else:

            interpretation = (
                "Low RAG quality signal. Consider "
                "improving chunking, retrieval settings, "
                "or the retrieved context."
            )

        return {
            "context_relevance":
                context_relevance,

            "answer_grounding":
                answer_grounding,

            "answer_question_relevance":
                answer_question_relevance,

            "overall_score":
                average,

            "interpretation":
                interpretation,
        }