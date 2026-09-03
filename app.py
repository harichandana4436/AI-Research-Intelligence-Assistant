import hashlib
import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import (
    extract_pages_from_pdf,
    chunk_text,
    extract_metadata,
)

from src.vector_store import VectorStore
from src.rag import RAGAssistant


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Research Intelligence Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 AI Research Intelligence Assistant")

st.caption(
    "AI-powered research paper analysis using RAG, hybrid "
    "BM25 + vector search, conversation memory, automatic "
    "summarization, paper comparison, metadata extraction "
    "and RAG evaluation."
)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# ============================================================
# INITIALIZE BACKEND
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_vector_store():
    return VectorStore()


@st.cache_resource(show_spinner=False)
def initialize_rag(_vector_store):
    return RAGAssistant(_vector_store)


try:
    vector_store = initialize_vector_store()
    rag = initialize_rag(vector_store)

except Exception as e:
    st.error(f"Application initialization failed: {e}")

    st.info(
        "Check your dependencies and make sure GROQ_API_KEY "
        "is correctly configured in the .env file."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Research Paper Library")

    st.write(
        "Upload research-paper PDFs directly into the "
        "application knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research-paper PDF files.",
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} PDF(s) selected."
        )

        process_button = st.button(
            "📥 Process Uploaded Papers",
            type="primary",
            use_container_width=True,
        )

        if process_button:

            all_documents = []

            progress_bar = st.progress(0)

            status_text = st.empty()

            total_files = len(uploaded_files)

            successful_files = 0

            for file_index, uploaded_file in enumerate(
                uploaded_files
            ):

                status_text.write(
                    f"Processing: {uploaded_file.name}"
                )

                try:

                    # ========================================
                    # EXTRACT PAGES
                    # ========================================

                    pages = extract_pages_from_pdf(
                        uploaded_file
                    )

                    if not pages:

                        st.warning(
                            f"No readable pages found in "
                            f"{uploaded_file.name}"
                        )

                        continue

                    # ========================================
                    # METADATA
                    # ========================================

                    metadata = extract_metadata(
                        pages
                    )

                    with st.expander(
                        f"🏷️ Metadata: {uploaded_file.name}"
                    ):

                        st.write(
                            f"**Title:** "
                            f"{metadata.get('title', 'Unknown')}"
                        )

                        st.write(
                            f"**Authors:** "
                            f"{metadata.get('authors', 'Unknown')}"
                        )

                        st.write(
                            f"**Year:** "
                            f"{metadata.get('year', 'Unknown')}"
                        )

                        st.write(
                            f"**DOI:** "
                            f"{metadata.get('doi', 'Not found')}"
                        )

                    # ========================================
                    # CHUNKING
                    # ========================================

                    paper_chunk_count = 0

                    for page in pages:

                        page_number = page["page"]

                        page_text = page["text"]

                        if not page_text.strip():
                            continue

                        chunks = chunk_text(
                            page_text
                        )

                        for chunk_index, chunk in enumerate(
                            chunks
                        ):

                            unique_string = (
                                f"{uploaded_file.name}|"
                                f"{page_number}|"
                                f"{chunk_index}|"
                                f"{chunk[:100]}"
                            )

                            document_id = hashlib.sha256(
                                unique_string.encode("utf-8")
                            ).hexdigest()

                            document = {
                                "id": document_id,
                                "text": chunk,
                                "source": uploaded_file.name,
                                "page": page_number,
                                "paper_title": metadata.get(
                                    "title",
                                    "Unknown title",
                                ),
                                "authors": metadata.get(
                                    "authors",
                                    "Unknown authors",
                                ),
                                "year": metadata.get(
                                    "year",
                                    "Unknown year",
                                ),
                                "doi": metadata.get(
                                    "doi",
                                    "Not found",
                                ),
                            }

                            all_documents.append(
                                document
                            )

                            paper_chunk_count += 1

                    successful_files += 1

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{paper_chunk_count} chunks created."
                    )

                except Exception as e:

                    st.error(
                        f"Error processing "
                        f"{uploaded_file.name}: {e}"
                    )

                progress_bar.progress(
                    (file_index + 1) / total_files
                )

            status_text.empty()

            # ================================================
            # STORE DOCUMENTS
            # ================================================

            if all_documents:

                try:

                    vector_store.add_documents(
                        all_documents
                    )

                    st.success(
                        f"Successfully indexed "
                        f"{successful_files} paper(s)."
                    )

                    st.info(
                        f"Total stored chunks: "
                        f"{vector_store.count()}"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Database storage failed: {e}"
                    )

            else:

                st.warning(
                    "No readable text was found "
                    "in the uploaded PDFs."
                )

    st.divider()

    # ========================================================
    # RETRIEVAL SETTINGS
    # ========================================================

    st.header("⚙️ Retrieval Settings")

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=2,
        max_value=10,
        value=5,
    )

    hybrid_search = st.checkbox(
        "Enable Hybrid BM25 + Vector Search",
        value=True,
    )

    st.divider()

    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.header("📊 Knowledge Base")

    st.metric(
        "Stored Chunks",
        vector_store.count(),
    )

    st.metric(
        "Indexed Papers",
        len(vector_store.list_sources()),
    )

    st.divider()

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    ):

        vector_store.clear()

        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.session_state.last_context = ""
        st.session_state.last_question = ""

        st.success("Knowledge base cleared.")

        st.rerun()


# ============================================================
# TABS
# ============================================================

(
    tab_chat,
    tab_library,
    tab_summary,
    tab_compare,
    tab_eval,
) = st.tabs(
    [
        "💬 Research Chat",
        "📁 My Library",
        "📝 Summarize Paper",
        "⚖️ Compare Papers",
        "📊 RAG Evaluation",
    ]
)


# ============================================================
# LIBRARY
# ============================================================

with tab_library:

    st.header("📚 My Research Paper Library")

    sources = vector_store.list_sources()

    if not sources:

        st.info(
            "No papers indexed yet. "
            "Upload PDF files from the sidebar."
        )

    else:

        st.success(
            f"{len(sources)} research paper(s) indexed."
        )

        for source in sources:

            metadata = (
                vector_store.get_source_metadata(
                    source
                )
            )

            with st.expander(
                f"📄 {source}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Title:**",
                        metadata.get(
                            "paper_title",
                            "Unknown",
                        ),
                    )

                    st.write(
                        "**Authors:**",
                        metadata.get(
                            "authors",
                            "Unknown",
                        ),
                    )

                with col2:

                    st.write(
                        "**Year:**",
                        metadata.get(
                            "year",
                            "Unknown",
                        ),
                    )

                    st.write(
                        "**DOI:**",
                        metadata.get(
                            "doi",
                            "Not found",
                        ),
                    )


# ============================================================
# RESEARCH CHAT
# ============================================================

with tab_chat:

    st.header("💬 Research Paper Chat")

    st.caption(
        "Ask questions about your uploaded research papers. "
        "Answers are generated from retrieved paper content."
    )

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask a question about your research papers..."
    )

    if question:

        st.session_state.last_question = question

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        if vector_store.count() == 0:

            answer = (
                "Please upload and process at least "
                "one research paper first."
            )

            results = {}

            with st.chat_message("assistant"):
                st.warning(answer)

        else:

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching papers and generating answer..."
                ):

                    try:

                        answer, results = rag.answer(
                            question=question,
                            n_results=top_k,
                            history=(
                                st.session_state
                                .chat_history[:-1]
                            ),
                            hybrid=hybrid_search,
                        )

                    except Exception as e:

                        answer = (
                            f"Error while generating answer: "
                            f"{e}"
                        )

                        results = {}

                st.markdown(answer)

                st.session_state.last_answer = answer

                documents = results.get(
                    "documents",
                    [[]],
                )

                if documents:
                    documents = documents[0]

                metadatas = results.get(
                    "metadatas",
                    [[]],
                )

                if metadatas:
                    metadatas = metadatas[0]

                scores = results.get(
                    "scores",
                    [[]],
                )

                if scores:
                    scores = scores[0]

                if documents:

                    st.subheader(
                        "📚 Retrieved Sources"
                    )

                    context_for_eval = []

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

                        score = (
                            scores[index]
                            if index < len(scores)
                            else None
                        )

                        context_for_eval.append(
                            document
                        )

                        with st.expander(
                            f"📄 {source} — Page {page}"
                        ):

                            if score is not None:

                                st.caption(
                                    f"Hybrid relevance score: "
                                    f"{score:.4f}"
                                )

                            st.write(document)

                    st.session_state.last_context = (
                        "\n\n".join(
                            context_for_eval
                        )
                    )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    if st.session_state.chat_history:

        st.divider()

        if st.button(
            "🧹 Clear Conversation Memory"
        ):

            st.session_state.chat_history = []

            st.session_state.last_answer = ""
            st.session_state.last_context = ""
            st.session_state.last_question = ""

            st.rerun()


# ============================================================
# SUMMARIZATION
# ============================================================

with tab_summary:

    st.header(
        "📝 Automatic Research Paper Summarization"
    )

    sources = vector_store.list_sources()

    if not sources:

        st.info(
            "Upload and process a research paper first."
        )

    else:

        selected_paper = st.selectbox(
            "Select Research Paper",
            sources,
            key="summary_paper",
        )

        if st.button(
            "📝 Generate Summary",
            type="primary",
        ):

            with st.spinner(
                "Generating structured research summary..."
            ):

                try:

                    summary = rag.summarize_paper(
                        selected_paper
                    )

                    st.markdown(summary)

                except Exception as e:

                    st.error(
                        f"Summary generation failed: {e}"
                    )


# ============================================================
# PAPER COMPARISON
# ============================================================

with tab_compare:

    st.header(
        "⚖️ Research Paper Comparison"
    )

    st.caption(
        "Select two or more uploaded papers. The system "
        "creates compact summaries first and then compares "
        "the papers to avoid oversized LLM requests."
    )

    sources = vector_store.list_sources()

    if len(sources) < 2:

        st.info(
            "Upload and process at least two "
            "research papers to compare them."
        )

    else:

        selected_papers = st.multiselect(
            "Select papers to compare",
            sources,
            max_selections=3,
            key="comparison_papers",
        )

        if st.button(
            "⚖️ Compare Papers",
            type="primary",
        ):

            if len(selected_papers) < 2:

                st.warning(
                    "Please select at least two papers."
                )

            else:

                with st.spinner(
                    "Creating paper summaries and comparing them..."
                ):

                    try:

                        comparison = rag.compare_papers(
                            selected_papers
                        )

                        st.markdown(comparison)

                    except Exception as e:

                        st.error(
                            f"Paper comparison failed: {e}"
                        )


# ============================================================
# RAG EVALUATION
# ============================================================

with tab_eval:

    st.header(
        "📊 RAG Evaluation Metrics"
    )

    st.info(
        """
        These are diagnostic RAG metrics. They help estimate
        retrieval relevance, answer grounding and question
        relevance. For a research-quality evaluation, use a
        human-labelled benchmark as well.
        """
    )

    evaluation_question = st.text_input(
        "Evaluation Question",
        value=st.session_state.get(
            "last_question",
            "",
        ),
    )

    evaluation_answer = st.text_area(
        "Generated Answer",
        value=st.session_state.last_answer,
        height=180,
    )

    evaluation_context = st.text_area(
        "Retrieved Context",
        value=st.session_state.last_context,
        height=250,
    )

    if st.button(
        "📊 Calculate Metrics",
        type="primary",
    ):

        if not evaluation_question.strip():

            st.warning(
                "Please enter an evaluation question."
            )

        elif not evaluation_answer.strip():

            st.warning(
                "Please enter the generated answer."
            )

        elif not evaluation_context.strip():

            st.warning(
                "Please enter the retrieved context."
            )

        else:

            try:

                metrics = rag.evaluate_rag(
                    question=evaluation_question,
                    answer=evaluation_answer,
                    context=evaluation_context,
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Context Relevance",
                        f"{metrics['context_relevance']:.2f}",
                    )

                with col2:

                    st.metric(
                        "Answer Grounding",
                        f"{metrics['answer_grounding']:.2f}",
                    )

                with col3:

                    st.metric(
                        "Answer Relevance",
                        f"{metrics['answer_question_relevance']:.2f}",
                    )

                with col4:

                    st.metric(
                        "Overall Score",
                        f"{metrics['overall_score']:.2f}",
                    )

                st.subheader(
                    "📈 Evaluation Interpretation"
                )

                st.write(
                    metrics["interpretation"]
                )

            except Exception as e:

                st.error(
                    f"Evaluation failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Research Intelligence Assistant | "
    "Streamlit + ChromaDB + Sentence Transformers + "
    "BM25 + Hybrid Search + Groq"
)