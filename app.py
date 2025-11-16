import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator
from finetuning import EmbeddingFineTuner
from config import AVAILABLE_MODELS, DEFAULT_MODEL, EMBEDDING_MODEL, FAISS_INDEX_PATH
import tempfile
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stAlert > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.vector_store = VectorStore(model_name=EMBEDDING_MODEL)
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.vector_store,
            model_name=DEFAULT_MODEL,
            use_memory=True
        )
        st.session_state.chat_history = []
        st.session_state.documents_processed = 0
        st.session_state.current_model = DEFAULT_MODEL
        st.session_state.initialized = True
        st.session_state.show_advanced = False

        if os.path.exists(FAISS_INDEX_PATH):
            if st.session_state.vector_store.load_index(FAISS_INDEX_PATH):
                st.session_state.documents_processed = st.session_state.vector_store.document_count

initialize_session_state()

with st.sidebar:
    st.title("RAG System")

    st.subheader("Model Settings")
    try:
        current_model_index = list(AVAILABLE_MODELS.values()).index(st.session_state.current_model)
    except ValueError:
        current_model_index = 0
        st.session_state.current_model = list(AVAILABLE_MODELS.values())[0]

    selected_model_name = st.selectbox(
        "Select LLM Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=current_model_index
    )

    selected_model = AVAILABLE_MODELS[selected_model_name]

    if selected_model != st.session_state.current_model:
        st.session_state.rag_pipeline.update_model(selected_model)
        st.session_state.current_model = selected_model
        st.success(f"Switched to {selected_model_name}")

    st.divider()

    st.subheader("Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to query"
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process", use_container_width=True, type="primary")
    with col2:
        save_btn = st.button("Save Index", use_container_width=True)

    if process_btn and uploaded_files:
        all_chunks = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = st.session_state.processor.process_file(tmp_path)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("Creating embeddings...")
        st.session_state.vector_store.add_documents(all_chunks)
        st.session_state.documents_processed = st.session_state.vector_store.document_count

        progress_bar.empty()
        status_text.empty()
        st.success(f"Processed {len(all_chunks)} chunks from {len(uploaded_files)} documents!")

    if save_btn:
        st.session_state.vector_store.save_index(FAISS_INDEX_PATH)
        st.success("Index saved!")

    st.divider()

    st.subheader("Statistics")
    stats = st.session_state.vector_store.get_statistics()

    st.metric("Total Documents", stats.get('total_documents', 0))
    st.metric("Unique Sources", stats.get('unique_sources', 0))

    if stats.get('source_files'):
        with st.expander("View Source Files"):
            for source in stats['source_files']:
                st.text(f"• {source}")

    st.divider()

    st.session_state.show_advanced = st.checkbox("Show Advanced Features", 
                                                   value=st.session_state.show_advanced)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.rag_pipeline.clear_memory()
        st.rerun()

st.title("Document Q&A Chat")

if st.session_state.show_advanced:
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Evaluation", "Fine-tuning", "About"])
else:
    tab1, tab4 = st.tabs(["Chat", "About"])
    tab2, tab3 = None, None

with tab1:
    if st.session_state.vector_store.vector_store is None:
        st.info("Please upload and process documents using the sidebar to get started.")
    else:
        if not st.session_state.chat_history:
            st.subheader("Suggested Questions")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Summarize documents", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Please provide a summary of the uploaded documents"
                    })
                    st.rerun()

            with col2:
                if st.button("Suggest questions", use_container_width=True):
                    with st.spinner("Generating questions..."):
                        questions = st.session_state.rag_pipeline.suggest_questions(3)
                        if questions:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "Here are some questions you can ask:\n\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                            })
                    st.rerun()

            with col3:
                if st.button("List sources", use_container_width=True):
                    sources = st.session_state.vector_store.get_unique_sources()
                    content = "Available documents:\n\n" + "\n".join(f"• {s}" for s in sources)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    st.rerun()

        st.divider()

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("View Sources", expanded=False):
                            for idx, source in enumerate(message["sources"], 1):
                                page_num = source.get('page', 0)
                                page_display = page_num + 1 if isinstance(page_num, int) else page_num

                                st.markdown(f"**Source {idx}:** {source['source']}, Page {page_display}")
                                st.caption(f"Relevance Score: {source['score']:.4f}")
                                st.text(source['content'][:200] + "...")
                                st.divider()

                    if message["role"] == "assistant" and "confidence" in message:
                        conf = message["confidence"]
                        conf_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                        st.caption(f"{conf_color} Confidence: {conf:.2%}")

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })

            with st.spinner("Thinking..."):
                history_tuples = []
                for i in range(0, len(st.session_state.chat_history) - 1, 2):
                    if i + 1 < len(st.session_state.chat_history):
                        user_msg = st.session_state.chat_history[i]["content"]
                        asst_msg = st.session_state.chat_history[i + 1]["content"]
                        history_tuples.append((user_msg, asst_msg))

                result = st.session_state.rag_pipeline.generate_answer(
                    prompt, 
                    k=5,
                    chat_history=history_tuples[-3:]
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources'],
                    "confidence": result['confidence']
                })

            st.rerun()

if tab2 and st.session_state.show_advanced:
    with tab2:
        st.header("System Evaluation")

        if st.session_state.vector_store.vector_store is None:
            st.warning("Please upload and process documents first.")
        else:
            evaluator = RAGEvaluator(
                st.session_state.rag_pipeline,
                st.session_state.vector_store
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Generate Test Set")
                num_questions = st.number_input("Number of questions", 10, 50, 20)

                if st.button("Generate Test Questions", use_container_width=True):
                    with st.spinner("Generating test questions..."):
                        test_set = evaluator.generate_test_set(num_questions)
                        evaluator.save_test_set("test_set.json")
                        st.success(f"Generated {len(test_set)} questions")

                        with st.expander("View Test Questions"):
                            for idx, qa in enumerate(test_set[:5], 1):
                                st.markdown(f"**Q{idx}:** {qa['question']}")
                                st.caption(f"Source: {qa.get('source_doc', 'N/A')}")
                                st.divider()

            with col2:
                st.subheader("Run Evaluation")
                k_value = st.slider("Retrieval K", 1, 10, 5)

                if st.button("valuate System", use_container_width=True):
                    with st.spinner("Running evaluation..."):
                        if os.path.exists("test_set.json"):
                            evaluator.load_test_set("test_set.json")

                        if evaluator.test_set:
                            retrieval_results = evaluator.evaluate_retrieval(k=k_value)
                            generation_results = evaluator.evaluate_generation(sample_size=10)

                            st.subheader("Results")

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Precision@K", f"{retrieval_results['precision@k']:.3f}")
                            with col_b:
                                st.metric("Recall", f"{retrieval_results['recall']:.3f}")
                            with col_c:
                                st.metric("MRR", f"{retrieval_results['mrr']:.3f}")

                            st.metric("Generation Score", f"{generation_results['average_score']:.1f}/10")

                            evaluator.save_evaluation_results(
                                retrieval_results,
                                generation_results,
                                f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            )
                        else:
                            st.error("No test set available. Generate one first!")

if tab3 and st.session_state.show_advanced:
    with tab3:
        st.header("Embedding Fine-tuning")

        if st.session_state.vector_store.vector_store is None:
            st.warning("Please upload and process documents first.")
        else:
            finetuner = EmbeddingFineTuner(EMBEDDING_MODEL)

            st.subheader("1. Generate Training Data")

            col1, col2 = st.columns(2)
            with col1:
                num_pairs = st.number_input("Training pairs", 20, 200, 50)
            with col2:
                if st.button("Generate Training Data", use_container_width=True):
                    with st.spinner("Generating training data..."):
                        docs = st.session_state.vector_store.get_all_documents()
                        training_data = finetuner.generate_training_data_from_docs(
                            docs,
                            st.session_state.rag_pipeline.llm,
                            num_pairs=num_pairs
                        )
                        finetuner.save_training_data("training_data.json")
                        st.success(f"Generated {len(training_data)} pairs")

            st.divider()

            st.subheader("2. Fine-tune Model")

            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.number_input("Epochs", 1, 10, 3)
            with col2:
                batch_size = st.number_input("Batch size", 8, 32, 16)
            with col3:
                if st.button("Start Fine-tuning", use_container_width=True):
                    if os.path.exists("training_data.json"):
                        with st.spinner("Fine-tuning in progress... This may take a while."):
                            finetuner.load_training_data("training_data.json")
                            finetuner.fine_tune(
                                output_path="models/finetuned_embeddings",
                                epochs=epochs,
                                batch_size=batch_size
                            )
                            st.success("Fine-tuning complete!")
                            st.info("Restart the app and update EMBEDDING_MODEL in config.py to use the fine-tuned model")
                    else:
                        st.error("Generate training data first!")

            st.divider()

            st.subheader("3. Compare Models")

            test_query = st.text_input("Test query", "What is the main topic of the documents?")

            if st.button("Compare Base vs Fine-tuned", use_container_width=True):
                if os.path.exists("models/finetuned_embeddings"):
                    with st.spinner("Comparing models..."):
                        docs = list(st.session_state.vector_store.get_all_documents())[:10]
                        doc_texts = [doc.page_content for doc in docs]

                        results = finetuner.compare_models(
                            test_query,
                            doc_texts,
                            "models/finetuned_embeddings"
                        )

                        if results:
                            st.write("**Comparison Results:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Base Model Top Doc", results['base_model_top_doc'])
                            with col2:
                                st.metric("Fine-tuned Top Doc", results['finetuned_model_top_doc'])
                else:
                    st.error("Fine-tuned model not found. Train it first!")

with tab4:
    st.header("About This RAG System")

    st.markdown("""
    ### Features

    - **Document Processing**: Upload and process multiple PDF documents
    - **Semantic Search**: Advanced vector similarity search using FAISS
    - **Conversational AI**: Chat with your documents using Groq LLMs
    - **Memory**: Maintains conversation context across multiple queries
    - **Evaluation**: Comprehensive metrics for retrieval and generation quality
    - **Fine-tuning**: Custom embedding model training for better accuracy
    - **Persistence**: Save and load vector indices
    - **Proper Citations**: Accurate page number referencing (1-indexed)

    ### Tech Stack

    - **LangChain**: RAG pipeline orchestration
    - **FAISS**: Vector similarity search
    - **Groq**: Ultra-fast LLM inference
    - **Sentence Transformers**: Embeddings generation
    - **Streamlit**: Interactive web interface

    ### How to Use

    1. **Upload Documents**: Use the sidebar to upload PDF files
    2. **Process**: Click "Process" to create embeddings
    3. **Ask Questions**: Chat with your documents in natural language
    4. **View Sources**: Expand sources to see exact page numbers
    5. **Advanced**: Enable advanced features for evaluation and fine-tuning

    ### Models Available

    """)

    for name, model_id in AVAILABLE_MODELS.items():
        st.markdown(f"- **{name}**: `{model_id}`")

    st.divider()

    st.markdown("""
    ### Citation Format

    The system now properly cites sources with accurate page numbers:
    - Page numbers are displayed as 1-indexed (Page 1, Page 2, etc.)
    - Format: `[Document: filename.pdf, Page: X]`
    - Each answer includes source attribution with confidence scores
    """)

st.sidebar.divider()
st.sidebar.caption(f"Model: {st.session_state.current_model}")
st.sidebar.caption(f"Embeddings: {EMBEDDING_MODEL}")
