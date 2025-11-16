# Advanced RAG System with Groq

A production-ready Retrieval-Augmented Generation (RAG) system with conversational memory, evaluation metrics, and embedding fine-tuning capabilities.

##  Features

- **Multi-Document Processing**: Upload and process multiple PDF documents simultaneously
- **Semantic Search**: FAISS-powered vector similarity search
- **Conversational AI**: Chat with documents using Groq's ultra-fast LLMs
- **Memory Management**: Maintains context across multiple conversation turns
- **Model Selection**: Switch between 7+ Groq models on the fly
- **Evaluation Suite**: Comprehensive metrics for retrieval and generation quality
- **Fine-tuning**: Custom embedding model training for domain-specific improvements
- **Persistent Storage**: Save and load vector indices
- **Source Citations**: Automatic source attribution with page numbers

##  Requirements

- Python 3.8+
- Groq API key (get one at [console.groq.com](https://console.groq.com))

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <https://github.com/sakshamo5/DocumentRag>
cd rag-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env_template .env
# Edit .env and add your GROQ_API_KEY
```

##  Quick Start

### Basic Usage

```bash
streamlit run app.py
```

Then:
1. Upload PDF documents using the sidebar
2. Click "Process" to create embeddings
3. Ask questions in the chat interface

### Programmatic Usage

```python
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Initialize components
processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
vector_store = VectorStore()
rag = RAGPipeline(vector_store, model_name="llama-3.3-70b-versatile")

# Process documents
chunks = processor.process_file("document.pdf")
vector_store.create_vector_store(chunks)

# Ask questions
answer = rag.ask("What is the main topic?")
print(answer)
```

##  Project Structure

```
rag-system/
├── config.py                 # Configuration and available models
├── document_processor.py     # PDF loading and text chunking
├── vector_store.py          # FAISS vector store operations
├── rag_pipeline.py          # RAG logic with conversational memory
├── evaluation.py            # Evaluation metrics and test generation
├── finetuning.py           # Embedding model fine-tuning
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from template)
└── README.md              # This file
```

## Available Models

The system supports multiple Groq models:

- **Llama 3.3 70B**: Best overall performance (default)
- **Llama 3.1 70B & 8B**: High quality alternatives
- **Mixtral 8x7B**: Great for long context (32k tokens)
- **Gemma 2 9B**: Fastest, lightweight option

Switch models in the UI or update `DEFAULT_MODEL` in `config.py`.

##  Configuration

Edit `config.py` to customize:

```python
# Model settings
DEFAULT_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# Vector store settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval settings
DEFAULT_K = 5  # Number of chunks to retrieve
```

##  Evaluation

### Generate Test Set

```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator(rag_pipeline, vector_store)
test_set = evaluator.generate_test_set(num_questions=30)
evaluator.save_test_set("test_set.json")
```

### Run Evaluation

```python
# Evaluate retrieval
retrieval_results = evaluator.evaluate_retrieval(k=5)

# Evaluate generation
generation_results = evaluator.evaluate_generation(sample_size=10)

# Save results
evaluator.save_evaluation_results(retrieval_results, generation_results)
```

## Fine-tuning Embeddings

### Generate Training Data

```python
from finetuning import EmbeddingFineTuner

finetuner = EmbeddingFineTuner()
docs = vector_store.get_all_documents()
training_data = finetuner.generate_training_data_from_docs(
    docs, 
    rag_pipeline.llm,
    num_pairs=100
)
finetuner.save_training_data("training_data.json")
```

### Train Model

```python
finetuner.fine_tune(
    output_path="models/finetuned_embeddings",
    epochs=3,
    batch_size=16
)
```

### Use Fine-tuned Model

Update `config.py`:
```python
EMBEDDING_MODEL = "models/finetuned_embeddings"
```

## Advanced Features

### Conversational Memory

```python
# The system automatically maintains conversation context
rag = RAGPipeline(vector_store, use_memory=True)

# Chat history is maintained across queries
chat_history = [
    ("What is machine learning?", "Machine learning is..."),
    ("Tell me more about that", "Expanding on ML...")
]

answer = rag.generate_answer("How is it used?", chat_history=chat_history)
```

### Document Summarization

```python
# Summarize all documents
summary = rag.summarize_document()

# Summarize specific document
summary = rag.summarize_document(filename="document.pdf")
```

### Question Suggestions

```python
# Get suggested questions based on documents
questions = rag.suggest_questions(num_questions=5)
```

### Metadata Filtering

```python
# Search within specific document
results = vector_store.similarity_search_with_metadata(
    "query",
    k=5,
    filter_metadata={'filename': 'specific_document.pdf'}
)
```

##  Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **GROQ_API_KEY not found**: Check your `.env` file
   ```bash
   cat .env  # Should show GROQ_API_KEY=...
   ```

3. **FAISS errors on GPU**: If using CPU, install `faiss-cpu` instead of `faiss-gpu`
   ```bash
   pip uninstall faiss-gpu
   pip install faiss-cpu
   ```

4. **Out of memory**: Reduce `CHUNK_SIZE` or process fewer documents at once

##  Performance Tips

1. **GPU Acceleration**: Use `faiss-gpu` and set `device='cuda'` in `vector_store.py`
2. **Batch Processing**: Process multiple files at once for efficiency
3. **Persistence**: Save the vector index to avoid reprocessing
4. **Model Selection**: Use smaller models (Gemma 2 9B) for faster responses
5. **Chunk Optimization**: Experiment with chunk size (500-1000 chars works well)

##  Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for more document formats (Word, TXT, etc.)
- [ ] Multi-language support
- [ ] Advanced retrieval strategies (hybrid search, reranking)
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] API endpoint wrapper
- [ ] Batch evaluation tools

##  License

MIT License - feel free to use this project for learning or commercial purposes.

##  Acknowledgments

- [LangChain](https://python.langchain.com/) for RAG orchestration
- [Groq](https://groq.com/) for ultra-fast LLM inference
- [FAISS](https://faiss.ai/) for efficient similarity search
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## Contact

For questions or feedback, please open an issue on GitHub.

---

