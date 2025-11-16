from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


print("Initializing components...")
doc_processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
vector_store = VectorStore(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_pipeline = RAGPipeline(vector_store, model_name="openai/gpt-oss-20b")


print("\nProcessing documents...")
chunks = doc_processor.process_file("sample.pdf")
print(f"Created {len(chunks)} chunks")


print("\nCreating embeddings...")
vector_store.create_vector_store(chunks)


print("\n=== Testing Retrieval ===")
test_query = "What is machine learning?"
retrieved = rag_pipeline.retrieve_documents(test_query, k=3)
print(f"\nQuery: {test_query}")
print(f"Retrieved {len(retrieved)} documents:\n")

for idx, doc in enumerate(retrieved, 1):
    print(f"--- Result {idx} ---")
    print(f"Score: {doc['score']:.4f}")
    print(f"Source: {doc['source']}, Page: {doc['page']}")
    print(f"Content: {doc['content'][:150]}...\n")


print("\n=== Testing RAG Generation ===")
result = rag_pipeline.generate_answer(test_query, k=3)
print(f"\nQuery: {result['query']}")
print(f"\nAnswer:\n{result['answer']}")
print(f"\nSources used: {len(result['sources'])}")


print("\n=== Simple Ask Interface ===")
answer = rag_pipeline.ask("Explain deep learning in simple terms")
print(answer)