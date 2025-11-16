from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Tuple, Optional
import os
import json

class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Vector Store with embeddings model

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_store = None
        self.document_count = 0
        print(f"✓ Embedding model loaded successfully!")

    def create_vector_store(self, documents: List) -> None:
        """
        Create FAISS vector store from documents

        Args:
            documents: List of LangChain Document objects with text and metadata
        """
        if not documents:
            print("Error: No documents provided!")
            return

        print(f"Creating vector store from {len(documents)} documents...")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        self.document_count = len(documents)
        print(f"✓ Vector store created with {len(documents)} embeddings!")

    def add_documents(self, documents: List) -> None:
        """
        Add new documents to existing vector store

        Args:
            documents: List of document chunks to add
        """
        if not documents:
            print("No documents to add!")
            return

        if self.vector_store is None:
            self.create_vector_store(documents)
        else:
            print(f"Adding {len(documents)} new chunks to vector store...")
            self.vector_store.add_documents(documents)
            self.document_count += len(documents)
            print(f"✓ Chunks added! Total documents: {self.document_count}")

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple]:
        """
        Search for similar documents using semantic similarity

        Args:
            query: User's search query
            k: Number of top results to return

        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            print("Error: Vector store is empty!")
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def similarity_search_with_metadata(self, query: str, k: int = 5, 
                                       filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search and return results with formatted metadata

        Args:
            query: User's search query
            k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {'filename': 'doc.pdf'})

        Returns:
            List of dictionaries with content, metadata, and scores
        """
        results = self.similarity_search(query, k=k)

        formatted_results = []
        for doc, score in results:
            if filter_metadata:
                match = all(doc.metadata.get(key) == value 
                          for key, value in filter_metadata.items())
                if not match:
                    continue

            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'source': doc.metadata.get('filename', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'chunk_index': doc.metadata.get('chunk_index', 0)
            })

        return formatted_results

    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        """
        Get a LangChain retriever for RAG pipeline

        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity", "mmr")

        Returns:
            LangChain retriever object
        """
        if self.vector_store is None:
            print("Error: Vector store is empty!")
            return None

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def save_local(self, path: str = "faiss_index") -> None:
        """
        Save vector store to disk

        Args:
            path: Directory path to save the index
        """
        if self.vector_store is None:
            print("Error: No vector store to save!")
            return

        os.makedirs(path, exist_ok=True)
        self.vector_store.save_local(path)

        metadata = {
            'model_name': self.model_name,
            'document_count': self.document_count
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"✓ Vector store saved to {path}")

    def load_local(self, path: str = "faiss_index") -> bool:
        """
        Load vector store from disk

        Args:
            path: Directory path to load the index from

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist!")
            return False

        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            metadata_path = os.path.join(path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.document_count = metadata.get('document_count', 0)

            print(f"✓ Vector store loaded from {path}")
            print(f"  Documents: {self.document_count}")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def get_all_documents(self) -> List:
        """Get all documents from vector store"""
        if self.vector_store is None:
            return []
        return self.vector_store.docstore._dict.values()

    def get_unique_sources(self) -> List[str]:
        """Get list of unique source documents"""
        docs = self.get_all_documents()
        sources = set(doc.metadata.get('filename', 'Unknown') for doc in docs)
        return list(sources)

    def delete_by_source(self, filename: str) -> bool:
        """Delete all chunks from a specific source file"""

        print(f"Note: FAISS doesn't support direct deletion. Consider rebuilding index.")
        return False

    def get_statistics(self) -> Dict:
        """Get vector store statistics"""
        return {
            'total_documents': self.document_count,
            'embedding_model': self.model_name,
            'unique_sources': len(self.get_unique_sources()),
            'source_files': self.get_unique_sources()
        }
