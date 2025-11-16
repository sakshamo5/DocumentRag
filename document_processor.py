from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict
import os
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size=600, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.processed_files = {}

    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect duplicates"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def load_pdf(self, file_path: str) -> List[Dict]:
        """Load a single PDF and return documents with metadata"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []

    def process_file(self, file_path: str) -> List[Dict]:
        """
        Process a single PDF file and return chunks
        Args:
            file_path: Path to PDF file
        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return []

        file_hash = self.get_file_hash(file_path)
        if file_hash in self.processed_files:
            print(f"File already processed: {file_path}")
            return self.processed_files[file_hash]

        documents = self.load_pdf(file_path)
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata['source'] = os.path.basename(file_path)
            chunk.metadata['file_hash'] = file_hash

        self.processed_files[file_hash] = chunks
        print(f"✓ Processed {file_path}: {len(chunks)} chunks")
        return chunks

    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all PDFs in a directory
        Args:
            directory_path: Path to directory containing PDFs
        Returns:
            List of all document chunks
        """
        all_chunks = []

        if not os.path.exists(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return []

        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return []

        print(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

        print(f"\n✓ Total chunks from all files: {len(all_chunks)}")
        return all_chunks

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        total_files = len(self.processed_files)
        total_chunks = sum(len(chunks) for chunks in self.processed_files.values())

        return {
            'total_files': total_files,
            'total_chunks': total_chunks,
            'avg_chunks_per_file': total_chunks / total_files if total_files > 0 else 0
        }
