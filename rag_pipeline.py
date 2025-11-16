from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from typing import List, Dict, Tuple
from config import GROQ_API_KEY
import json

class ConversationMemory:
    """Simple conversation memory implementation"""

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(("user", message))

    def add_ai_message(self, message: str):
        self.messages.append(("assistant", message))

    def clear(self):
        self.messages = []

    def get_messages(self) -> List[Tuple[str, str]]:
        return self.messages

class RAGPipeline:
    def __init__(self, vector_store, model_name: str = "llama-3.3-70b-versatile", 
                 use_memory: bool = True):
        """
        Initialize RAG pipeline with Groq LLM and conversation memory

        Args:
            vector_store: Initialized VectorStore object
            model_name: Groq model name
            use_memory: Whether to use conversation memory
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.use_memory = use_memory

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=0.2,
            max_tokens=2048,
            timeout=60,
            max_retries=2
        )

        if use_memory:
            self.memory = ConversationMemory()
        else:
            self.memory = None

        self.qa_prompt = self._create_qa_prompt()
        self.conversational_prompt = self._create_conversational_prompt()

        print(f"✓ RAG Pipeline initialized with model: {model_name}")
        print(f"✓ Conversation memory: {'Enabled' if use_memory else 'Disabled'}")

    def _create_qa_prompt(self) -> PromptTemplate:
        """Create a prompt template for Q&A without conversation history"""
        template = """You are a helpful AI assistant. Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say so. Don't make up information.
Always cite which document and page your answer comes from using the format [Document: filename, Page: X].

Context:
{context}

Question: {question}

Detailed Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_conversational_prompt(self) -> ChatPromptTemplate:
        """Create a prompt template for conversational RAG"""
        template = """You are a helpful AI assistant. Use the context and conversation history to answer questions.

Context from documents:
{context}

Previous conversation:
{chat_history}

Current question: {question}

Provide a detailed answer, citing sources using the format [Document: filename, Page: X] when possible."""

        return ChatPromptTemplate.from_template(template)

    def update_model(self, model_name: str) -> None:
        """Update the LLM model"""
        self.model_name = model_name
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=0.2,
            max_tokens=2048,
            timeout=60,
            max_retries=2
        )
        print(f"✓ Model updated to: {model_name}")

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from vector store

        Args:
            query: User's question
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents with metadata
        """
        if self.vector_store.vector_store is None:
            print("Error: Vector store is empty!")
            return []

        results = self.vector_store.similarity_search_with_metadata(query, k=k)
        return results

    def format_docs(self, docs: List[Dict]) -> str:
        """Format retrieved documents for context with proper page numbers"""
        formatted_context = []

        for idx, doc in enumerate(docs, 1):
            source = doc['metadata'].get('filename', 'Unknown')
            page = doc['metadata'].get('page', 0)
            page_display = page + 1 if isinstance(page, int) else page
            content = doc['content']

            formatted_context.append(
                f"[Document {idx}: {source}, Page {page_display}]\n{content}\n"
            )

        return "\n".join(formatted_context)

    def format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        """Format chat history for prompt"""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for human, ai in chat_history:
            formatted.append(f"Human: {human}\nAssistant: {ai}")

        return "\n\n".join(formatted)

    def generate_answer(self, query: str, k: int = 5, 
                       chat_history: List[Tuple[str, str]] = None) -> Dict:
        """
        Generate answer using RAG pipeline with optional conversation history

        Args:
            query: User's question
            k: Number of documents to retrieve
            chat_history: Optional list of (human, ai) message tuples

        Returns:
            Dictionary with answer, sources, and metadata
        """
        retrieved_docs = self.retrieve_documents(query, k=k)

        if not retrieved_docs:
            return {
                'answer': "I don't have any documents to answer from. Please upload documents first.",
                'sources': [],
                'query': query,
                'confidence': 0.0
            }

        context = self.format_docs(retrieved_docs)

        if chat_history and self.use_memory:
            formatted_history = self.format_chat_history(chat_history)
            prompt = self.conversational_prompt.format(
                context=context,
                chat_history=formatted_history,
                question=query
            )
        else:
            prompt = self.qa_prompt.format(
                context=context,
                question=query
            )

        try:
            response = self.llm.invoke(prompt)
            answer = response.content

            avg_score = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
            confidence = max(0.0, min(1.0, 1.0 - (avg_score / 2.0)))  # Normalize score

        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            confidence = 0.0

        return {
            'answer': answer,
            'sources': retrieved_docs,
            'query': query,
            'confidence': confidence,
            'model': self.model_name
        }

    def ask(self, query: str, k: int = 5, chat_history: List[Tuple[str, str]] = None) -> str:
        """
        Simple interface to ask a question

        Args:
            query: User's question
            k: Number of documents to retrieve
            chat_history: Optional conversation history

        Returns:
            Generated answer as string
        """
        result = self.generate_answer(query, k=k, chat_history=chat_history)
        return result['answer']

    def summarize_document(self, filename: str = None) -> str:
        """
        Generate a summary of a document or all documents

        Args:
            filename: Optional specific filename to summarize

        Returns:
            Summary text
        """
        if self.vector_store.vector_store is None:
            return "No documents available."

        all_docs = list(self.vector_store.get_all_documents())

        if filename:
            all_docs = [doc for doc in all_docs 
                       if doc.metadata.get('filename') == filename]

        if not all_docs:
            return f"No documents found{' for ' + filename if filename else ''}."

        combined_text = "\n\n".join([doc.page_content for doc in all_docs[:10]])

        prompt = f"""Provide a comprehensive summary of the following document content:

{combined_text}

Summary:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def suggest_questions(self, num_questions: int = 3) -> List[str]:
        """
        Suggest relevant questions based on available documents

        Args:
            num_questions: Number of questions to suggest

        Returns:
            List of suggested questions
        """
        if self.vector_store.vector_store is None:
            return []

        all_docs = list(self.vector_store.get_all_documents())
        sample_docs = all_docs[:5]

        combined_text = "\n\n".join([doc.page_content[:500] for doc in sample_docs])

        prompt = f"""Based on the following document content, suggest {num_questions} interesting and relevant questions that could be asked:

{combined_text}

Generate exactly {num_questions} questions, one per line."""

        try:
            response = self.llm.invoke(prompt)
            questions = [q.strip() for q in response.content.split('\n') if q.strip() and '?' in q]
            return questions[:num_questions]
        except Exception as e:
            print(f"Error suggesting questions: {e}")
            return []

    def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("✓ Conversation memory cleared")

    def get_memory_summary(self) -> str:
        """Get summary of conversation history"""
        if not self.memory:
            return "Memory not enabled."

        try:
            messages = self.memory.get_messages()
            return f"Conversation has {len(messages)} messages."
        except:
            return "No conversation history."
