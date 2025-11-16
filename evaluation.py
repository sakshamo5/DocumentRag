import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, ndcg_score
import os

class RAGEvaluator:
    """Evaluation utilities for RAG system"""

    def __init__(self, rag_pipeline, vector_store):
        self.rag_pipeline = rag_pipeline
        self.vector_store = vector_store
        self.test_set = []

    def generate_test_set(self, num_questions: int = 20) -> List[Dict]:
        """
        Generate synthetic test Q&A pairs using LLM

        Args:
            num_questions: Number of test questions to generate

        Returns:
            List of test question-answer pairs
        """
        print(f"Generating {num_questions} test questions...")

        all_docs = list(self.vector_store.get_all_documents())

        if not all_docs:
            print("No documents available for test generation!")
            return []

        test_set = []
        docs_per_question = max(1, len(all_docs) // num_questions)

        for i in range(0, min(len(all_docs), num_questions * docs_per_question), docs_per_question):
            doc = all_docs[i]
            context = doc.page_content

            prompt = f"""Based on the following text, generate a question and its answer.

Text: {context[:1000]}

Generate a JSON response with this format:
{{
    "question": "your question here",
    "answer": "the answer based on the text",
    "context": "relevant excerpt from text"
}}"""

            try:
                response = self.rag_pipeline.llm.invoke(prompt)
                content = response.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    qa_pair = json.loads(json_str)
                    qa_pair['source_doc'] = doc.metadata.get('filename', 'Unknown')
                    qa_pair['source_page'] = doc.metadata.get('page', 0)
                    test_set.append(qa_pair)

                    if len(test_set) >= num_questions:
                        break
            except Exception as e:
                print(f"Error generating question {len(test_set) + 1}: {e}")
                continue

        self.test_set = test_set
        print(f"Generated {len(test_set)} test questions")
        return test_set

    def save_test_set(self, filepath: str = "test_set.json") -> None:
        """Save test set to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.test_set, f, indent=2)
        print(f"Test set saved to {filepath}")

    def load_test_set(self, filepath: str = "test_set.json") -> List[Dict]:
        """Load test set from JSON file"""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []

        with open(filepath, 'r') as f:
            self.test_set = json.load(f)
        print(f"Loaded {len(self.test_set)} test questions")
        return self.test_set

    def evaluate_retrieval(self, k: int = 5) -> Dict:
        """
        Evaluate retrieval quality using test set

        Args:
            k: Number of documents to retrieve

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.test_set:
            print("No test set available. Generate one first!")
            return {}

        print(f"Evaluating retrieval with k={k}...")

        precisions = []
        recalls = []
        ndcg_scores = []
        mrr_scores = []

        for qa in self.test_set:
            query = qa['question']
            expected_doc = qa.get('source_doc', '')

            retrieved = self.rag_pipeline.retrieve_documents(query, k=k)

            if not retrieved:
                continue

            retrieved_docs = [r['source'] for r in retrieved]
            relevant = [1 if doc == expected_doc else 0 for doc in retrieved_docs]

            precision = sum(relevant) / k if k > 0 else 0
            precisions.append(precision)

            recall = 1.0 if sum(relevant) > 0 else 0.0
            recalls.append(recall)

            for idx, rel in enumerate(relevant, 1):
                if rel == 1:
                    mrr_scores.append(1.0 / idx)
                    break
            else:
                mrr_scores.append(0.0)

        results = {
            'precision@k': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'num_queries': len(self.test_set),
            'k': k
        }

        print(f"\nRetrieval Evaluation Results:")
        print(f"  Precision@{k}: {results['precision@k']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  MRR: {results['mrr']:.4f}")

        return results

    def evaluate_generation(self, sample_size: int = 10) -> Dict:
        """
        Evaluate generation quality

        Args:
            sample_size: Number of questions to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.test_set:
            print("No test set available. Generate one first!")
            return {}

        print(f"Evaluating generation quality on {sample_size} samples...")

        sample_questions = self.test_set[:sample_size]
        scores = []

        for qa in sample_questions:
            question = qa['question']
            expected_answer = qa['answer']

            result = self.rag_pipeline.generate_answer(question, k=5)
            generated_answer = result['answer']

            eval_prompt = f"""Rate the following answer on a scale of 0-10 based on relevance and accuracy.

Question: {question}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

Provide only a number from 0-10 as your response."""

            try:
                response = self.rag_pipeline.llm.invoke(eval_prompt)
                score_text = response.content.strip()
                score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                score = min(10.0, max(0.0, score))  # Clamp to 0-10
                scores.append(score)
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                continue

        avg_score = np.mean(scores) if scores else 0.0

        results = {
            'average_score': avg_score,
            'max_score': 10.0,
            'num_evaluated': len(scores)
        }

        print(f"\nGeneration Evaluation Results:")
        print(f"  Average Score: {avg_score:.2f}/10")
        print(f"  Questions Evaluated: {len(scores)}")

        return results

    def save_evaluation_results(self, retrieval_results: Dict, 
                                generation_results: Dict, 
                                filepath: str = "evaluation_results.json") -> None:
        """Save evaluation results to file"""
        results = {
            'retrieval': retrieval_results,
            'generation': generation_results,
            'test_set_size': len(self.test_set)
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Evaluation results saved to {filepath}")
