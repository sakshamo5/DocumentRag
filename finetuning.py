from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import json
import os

class EmbeddingFineTuner:
    """Fine-tune embedding models for better domain-specific retrieval"""

    def __init__(self, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize fine-tuner
        Args:
            base_model_name: Base model to fine-tune
        """
        self.base_model_name = base_model_name
        self.model = SentenceTransformer(base_model_name)
        self.training_data = []
        print(f"✓ Loaded base model: {base_model_name}")

    def generate_training_data_from_docs(self, documents: List,
                                         llm,
                                         num_pairs: int = 50) -> List[Dict]:
        """
        Generate training data (query, positive, negative) from documents
        Args:
            documents: List of document chunks
            llm: LLM instance for generating queries
            num_pairs: Number of training pairs to generate
        Returns:
            List of training examples
        """
        print(f"Generating {num_pairs} training pairs...")
        training_data = []
        docs_list = list(documents)

        for i in range(min(num_pairs, len(docs_list))):
            positive_doc = docs_list[i]
            positive_text = positive_doc.page_content

            query_prompt = f"""Generate a specific question that would be answered by the following text.
Only provide the question, nothing else.

Text: {positive_text[:500]}

Question:"""

            try:
                response = llm.invoke(query_prompt)
                query = response.content.strip()

                negative_idx = (i + len(docs_list) // 2) % len(docs_list)
                negative_doc = docs_list[negative_idx]
                negative_text = negative_doc.page_content

                training_data.append({
                    'query': query,
                    'positive': positive_text[:1000],
                    'negative': negative_text[:1000]
                })

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{num_pairs} pairs")

            except Exception as e:
                print(f"Error generating pair {i}: {e}")
                continue

        self.training_data = training_data
        print(f"✓ Generated {len(training_data)} training pairs")
        return training_data

    def save_training_data(self, filepath: str = "training_data.json") -> None:
        """Save training data to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"✓ Training data saved to {filepath}")

    def load_training_data(self, filepath: str = "training_data.json") -> List[Dict]:
        """Load training data from JSON"""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []

        with open(filepath, 'r') as f:
            self.training_data = json.load(f)
        print(f"✓ Loaded {len(self.training_data)} training pairs")
        return self.training_data

    def prepare_training_examples(self) -> List[InputExample]:
        """Convert training data to InputExample format"""
        examples = []

        for item in self.training_data:
            examples.append(InputExample(
                texts=[item['query'], item['positive']],
                label=1.0
            ))

            examples.append(InputExample(
                texts=[item['query'], item['negative']],
                label=0.0
            ))

        print(f"✓ Prepared {len(examples)} training examples")
        return examples

    def fine_tune(self, output_path: str = "models/finetuned_embeddings",
                  epochs: int = 3, batch_size: int = 16,
                  warmup_steps: int = 100) -> None:
        """
        Fine-tune the embedding model
        Args:
            output_path: Path to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_steps: Number of warmup steps
        """
        if not self.training_data:
            print("No training data available. Generate or load training data first!")
            return

        print(f"\nStarting fine-tuning...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training examples: {len(self.training_data)}")

        train_examples = self.prepare_training_examples()

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )

        train_loss = losses.CosineSimilarityLoss(self.model)

        print("\nTraining in progress...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )

        print(f"\n✓ Fine-tuning complete!")
        print(f"  Model saved to: {output_path}")

    def evaluate_model(self, test_pairs: List[Tuple[str, str, float]]) -> float:
        """
        Evaluate model performance
        Args:
            test_pairs: List of (text1, text2, similarity_score) tuples
        Returns:
            Evaluation score
        """
        if not test_pairs:
            print("No test pairs provided")
            return 0.0

        sentences1 = [pair[0] for pair in test_pairs]
        sentences2 = [pair[1] for pair in test_pairs]
        scores = [pair[2] for pair in test_pairs]

        evaluator = EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores
        )

        score = evaluator(self.model)
        print(f"Evaluation score: {score:.4f}")
        return score

    def compare_models(self, query: str, documents: List[str],
                      finetuned_model_path: str) -> Dict:
        """
        Compare base model vs fine-tuned model
        Args:
            query: Test query
            documents: List of document texts
            finetuned_model_path: Path to fine-tuned model
        Returns:
            Comparison results
        """
        base_query_emb = self.model.encode(query)
        base_doc_embs = self.model.encode(documents)

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        base_sims = cosine_similarity([base_query_emb], base_doc_embs)[0]

        try:
            finetuned_model = SentenceTransformer(finetuned_model_path)
            ft_query_emb = finetuned_model.encode(query)
            ft_doc_embs = finetuned_model.encode(documents)
            ft_sims = cosine_similarity([ft_query_emb], ft_doc_embs)[0]

            results = {
                'query': query,
                'base_model_similarities': base_sims.tolist(),
                'finetuned_model_similarities': ft_sims.tolist(),
                'base_model_top_doc': int(np.argmax(base_sims)),
                'finetuned_model_top_doc': int(np.argmax(ft_sims))
            }

            print(f"\nModel Comparison:")
            print(f"  Base model top doc: {results['base_model_top_doc']}")
            print(f"  Fine-tuned top doc: {results['finetuned_model_top_doc']}")

            return results

        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            return {}
