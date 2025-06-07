#!/usr/bin/env python3
"""
Inference example for multilingual-e5-large-instruct model on Qualcomm devices.

This script demonstrates how to use the deployed model for text embedding tasks
including semantic similarity, retrieval, and multilingual processing.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse

# Add ExecutorCH to path if available
EXECUTORCH_ROOT = os.environ.get('EXECUTORCH_ROOT')
if EXECUTORCH_ROOT:
    sys.path.append(EXECUTORCH_ROOT)

try:
    from transformers import AutoTokenizer
    from executorch.runtime import Runtime
except ImportError as e:
    print(f"Import error: {e}")
    print("For host inference, install: pip install transformers")
    print("For device inference, ensure ExecutorCH is properly installed")


class E5InferenceEngine:
    """
    Inference engine for multilingual-e5-large-instruct model.
    Supports both host (original model) and device (exported .pte) inference.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_name: str = 'intfloat/multilingual-e5-large-instruct',
        device_inference: bool = False
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to .pte file for device inference or HF model for host
            tokenizer_name: Hugging Face tokenizer name
            device_inference: Whether to use device (.pte) or host inference
        """
        self.device_inference = device_inference
        self.max_length = 512
        
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if device_inference:
            # Load device model
            if not model_path or not model_path.endswith('.pte'):
                raise ValueError("Device inference requires a .pte model file")
            
            print(f"Loading device model: {model_path}")
            self.runtime = Runtime.get()
            self.program = self.runtime.load_program(model_path)
            self.method = self.program.load_method("forward")
            self.model = None
            
        else:
            # Load host model
            from transformers import AutoModel
            
            model_name = model_path if model_path else tokenizer_name
            print(f"Loading host model: {model_name}")
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            self.model.eval()
            self.runtime = None
            self.program = None
            self.method = None
        
        print("Inference engine initialized successfully")
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        Format query with instruction for the model.
        
        Args:
            task_description: Description of the task
            query: The actual query text
            
        Returns:
            Formatted instruction + query string
        """
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average pooling for sequence embeddings."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
    
    def encode_host(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using host model.
        
        Args:
            texts: List of input texts
            
        Returns:
            Normalized embeddings tensor
        """
        if self.model is None:
            raise ValueError("Host model not loaded")
        
        # Tokenize
        inputs = self.tokenize_texts(texts)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode_device(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using device model.
        
        Args:
            texts: List of input texts
            
        Returns:
            Normalized embeddings tensor
        """
        if self.method is None:
            raise ValueError("Device model not loaded")
        
        # Tokenize
        inputs = self.tokenize_texts(texts)
        
        # Run inference on device
        device_outputs = self.method.execute([inputs['input_ids'], inputs['attention_mask']])
        
        if len(device_outputs) == 0:
            raise RuntimeError("Device inference failed - no outputs")
        
        # Convert to tensor
        embeddings = torch.tensor(device_outputs[0])
        
        return embeddings
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of input texts
            
        Returns:
            Normalized embeddings tensor [batch_size, embedding_dim]
        """
        if self.device_inference:
            return self.encode_device(texts)
        else:
            return self.encode_host(texts)
    
    def encode_single(self, text: str) -> torch.Tensor:
        """Encode a single text."""
        return self.encode([text])[0]
    
    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        return torch.mm(embeddings1, embeddings2.t())
    
    def benchmark_inference(self, texts: List[str], iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            texts: Test texts
            iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        print(f"Benchmarking inference with {len(texts)} texts, {iterations} iterations...")
        
        # Warm up
        for _ in range(5):
            self.encode(texts)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            embeddings = self.encode(texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / iterations
        avg_time_per_text = avg_time_per_batch / len(texts)
        
        return {
            'total_time': total_time,
            'avg_time_per_batch': avg_time_per_batch,
            'avg_time_per_text': avg_time_per_text,
            'throughput': len(texts) * iterations / total_time
        }


def semantic_similarity_example(engine: E5InferenceEngine):
    """Example: Semantic similarity between queries and documents."""
    print("\n" + "="*50)
    print("SEMANTIC SIMILARITY EXAMPLE")
    print("="*50)
    
    # Task description
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    # Queries
    queries = [
        engine.get_detailed_instruct(task, 'How to optimize deep learning models for mobile devices?'),
        engine.get_detailed_instruct(task, 'What are the benefits of edge computing?'),
        engine.get_detailed_instruct(task, '如何在移动设备上部署AI模型？')  # Chinese
    ]
    
    # Documents (no instruction needed for documents)
    documents = [
        "Mobile optimization techniques include quantization, pruning, and knowledge distillation to reduce model size and improve inference speed on resource-constrained devices.",
        "Edge computing brings computation closer to data sources, reducing latency, bandwidth usage, and improving privacy by processing data locally.",
        "在移动设备上部署AI模型需要考虑模型压缩、量化和硬件加速等技术来优化性能。",  # Chinese
        "Cloud computing provides scalable resources for training large neural networks but may have higher latency for real-time applications."
    ]
    
    print(f"Encoding {len(queries)} queries...")
    query_embeddings = engine.encode(queries)
    
    print(f"Encoding {len(documents)} documents...")
    doc_embeddings = engine.encode(documents)
    
    # Compute similarities
    similarities = engine.compute_similarity(query_embeddings, doc_embeddings)
    
    # Display results
    print("\nSimilarity Matrix (Query x Document):")
    print("Queries:")
    for i, q in enumerate(queries):
        print(f"  Q{i+1}: {q[:60]}...")
    
    print("\nDocuments:")
    for i, d in enumerate(documents):
        print(f"  D{i+1}: {d[:60]}...")
    
    print("\nSimilarity Scores:")
    for i in range(len(queries)):
        print(f"\nQuery {i+1}:")
        for j in range(len(documents)):
            score = similarities[i][j].item()
            print(f"  Document {j+1}: {score:.4f}")
    
    # Find best matches
    print("\nBest Matches:")
    for i, query in enumerate(queries):
        best_doc_idx = torch.argmax(similarities[i]).item()
        best_score = similarities[i][best_doc_idx].item()
        print(f"Query {i+1} → Document {best_doc_idx+1} (score: {best_score:.4f})")


def multilingual_example(engine: E5InferenceEngine):
    """Example: Multilingual text embedding."""
    print("\n" + "="*50)
    print("MULTILINGUAL EXAMPLE")
    print("="*50)
    
    # Same query in different languages
    task = 'Given a question, retrieve relevant information'
    queries = [
        engine.get_detailed_instruct(task, 'What is artificial intelligence?'),
        engine.get_detailed_instruct(task, '¿Qué es la inteligencia artificial?'),  # Spanish
        engine.get_detailed_instruct(task, "Qu'est-ce que l'intelligence artificielle?"),  # French
        engine.get_detailed_instruct(task, '人工知能とは何ですか？'),  # Japanese
        engine.get_detailed_instruct(task, '什么是人工智能？'),  # Chinese
    ]
    
    languages = ['English', 'Spanish', 'French', 'Japanese', 'Chinese']
    
    print("Encoding multilingual queries...")
    embeddings = engine.encode(queries)
    
    # Compute cross-lingual similarities
    similarities = engine.compute_similarity(embeddings, embeddings)
    
    print("\nCross-lingual Similarity Matrix:")
    print("Languages: " + " | ".join(f"{lang:>8}" for lang in languages))
    
    for i, lang1 in enumerate(languages):
        row = f"{lang1:>8}: "
        for j in range(len(languages)):
            score = similarities[i][j].item()
            row += f"{score:>8.3f} "
        print(row)
    
    # Average cross-lingual similarity (excluding self-similarity)
    cross_lingual_scores = []
    for i in range(len(queries)):
        for j in range(len(queries)):
            if i != j:
                cross_lingual_scores.append(similarities[i][j].item())
    
    avg_cross_lingual = np.mean(cross_lingual_scores)
    print(f"\nAverage cross-lingual similarity: {avg_cross_lingual:.4f}")


def clustering_example(engine: E5InferenceEngine):
    """Example: Document clustering."""
    print("\n" + "="*50)
    print("DOCUMENT CLUSTERING EXAMPLE")
    print("="*50)
    
    # Technology topics
    task = 'Represent this document for clustering'
    documents = [
        engine.get_detailed_instruct(task, 'Machine learning algorithms can automatically learn patterns from data without explicit programming.'),
        engine.get_detailed_instruct(task, 'Deep neural networks use multiple layers to learn hierarchical representations of data.'),
        engine.get_detailed_instruct(task, 'Natural language processing enables computers to understand and generate human language.'),
        engine.get_detailed_instruct(task, 'Blockchain technology provides a decentralized ledger for secure transactions.'),
        engine.get_detailed_instruct(task, 'Cryptocurrency uses cryptographic techniques to secure digital financial transactions.'),
        engine.get_detailed_instruct(task, 'Smart contracts automatically execute agreements when predefined conditions are met.'),
        engine.get_detailed_instruct(task, 'Climate change is causing rising sea levels and extreme weather patterns globally.'),
        engine.get_detailed_instruct(task, 'Renewable energy sources like solar and wind are becoming more cost-effective.'),
        engine.get_detailed_instruct(task, 'Carbon footprint reduction requires changes in transportation and energy consumption.')
    ]
    
    labels = [
        'AI/ML', 'AI/ML', 'AI/ML',
        'Blockchain', 'Blockchain', 'Blockchain', 
        'Climate', 'Climate', 'Climate'
    ]
    
    print(f"Encoding {len(documents)} documents...")
    embeddings = engine.encode(documents)
    
    # Compute pairwise similarities
    similarities = engine.compute_similarity(embeddings, embeddings)
    
    print("\nDocument Similarity Matrix:")
    for i in range(len(documents)):
        row = f"Doc {i+1} ({labels[i]:>9}): "
        for j in range(len(documents)):
            score = similarities[i][j].item()
            row += f"{score:>6.3f} "
        print(row)
    
    # Compute cluster coherence
    clusters = {'AI/ML': [], 'Blockchain': [], 'Climate': []}
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    print("\nCluster Coherence:")
    for cluster_name, indices in clusters.items():
        intra_cluster_scores = []
        for i in indices:
            for j in indices:
                if i != j:
                    intra_cluster_scores.append(similarities[i][j].item())
        
        avg_intra_cluster = np.mean(intra_cluster_scores) if intra_cluster_scores else 0
        print(f"{cluster_name}: {avg_intra_cluster:.4f}")


def performance_benchmark(engine: E5InferenceEngine):
    """Benchmark inference performance."""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Test cases with different batch sizes and text lengths
    test_cases = [
        {
            'name': 'Single short query',
            'texts': ['How to optimize models for mobile?'],
            'iterations': 100
        },
        {
            'name': 'Batch of 4 queries',
            'texts': [
                'How to optimize models for mobile?',
                'What is edge computing?',
                'Best practices for AI deployment',
                'Mobile AI performance tuning'
            ],
            'iterations': 50
        },
        {
            'name': 'Long document',
            'texts': [
                'Artificial intelligence has revolutionized many industries by providing automated solutions to complex problems. ' +
                'Machine learning algorithms can process vast amounts of data to identify patterns and make predictions. ' +
                'Deep learning, a subset of machine learning, uses neural networks with multiple layers to learn hierarchical ' +
                'representations of data. Natural language processing enables computers to understand, interpret, and generate ' +
                'human language, while computer vision allows machines to analyze and understand visual information.'
            ],
            'iterations': 50
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print(f"Texts: {len(case['texts'])}, Iterations: {case['iterations']}")
        
        # Add instruction to texts
        task = 'Given a text, create a semantic embedding representation'
        formatted_texts = [engine.get_detailed_instruct(task, text) for text in case['texts']]
        
        metrics = engine.benchmark_inference(formatted_texts, case['iterations'])
        
        print(f"Results:")
        print(f"  Average time per batch: {metrics['avg_time_per_batch']*1000:.2f} ms")
        print(f"  Average time per text:  {metrics['avg_time_per_text']*1000:.2f} ms")
        print(f"  Throughput:            {metrics['throughput']:.2f} texts/sec")
        
        results.append({
            'name': case['name'],
            'batch_size': len(case['texts']),
            **metrics
        })
    
    print(f"\nSummary:")
    print(f"{'Test Case':<20} {'Batch':<5} {'ms/batch':<10} {'ms/text':<10} {'texts/sec':<10}")
    print("-" * 60)
    for result in results:
        print(f"{result['name']:<20} {result['batch_size']:<5} "
              f"{result['avg_time_per_batch']*1000:<10.2f} "
              f"{result['avg_time_per_text']*1000:<10.2f} "
              f"{result['throughput']:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Multilingual E5 Inference Examples")
    parser.add_argument(
        '--model',
        type=str,
        help="Model path (.pte for device, HF model name for host)"
    )
    parser.add_argument(
        '--device',
        action='store_true',
        help="Use device inference (.pte model)"
    )
    parser.add_argument(
        '--host',
        action='store_true',
        help="Use host inference (original model)"
    )
    parser.add_argument(
        '--examples',
        nargs='+',
        choices=['similarity', 'multilingual', 'clustering', 'benchmark', 'all'],
        default=['all'],
        help="Examples to run"
    )
    
    args = parser.parse_args()
    
    # Determine inference mode
    if args.device and args.host:
        print("Error: Cannot specify both --device and --host")
        sys.exit(1)
    
    device_inference = args.device
    if not args.device and not args.host:
        # Auto-detect based on model path
        if args.model and args.model.endswith('.pte'):
            device_inference = True
        else:
            device_inference = False
    
    print("Multilingual E5 Inference Examples")
    print("=" * 50)
    print(f"Inference mode: {'Device (.pte)' if device_inference else 'Host (HF)'}")
    print(f"Model: {args.model or 'Default'}")
    
    # Initialize engine
    try:
        engine = E5InferenceEngine(
            model_path=args.model,
            device_inference=device_inference
        )
    except Exception as e:
        print(f"Failed to initialize inference engine: {e}")
        sys.exit(1)
    
    # Run examples
    examples_to_run = args.examples
    if 'all' in examples_to_run:
        examples_to_run = ['similarity', 'multilingual', 'clustering', 'benchmark']
    
    for example in examples_to_run:
        try:
            if example == 'similarity':
                semantic_similarity_example(engine)
            elif example == 'multilingual':
                multilingual_example(engine)
            elif example == 'clustering':
                clustering_example(engine)
            elif example == 'benchmark':
                performance_benchmark(engine)
        except Exception as e:
            print(f"Error running {example} example: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("Examples completed!")


if __name__ == "__main__":
    main() 