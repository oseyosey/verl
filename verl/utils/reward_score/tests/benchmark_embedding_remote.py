#!/usr/bin/env python3
"""
Performance benchmark for remote embedding reward function.

This script benchmarks the performance of the remote embedding server
and compares it with local alternatives.

Usage:
    python benchmark_embedding_remote.py [--server-url URL] [--iterations N]
"""

import os
import sys
import time
import argparse
import statistics
from typing import List, Dict, Any, Tuple
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from embedding_remote import compute_score, compute_score_batched
    from embedding_client import EmbeddingClient
    from embedding import compute_score as compute_score_local
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


class BenchmarkSuite:
    """Benchmark suite for embedding performance."""
    
    def __init__(self, server_url: str = None):
        """Initialize benchmark suite."""
        self.server_url = server_url or os.environ.get("EMBEDDING_SERVER_URL")
        self.results = {}
        
        # Test data
        self.test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is revolutionizing many industries.",
            "Python is a versatile programming language.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models can achieve remarkable accuracy.",
            "Text embeddings capture semantic meaning effectively.",
            "Transfer learning accelerates model development.",
            "Attention mechanisms improved transformer architectures.",
            "BERT revolutionized natural language understanding.",
            "GPT models excel at text generation tasks."
        ]
        
        self.reference_sentences = [
            "A swift fox leaps across a resting canine.",
            "AI is transforming various sectors dramatically.",
            "Python offers great flexibility for developers.",
            "NLP helps machines comprehend human language.",
            "Neural networks can deliver impressive results.",
            "Embeddings represent text meaning numerically.",
            "Pre-trained models speed up development significantly.",
            "Self-attention enhanced neural network performance.",
            "BERT transformed how we process language.",
            "GPT architectures are excellent text generators."
        ]
    
    def benchmark_single_requests(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark single request performance."""
        print(f"\nBenchmarking single requests ({iterations} iterations)...")
        
        client = EmbeddingClient(server_url=self.server_url)
        times = []
        
        for i in range(iterations):
            idx = i % len(self.test_sentences)
            
            start_time = time.time()
            score = compute_score(
                data_source="embedding_remote_test",
                solution_str=self.test_sentences[idx],
                ground_truth=self.reference_sentences[idx],
                extra_info={"server_url": self.server_url}
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{iterations}", end="\r")
        
        print()
        return self._calculate_stats(times, "Single Request")
    
    def benchmark_batch_requests(self, batch_sizes: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Any]:
        """Benchmark batch request performance."""
        print("\nBenchmarking batch requests...")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            times = []
            throughputs = []
            
            # Prepare batch data
            solutions = (self.test_sentences * (batch_size // len(self.test_sentences) + 1))[:batch_size]
            references = (self.reference_sentences * (batch_size // len(self.reference_sentences) + 1))[:batch_size]
            
            # Run multiple iterations
            iterations = max(10, 100 // batch_size)
            
            for i in range(iterations):
                start_time = time.time()
                scores = compute_score_batched(
                    data_sources=["test"] * batch_size,
                    solution_strs=solutions,
                    ground_truths=references,
                    extra_infos=[{"server_url": self.server_url}] * batch_size
                )
                elapsed = time.time() - start_time
                
                times.append(elapsed)
                throughputs.append(batch_size / elapsed)
                
                if (i + 1) % 5 == 0:
                    print(f"    Progress: {i + 1}/{iterations}", end="\r")
            
            print()
            stats = self._calculate_stats(times, f"Batch Size {batch_size}")
            stats["throughput"] = {
                "mean": statistics.mean(throughputs),
                "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                "max": max(throughputs)
            }
            results[f"batch_{batch_size}"] = stats
        
        return results
    
    def benchmark_embedding_dimensions(self) -> Dict[str, Any]:
        """Benchmark impact of embedding dimensions."""
        print("\nBenchmarking embedding retrieval...")
        
        client = EmbeddingClient(server_url=self.server_url)
        
        # Test different text lengths
        text_lengths = [10, 50, 100, 500, 1000]  # words
        results = {}
        
        for length in text_lengths:
            # Generate text of specific length
            text = " ".join(["word"] * length)
            
            times = []
            for _ in range(20):
                start_time = time.time()
                embeddings = client.embed_texts([text])
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            stats = self._calculate_stats(times, f"{length} words")
            if embeddings is not None:
                stats["embedding_dim"] = embeddings.shape[1]
            results[f"length_{length}"] = stats
        
        return results
    
    def benchmark_cache_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark cache hit performance."""
        print(f"\nBenchmarking cache performance ({iterations} iterations)...")
        
        # Use only 5 unique sentences, repeated
        unique_sentences = self.test_sentences[:5]
        
        times_first = []
        times_cached = []
        
        # Create client with cache enabled
        client = EmbeddingClient(
            server_url=self.server_url,
            enable_cache=True,
            cache_size=10
        )
        
        for i in range(iterations):
            sentence = unique_sentences[i % len(unique_sentences)]
            
            # Clear cache every N iterations to test both paths
            if i % len(unique_sentences) == 0 and i > 0:
                # Re-create client to clear cache
                client = EmbeddingClient(
                    server_url=self.server_url,
                    enable_cache=True,
                    cache_size=10
                )
            
            start_time = time.time()
            score = compute_score(
                data_source="embedding_remote_test",
                solution_str=sentence,
                ground_truth=self.reference_sentences[0],
                extra_info={"server_url": self.server_url}
            )
            elapsed = time.time() - start_time
            
            # First time seeing this sentence in current cache
            if i % len(unique_sentences) < len(unique_sentences):
                times_first.append(elapsed)
            else:
                times_cached.append(elapsed)
        
        return {
            "first_request": self._calculate_stats(times_first, "First Request"),
            "cached_request": self._calculate_stats(times_cached, "Cached Request") if times_cached else None
        }
    
    def benchmark_fallback_performance(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark fallback to lexical similarity."""
        print(f"\nBenchmarking fallback performance ({iterations} iterations)...")
        
        # Force fallback by using invalid server URL
        times_fallback = []
        times_normal = []
        
        for i in range(iterations):
            idx = i % len(self.test_sentences)
            
            # Fallback timing (invalid server)
            start_time = time.time()
            score = compute_score(
                data_source="embedding_remote_test",
                solution_str=self.test_sentences[idx],
                ground_truth=self.reference_sentences[idx],
                extra_info={"server_url": "http://invalid-server:9999"}
            )
            elapsed = time.time() - start_time
            times_fallback.append(elapsed)
            
            # Normal timing (if server available)
            if self.server_url:
                start_time = time.time()
                score = compute_score(
                    data_source="embedding_remote_test",
                    solution_str=self.test_sentences[idx],
                    ground_truth=self.reference_sentences[idx],
                    extra_info={"server_url": self.server_url}
                )
                elapsed = time.time() - start_time
                times_normal.append(elapsed)
        
        results = {
            "fallback_lexical": self._calculate_stats(times_fallback, "Fallback (Lexical)")
        }
        
        if times_normal:
            results["normal_embedding"] = self._calculate_stats(times_normal, "Normal (Embedding)")
            
        return results
    
    def compare_with_local(self, iterations: int = 50) -> Dict[str, Any]:
        """Compare remote embedding with local embedding."""
        print(f"\nComparing with local embedding ({iterations} iterations)...")
        
        times_remote = []
        times_local = []
        scores_remote = []
        scores_local = []
        
        for i in range(iterations):
            idx = i % len(self.test_sentences)
            
            # Remote embedding
            if self.server_url:
                start_time = time.time()
                score_remote = compute_score(
                    data_source="embedding_remote_test",
                    solution_str=self.test_sentences[idx],
                    ground_truth=self.reference_sentences[idx],
                    extra_info={"server_url": self.server_url}
                )
                elapsed = time.time() - start_time
                times_remote.append(elapsed)
                scores_remote.append(score_remote)
            
            # Local embedding (FastText)
            try:
                start_time = time.time()
                score_local = compute_score_local(
                    data_source="embedding_match_test",
                    solution_str=self.test_sentences[idx],
                    ground_truth=self.reference_sentences[idx]
                )
                elapsed = time.time() - start_time
                times_local.append(elapsed)
                scores_local.append(score_local)
            except Exception as e:
                print(f"  Local embedding failed: {e}")
                break
        
        results = {}
        
        if times_remote:
            results["remote"] = self._calculate_stats(times_remote, "Remote (Qwen3)")
            results["remote"]["avg_score"] = statistics.mean(scores_remote)
            
        if times_local:
            results["local"] = self._calculate_stats(times_local, "Local (FastText)")
            results["local"]["avg_score"] = statistics.mean(scores_local)
            
        return results
    
    def _calculate_stats(self, times: List[float], label: str) -> Dict[str, Any]:
        """Calculate statistics for timing data."""
        if not times:
            return {}
        
        sorted_times = sorted(times)
        
        stats = {
            "label": label,
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "p50": sorted_times[len(sorted_times) // 2],
            "p90": sorted_times[int(len(sorted_times) * 0.9)],
            "p95": sorted_times[int(len(sorted_times) * 0.95)],
            "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 100 else sorted_times[-1]
        }
        
        # Convert to milliseconds for readability
        for key in ["mean", "median", "std", "min", "max", "p50", "p90", "p95", "p99"]:
            stats[f"{key}_ms"] = stats[key] * 1000
        
        return stats
    
    def print_results(self, results: Dict[str, Any]):
        """Pretty print benchmark results."""
        for test_name, test_results in results.items():
            print(f"\n{'=' * 60}")
            print(f"Test: {test_name}")
            print('=' * 60)
            
            if isinstance(test_results, dict) and "label" in test_results:
                self._print_single_result(test_results)
            else:
                for sub_name, sub_results in test_results.items():
                    if isinstance(sub_results, dict) and "label" in sub_results:
                        self._print_single_result(sub_results)
    
    def _print_single_result(self, stats: Dict[str, Any]):
        """Print a single benchmark result."""
        print(f"\n{stats['label']}:")
        print(f"  Samples: {stats.get('count', 'N/A')}")
        print(f"  Mean: {stats.get('mean_ms', 0):.2f} ms")
        print(f"  Median: {stats.get('median_ms', 0):.2f} ms")
        print(f"  Std Dev: {stats.get('std_ms', 0):.2f} ms")
        print(f"  Min: {stats.get('min_ms', 0):.2f} ms")
        print(f"  Max: {stats.get('max_ms', 0):.2f} ms")
        print(f"  P90: {stats.get('p90_ms', 0):.2f} ms")
        print(f"  P95: {stats.get('p95_ms', 0):.2f} ms")
        print(f"  P99: {stats.get('p99_ms', 0):.2f} ms")
        
        if "throughput" in stats:
            print(f"  Throughput: {stats['throughput']['mean']:.2f} items/sec (max: {stats['throughput']['max']:.2f})")
        
        if "avg_score" in stats:
            print(f"  Average Score: {stats['avg_score']:.3f}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark remote embedding performance")
    parser.add_argument(
        "--server-url",
        help="TEI server URL (default: use EMBEDDING_SERVER_URL env var)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for single request benchmark (default: 100)"
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batch benchmarks"
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache benchmarks"
    )
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="Skip fallback benchmarks"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite(server_url=args.server_url)
    
    if not suite.server_url:
        print("Warning: No server URL provided. Some benchmarks will be skipped.")
        print("Set EMBEDDING_SERVER_URL or use --server-url flag.")
    
    print(f"Starting benchmark suite...")
    if suite.server_url:
        print(f"Server URL: {suite.server_url}")
    print("=" * 60)
    
    all_results = {}
    
    # Run benchmarks
    try:
        # Single request benchmark
        single_results = suite.benchmark_single_requests(iterations=args.iterations)
        all_results["single_requests"] = single_results
        
        # Batch request benchmark
        if not args.skip_batch:
            batch_results = suite.benchmark_batch_requests()
            all_results["batch_requests"] = batch_results
        
        # Embedding dimension benchmark
        dim_results = suite.benchmark_embedding_dimensions()
        all_results["embedding_dimensions"] = dim_results
        
        # Cache performance benchmark
        if not args.skip_cache:
            cache_results = suite.benchmark_cache_performance()
            all_results["cache_performance"] = cache_results
        
        # Fallback performance benchmark
        if not args.skip_fallback:
            fallback_results = suite.benchmark_fallback_performance()
            all_results["fallback_performance"] = fallback_results
        
        # Comparison with local
        compare_results = suite.compare_with_local()
        all_results["comparison"] = compare_results
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        return 1
    
    # Print results
    suite.print_results(all_results)
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    if "single_requests" in all_results:
        mean_time = all_results["single_requests"]["mean_ms"]
        print(f"Single request average: {mean_time:.2f} ms")
    
    if "batch_requests" in all_results and "batch_50" in all_results["batch_requests"]:
        throughput = all_results["batch_requests"]["batch_50"]["throughput"]["mean"]
        print(f"Batch throughput (50): {throughput:.2f} items/sec")
    
    if "comparison" in all_results:
        if "remote" in all_results["comparison"] and "local" in all_results["comparison"]:
            remote_time = all_results["comparison"]["remote"]["mean_ms"]
            local_time = all_results["comparison"]["local"]["mean_ms"]
            speedup = local_time / remote_time
            print(f"Remote vs Local speedup: {speedup:.2f}x")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
