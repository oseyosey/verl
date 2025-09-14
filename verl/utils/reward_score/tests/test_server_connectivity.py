#!/usr/bin/env python3
"""
Simple script to test connectivity with Text Embeddings Inference (TEI) server.

Usage:
    python test_server_connectivity.py [SERVER_URL]
    
    If SERVER_URL is not provided, it will use EMBEDDING_SERVER_URL environment variable.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from embedding_client import EmbeddingClient
    import numpy as np
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you have installed required dependencies: pip install requests numpy")
    sys.exit(1)


def test_server_health(client: EmbeddingClient) -> bool:
    """Test server health endpoint."""
    print("\n1. Testing server health check...")
    try:
        is_healthy = client.health_check()
        if is_healthy:
            print("✓ Server is healthy")
            return True
        else:
            print("✗ Server health check failed")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_server_info(client: EmbeddingClient) -> bool:
    """Test server info endpoint."""
    print("\n2. Getting server information...")
    try:
        info = client.get_server_info()
        if info:
            print("✓ Server info retrieved successfully:")
            print(json.dumps(info, indent=2))
            return True
        else:
            print("✗ Failed to get server info")
            return False
    except Exception as e:
        print(f"✗ Server info error: {e}")
        return False


def test_single_embedding(client: EmbeddingClient) -> bool:
    """Test single text embedding."""
    print("\n3. Testing single text embedding...")
    test_text = "Hello, this is a test sentence for embedding."
    
    try:
        start_time = time.time()
        embeddings = client.embed_texts([test_text])
        elapsed = time.time() - start_time
        
        if embeddings is not None and len(embeddings) > 0:
            print(f"✓ Single embedding successful")
            print(f"  - Shape: {embeddings.shape}")
            print(f"  - Dimension: {embeddings.shape[1]}")
            print(f"  - Time: {elapsed:.3f}s")
            print(f"  - First 5 values: {embeddings[0][:5]}")
            return True
        else:
            print("✗ Failed to get embedding")
            return False
    except Exception as e:
        print(f"✗ Single embedding error: {e}")
        return False


def test_batch_embedding(client: EmbeddingClient) -> bool:
    """Test batch text embedding."""
    print("\n4. Testing batch text embedding...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Text embeddings capture semantic meaning of sentences.",
        "This is the fifth test sentence in our batch."
    ]
    
    try:
        start_time = time.time()
        embeddings = client.embed_texts(test_texts)
        elapsed = time.time() - start_time
        
        if embeddings is not None and len(embeddings) == len(test_texts):
            print(f"✓ Batch embedding successful")
            print(f"  - Shape: {embeddings.shape}")
            print(f"  - Batch size: {len(test_texts)}")
            print(f"  - Time: {elapsed:.3f}s ({elapsed/len(test_texts):.3f}s per text)")
            
            # Test similarity between similar sentences
            # Sentences 1 and 3 are about ML/AI, should have high similarity
            from embedding_remote import _cosine
            sim_score = _cosine(embeddings[1], embeddings[3])
            print(f"  - Similarity between ML sentences: {sim_score:.3f}")
            
            return True
        else:
            print("✗ Failed to get batch embeddings")
            return False
    except Exception as e:
        print(f"✗ Batch embedding error: {e}")
        return False


def test_large_batch(client: EmbeddingClient, batch_size: int = 50) -> bool:
    """Test larger batch to check performance."""
    print(f"\n5. Testing large batch ({batch_size} texts)...")
    
    # Generate test texts
    test_texts = []
    for i in range(batch_size):
        test_texts.append(f"This is test sentence number {i}. It contains some variation to make it unique.")
    
    try:
        start_time = time.time()
        embeddings = client.embed_texts(test_texts, batch_size=25)  # Process in smaller batches
        elapsed = time.time() - start_time
        
        if embeddings is not None and len(embeddings) == batch_size:
            print(f"✓ Large batch embedding successful")
            print(f"  - Total time: {elapsed:.3f}s")
            print(f"  - Average time per text: {elapsed/batch_size:.3f}s")
            print(f"  - Throughput: {batch_size/elapsed:.1f} texts/second")
            return True
        else:
            print("✗ Failed to process large batch")
            return False
    except Exception as e:
        print(f"✗ Large batch error: {e}")
        return False


def test_edge_cases(client: EmbeddingClient) -> bool:
    """Test edge cases."""
    print("\n6. Testing edge cases...")
    
    # Test empty text
    try:
        embeddings = client.embed_texts([""])
        if embeddings is not None:
            print("✓ Empty text handled")
        else:
            print("✗ Empty text failed")
    except Exception as e:
        print(f"✗ Empty text error: {e}")
    
    # Test very long text
    long_text = "This is a very long sentence. " * 200  # ~2000 words
    try:
        start_time = time.time()
        embeddings = client.embed_texts([long_text], truncate=True)
        elapsed = time.time() - start_time
        if embeddings is not None:
            print(f"✓ Long text handled (truncated, {elapsed:.3f}s)")
        else:
            print("✗ Long text failed")
    except Exception as e:
        print(f"✗ Long text error: {e}")
    
    # Test special characters
    special_text = "Test with émojis 🚀 and spëcial cháracters! #ML @AI"
    try:
        embeddings = client.embed_texts([special_text])
        if embeddings is not None:
            print("✓ Special characters handled")
        else:
            print("✗ Special characters failed")
    except Exception as e:
        print(f"✗ Special characters error: {e}")
    
    return True


def test_reward_function(server_url: str) -> bool:
    """Test the full reward function."""
    print("\n7. Testing reward function integration...")
    
    from embedding_remote import compute_score
    
    try:
        # Test similar sentences
        score1 = compute_score(
            data_source="embedding_remote_test",
            solution_str="Machine learning is a powerful technology",
            ground_truth="AI and machine learning are transformative technologies",
            extra_info={"server_url": server_url}
        )
        print(f"✓ Similar sentences score: {score1:.3f}")
        
        # Test dissimilar sentences
        score2 = compute_score(
            data_source="embedding_remote_test",
            solution_str="The weather is sunny today",
            ground_truth="Machine learning is complex",
            extra_info={"server_url": server_url}
        )
        print(f"✓ Dissimilar sentences score: {score2:.3f}")
        
        # Test with length penalty
        score3 = compute_score(
            data_source="embedding_remote_test",
            solution_str="This is a very very very long answer with lots of unnecessary words and repetition",
            ground_truth="Short answer",
            extra_info={
                "server_url": server_url,
                "length_penalty": "ratio",
                "length_threshold": 1.5
            }
        )
        print(f"✓ Length penalty score: {score3:.3f}")
        
        if score1 > score2 and score1 > 0.5:
            print("✓ Reward function working correctly")
            return True
        else:
            print("✗ Unexpected scores")
            return False
            
    except Exception as e:
        print(f"✗ Reward function error: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test TEI server connectivity")
    parser.add_argument(
        "server_url",
        nargs="?",
        help="TEI server URL (default: use EMBEDDING_SERVER_URL env var)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Get server URL
    server_url = args.server_url or os.environ.get("EMBEDDING_SERVER_URL")
    if not server_url:
        print("Error: No server URL provided.")
        print("Either pass SERVER_URL as argument or set EMBEDDING_SERVER_URL environment variable.")
        sys.exit(1)
    
    print(f"Testing TEI server at: {server_url}")
    print("=" * 60)
    
    # Create client
    try:
        client = EmbeddingClient(
            server_url=server_url,
            api_key=args.api_key,
            timeout=args.timeout
        )
    except Exception as e:
        print(f"Failed to create client: {e}")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    tests_total = 7
    
    if test_server_health(client):
        tests_passed += 1
    
    if test_server_info(client):
        tests_passed += 1
    
    if test_single_embedding(client):
        tests_passed += 1
    
    if test_batch_embedding(client):
        tests_passed += 1
    
    if test_large_batch(client):
        tests_passed += 1
    
    if test_edge_cases(client):
        tests_passed += 1
    
    if test_reward_function(server_url):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓ All tests passed! Server is ready for use.")
        return 0
    else:
        print("✗ Some tests failed. Please check server configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
