#!/usr/bin/env python3
"""
TEI Server using Python (Direct PyTorch approach)
This is a lightweight alternative to the containerized TEI server.

Usage:
    python run_tei_python.py --model-id Qwen/Qwen3-Embedding-0.6B --port 8080

Features:
- Direct PyTorch implementation
- GPU acceleration if available
- Compatible with TEI API endpoints (/health, /info, /embed)
- No container dependencies
"""

import os
import asyncio
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI(title="Text Embeddings Inference Server")

class EmbedRequest(BaseModel):
    inputs: List[str]
    normalize: bool = True
    truncate: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class HealthResponse(BaseModel):
    status: str

# Global model variables
model = None
tokenizer = None
device = None

def load_model(model_id: str = "Qwen/Qwen3-Embedding-0.6B"):
    """Load the embedding model."""
    global model, tokenizer, device
    
    print(f"Loading model: {model_id}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully!")

def get_embeddings(texts: List[str], normalize: bool = True) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Convert to CPU and return as list
    return embeddings.cpu().numpy().tolist()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")

@app.get("/info")
async def get_info():
    """Get server information."""
    return {
        "model_id": os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B"),
        "device": device,
        "max_batch_size": 256,
        "embedding_dimension": 1024
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Embed texts."""
    try:
        embeddings = get_embeddings(request.inputs, request.normalize)
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    # Set environment variable
    os.environ["MODEL_ID"] = args.model_id
    
    # Load model
    load_model(args.model_id)
    
    # Start server
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
