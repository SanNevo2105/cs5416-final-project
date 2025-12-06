"""
Microservices class for all 7 steps, some constants, and some utility data classes
"""

import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue, Empty
import threading
import requests
import platform
from abc import ABC, abstractmethod

# Detect macOS
if platform.system() == "Darwin":
    print("Detected macOS → setting FAISS to single-thread mode for stability")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')
WORKERS = [NODE_0_IP, NODE_1_IP, NODE_2_IP]
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4)) 
BATCH_WAIT_SECONDS = int(os.environ.get("BATCH_WAIT_SECONDS", 10))

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768, #You must use this dimension
    'max_tokens': 128, #You must use this max token limit
    'retrieval_k': 10, #You must retrieve this many documents from the FAISS index
    'truncate_length': 512 # You must use this truncate length
}

# Map each step to the responsible node
# !!!!!!!!!CHANGE THIS IF YOU CHANGE WHERE THE SERVICES ARE HOSTED!!!!!!!!
# e.g. step 1 is on node 0, and so on
STEP_TO_NODEIP = {
    1: NODE_0_IP,  # Embedding
    2: NODE_1_IP,  # FAISS ANN
    3: NODE_1_IP,  # Document Retrieval
    4: NODE_1_IP,  # Reranking
    5: NODE_2_IP,  # Response Generation    
    6: NODE_2_IP,  # Sentiment Analysis
    7: NODE_2_IP   # Toxicity Detection
}

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float

@dataclass
class PipelineData():
    """
    Everything in PipelineRequest and PipelineResponse plus a data field

    Data structure sent between nodes for each intermediate steps
    """
    request_id: str
    query: str
    timestamp: float
    generated_response: Optional[str] = ""
    sentiment: Optional[str] = ""
    is_toxic: Optional[str] = ""
    processing_time: float = 0.0
    data: Optional[Any] = None        # intermediate outputs from each steps

class Microservices(ABC):
    """
    Abstract base class for all microservices
    """
    def __init__(self):
        device = torch.device("cpu")
        self.device = device       

    @abstractmethod
    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        pass


class Embedding(Microservices):
    """
    Step 1: Embedding microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing embedding microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")

        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.model = SentenceTransformer(self.embedding_model_name).to(self.device)

    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 1: Generate embeddings for a batch of queries"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings   

    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of inputs to embeddings
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]

        print("\n" + "="*60)
        print(f"Generating embeddings for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        query_embeddings = self._generate_embeddings_batch(queries)     

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} embedding processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                processing_time=request.processing_time + processing_time,
                data=query_embeddings[idx].tolist()
            ))
        
        return responses      

class FAISS_ANN(Microservices):
    """
    Step 2: FAISS ANN microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing FAISS ANN microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")

        print("Loading FAISS index into memory once for this process...")
        self.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])

    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 2: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        # print("Loading FAISS index")
        # index = faiss.read_index(CONFIG['faiss_index_path'])
        query_embeddings = query_embeddings.astype('float32')
        # _, indices = index.search(query_embeddings, CONFIG['retrieval_k'])
        # del index
        # gc.collect()
        _, indices = self.faiss_index.search(query_embeddings, CONFIG["retrieval_k"])
        return [row.tolist() for row in indices]

    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of embeddings to perform FAISS ANN search
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        query_embeddings = np.array([req.data for req in requests]).astype('float32')

        print("\n" + "="*60)
        print(f"Performing FAISS ANN search for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} FAISS ANN search processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                processing_time=request.processing_time + processing_time,
                data=doc_id_batches[idx]
            ))
        
        return responses

class DocumentRetrieval(Microservices):
    """
    Step 3: Document Retrieval microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing Document Retrieval microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"Documents path: {CONFIG['documents_path']}")

    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 3: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        conn.close()
        return documents_batch
    
    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of doc_id lists to fetch documents
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        doc_id_batches = [req.data for req in requests]

        print("\n" + "="*60)
        print(f"Fetching documents for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} document retrieval processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                processing_time=request.processing_time + processing_time,
                data=documents_batch[idx]
            ))
        
        return responses


class Reranking(Microservices):
    """
    Step 4: Reranking microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing Reranking microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")

        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
        self.model.eval()
    
    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Step 4: Rerank retrieved documents for each query in the batch"""
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches

    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of document lists to rerank them
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]
        documents_batch = [req.data for req in requests]

        print("\n" + "="*60)
        print(f"Reranking documents for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(
            queries,
            documents_batch
        )

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} reranking processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                processing_time=request.processing_time + processing_time,
                data=reranked_docs_batch[idx]
            ))
        
        return responses    

class ResponseGeneration(Microservices):
    """
    Step 5: Response Generation microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing Response Generation microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")

        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Step 5: Generate LLM responses for each query in the batch"""

        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        return responses
    
    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of reranked document lists to generate responses
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]
        documents_batch = [req.data for req in requests]

        print("\n" + "="*60)
        print(f"Generating responses for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(
            queries,
            documents_batch
        )

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} response generation processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                generated_response=responses_text[idx],
                processing_time=request.processing_time + processing_time
            ))
        
        return responses

class SentimentAnalysis(Microservices):
    """
    Step 6: Sentiment Analysis microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing Sentiment Analysis microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")

        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )

    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        return sentiments        
    
    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of generated responses to analyze sentiment
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        responses_text = [req.generated_response for req in requests]

        print("\n" + "="*60)
        print(f"Analyzing sentiment for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} sentiment analysis processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                generated_response=request.generated_response,
                sentiment=sentiments[idx],
                processing_time=request.processing_time + processing_time
            ))
        
        return responses

class ToxicityDetection(Microservices):
    """
    Step 7: Toxicity Detection microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing Toxicity Detection microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")

        self.safety_model_name = 'unitary/toxic-bert'
        self.classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )

    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""

        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        return toxicity_flags
    
    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of generated responses to detect toxicity
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        responses_text = [req.generated_response for req in requests]

        print("\n" + "="*60)
        print(f"Detecting toxicity for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} toxicity detection processed in {processing_time:.2f} seconds")
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                generated_response=request.generated_response,
                sentiment=request.sentiment,
                is_toxic=sensitivity_result,
                processing_time=request.processing_time + processing_time
            ))
        return responses

class FAISS_Retrieval_Reranking(Microservices):
    """
    Combined Step 2-4: FAISS ANN + Document Retrieval + Reranking microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing FAISS + Retrieval + Reranking microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")

        print("Loading FAISS index into memory once for this process...")
        self.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
        self.model.eval()

    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 2: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        # print("Loading FAISS index")
        # index = faiss.read_index(CONFIG['faiss_index_path'])
        query_embeddings = query_embeddings.astype('float32')
        # _, indices = index.search(query_embeddings, CONFIG['retrieval_k'])
        # del index
        # gc.collect()
        _, indices = self.faiss_index.search(query_embeddings, CONFIG["retrieval_k"])
        return [row.tolist() for row in indices]

    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 3: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        conn.close()
        return documents_batch

    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Step 4: Rerank retrieved documents for each query in the batch"""

        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches

    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of document lists to rerank them
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]
        query_embeddings = np.array([req.data for req in requests]).astype('float32')

        print("\n" + "="*60)
        print(f"Reranking documents for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(
            queries,
            documents_batch
        )

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} FAISS, retrieval, reranking processed in {processing_time:.2f} seconds")
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                processing_time=request.processing_time + processing_time,
                data=reranked_docs_batch[idx]
            ))
        
        return responses    

class LLM_sentiment_toxicity(Microservices):
    """
    Combined Step 5-7: Response Generation + Sentiment Analysis + Toxicity Detection microservice
    """
    def __init__(self):
        super().__init__()

        print(f"Initializing LLM + Sentiment + Toxicity microservice on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")

        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        self.sentiment_classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )

        self.safety_classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Step 5: Generate LLM responses for each query in the batch"""

        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        return responses

    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 6: Analyze sentiment for each generated response"""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.sentiment_classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        return sentiments         

    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 7: Filter responses for safety for each entry in the batch"""

        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.safety_classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        return toxicity_flags

    def process_batch(self, requests: List[PipelineData]) -> List[PipelineData]:
        """
        Process a batch of reranked document lists to generate responses, analyze sentiment, and detect toxicity
        """
        if not requests:
            return []
        
        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]
        reranked_docs_batch = [req.data for req in requests]

        print("\n" + "="*60)
        print(f"Generating responses, analyzing sentiment, and detecting toxicity for a batch of {batch_size} requests")
        print("="*60)     

        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(
            queries,
            reranked_docs_batch
        )

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} LLM, sentiment, toxicity processed in {processing_time:.2f} seconds")
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(PipelineData(
                request_id=request.request_id,
                query=request.query,
                timestamp=request.timestamp,
                generated_response=responses_text[idx],
                sentiment=sentiments[idx],
                is_toxic=sensitivity_result,
                processing_time=request.processing_time + processing_time
            ))
        
        return responses