"""
Start the server on node0

Node0 should handle requests from clients and distribute tasks to node1 and node2

Microservices on node0:
- Embedding service: Generate embeddings for queries
- Sentiment service
- Toxicity detection service
"""

import os
import json
import time
from flask import Flask, request, jsonify
from queue import Queue, Empty
import multiprocessing
import requests
from dataclasses import asdict

# import shared components from service.py
from lru_cache import LRUCache
from service import (
    TOTAL_NODES,
    NODE_NUMBER,
    NODE_0_IP,
    NODE_1_IP,
    NODE_2_IP,
    STEP_TO_NODEIP,
)
from service import WORKERS, CONFIG, BATCH_SIZE, BATCH_WAIT_SECONDS
from service import FAISS_INDEX_PATH, DOCUMENTS_DIR
from service import PipelineRequest, PipelineResponse, PipelineData
from service import data_to_response
from service import Embedding, SentimentAnalysis, ToxicityDetection
CACHE_CAPACITY = int(os.environ.get("CACHE_CAPACITY", 1000))

# Flask app
app = Flask(__name__)

# Initialize LRU Cache
cache = LRUCache(capacity=0, db_path="lru_cache.db")

# Request queue and results storage
request_queue = None
sentiment_queue = None
toxicity_queue = None

# manager = multiprocessing.Manager()
# results = manager.dict()  # request_id -> {"event": manager.Event(), "response": PipelineData, "count": int (how many requests are waiting for this result)}
# results_lock = manager.Lock()
results_lock = None
manager = None
results = None

# multiprocessing on node0
# number of worker processes to run each service
EMBEDDING_PROCESSES = 1  # TRY DIFFERENT NUMBER OF PROCESSES!
SENTIMENT_PROCESSES = 1  # TRY DIFFERENT NUMBER OF PROCESSES!
TOXICITY_PROCESSES = 1  # TRY DIFFERENT NUMBER OF PROCESSES!


def embedding_worker(request_queue):
    """
    embedding worker function

    Continuously:
      - pull up to BATCH_SIZE requests from request_queue
      - do embedding
      - send to the node with step 2 service
    """
    pipeline = Embedding()

    while True:
        # Block until at least 1 request is available
        req = request_queue.get()
        if req is None:  # shutdown signal if you want one
            break

        batch = [req]

        # Try to grab up to BATCH_SIZE-1 more without blocking too long
        batch_deadline = time.time() + BATCH_WAIT_SECONDS
        while len(batch) < BATCH_SIZE:
            remaining = batch_deadline - time.time()
            if remaining <= 0:
                break
            try:
                # Wait a tiny bit for more requests to form a fuller batch
                more_req = request_queue.get(timeout=remaining)
                if more_req is None:
                    # push back the sentinel for other threads and stop
                    request_queue.put(None)
                    break
                batch.append(more_req)
            except Empty:
                break

        reqs_data = [
            {"request_id": r["request_id"], "query": r["query"]} for r in batch
        ]

        # Create PipelineData objects
        reqs = [
            PipelineData(
                request_id=r["request_id"], query=r["query"], timestamp=time.time()
            )
            for r in reqs_data
        ]

        # Process request
        responses = pipeline.process_batch(reqs)
        responses = [asdict(r) for r in responses]

        # Prepare HTTP payload for step 2
        payload = {"requests": responses}

        # send payload to step 2 node
        try:
            resp = requests.post(
                f"http://{STEP_TO_NODEIP[2]}/query",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            print(f"sent batch of size {len(batch)} to step 2")

        except Exception as e:
            for _ in batch:
                print(f"Error processing request: {e}")

def sentiment_worker(sentiment_queue, results, results_lock):
    """
    sentiment worker function

    Continuously:
      - pull up to BATCH_SIZE requests from sentiment_queue
      - do sentiment analysis
    """
    # global results

    pipeline = SentimentAnalysis()

    while True:
        # Block until at least 1 request is available
        req = sentiment_queue.get()
        if req is None:  # shutdown signal if you want one
            break

        batch = [req]

        # Try to grab up to BATCH_SIZE-1 more without blocking too long
        batch_deadline = time.time() + BATCH_WAIT_SECONDS
        while len(batch) < BATCH_SIZE:
            remaining = batch_deadline - time.time()
            if remaining <= 0:
                break
            try:
                # Wait a tiny bit for more requests to form a fuller batch
                more_req = sentiment_queue.get(timeout=remaining)
                if more_req is None:
                    # push back the sentinel for other threads and stop
                    sentiment_queue.put(None)
                    break
                batch.append(more_req)
            except Empty:
                break

        # Process request
        responses = pipeline.process_batch(batch)

        # Update results storage
        with results_lock:
            for r in responses:
                if r.request_id in results:
                    entry = results[r.request_id]
                    if entry["response"] is None:
                        entry["response"] = r
                        results[r.request_id] = entry
                    else:
                        entry["response"].sentiment = r.sentiment
                        results[r.request_id] = entry
                        entry["event"].set()
                    

def toxicity_worker(toxicity_queue, results, results_lock):
    """
    toxicity worker function

    Continuously:
      - pull up to BATCH_SIZE requests from toxicity_queue
      - do toxicity analysis
    """
    pipeline = ToxicityDetection()

    while True:
        # Block until at least 1 request is available
        req = toxicity_queue.get()
        if req is None:  # shutdown signal if you want one
            break

        batch = [req]

        # Try to grab up to BATCH_SIZE-1 more without blocking too long
        batch_deadline = time.time() + BATCH_WAIT_SECONDS
        while len(batch) < BATCH_SIZE:
            remaining = batch_deadline - time.time()
            if remaining <= 0:
                break
            try:
                # Wait a tiny bit for more requests to form a fuller batch
                more_req = toxicity_queue.get(timeout=remaining)
                if more_req is None:
                    # push back the sentinel for other threads and stop
                    toxicity_queue.put(None)
                    break
                batch.append(more_req)
            except Empty:
                break

        # Process request
        responses = pipeline.process_batch(batch)

        # Update results storage
        with results_lock:
            for r in responses:
                if r.request_id in results:
                    entry = results[r.request_id]
                    if entry["response"] is None:
                        entry["response"] = r
                        results[r.request_id] = entry
                    else:
                        entry["response"].is_toxic = r.is_toxic
                        results[r.request_id] = entry
                        entry["event"].set()
                    

@app.route("/query", methods=["POST"])
def handle_query():
    """Handle incoming query requests"""
    try:
        data = request.json
        request_id = data.get("request_id")
        query = data.get("query")

        if not request_id or not query:
            return jsonify({"error": "Missing request_id or query"}), 400

        # Check cache
        cached_val = cache.get(query)
        if cached_val:
            print(f"Cache hit for query: {query}")
            response = PipelineResponse(
                request_id=request_id,
                generated_response=cached_val["generated_response"],
                sentiment=cached_val["sentiment"],
                is_toxic=cached_val["is_toxic"],
                processing_time=0.0,
            )
            return jsonify(response), 200

        # Check if result already exists and is finished
        with results_lock:
            # new request
            if request_id not in results:
                entry = {"event": manager.Event(), "response": None, "count": 1}
                results[request_id] = entry

                # Add to queue
                print(f"queueing request {request_id}")
                request_queue.put({"request_id": request_id, "query": query})

            # result already exists and is finished
            elif request_id in results and results[request_id]["event"].is_set():
                pipeline_response = data_to_response(results[request_id]["response"])

                return jsonify(pipeline_response), 200

            # result is being processed
            elif request_id in results and not results[request_id]["event"].is_set():
                entry = results[request_id]
                entry["count"] += 1
                results[request_id] = entry

        # wait for the result
        timeout = 300  # 5 minutes

        # time limit exceeded
        if not entry["event"].wait(timeout):
            with results_lock:
                results.pop(request_id, None)
            return jsonify({"error": "Request timed out"}), 504

        # grab a fresh copy of the entry
        with results_lock:
            entry = results.get(request_id)
            if not entry or entry["response"] is None:
                # Something went wrong in the worker; fail gracefully
                results.pop(request_id, None)
                return jsonify({"error": "Internal error: response missing"}), 500

            # Update cache
            cache.set(
                entry["response"].query,
                {
                    "generated_response": entry["response"].generated_response,
                    "sentiment": entry["response"].sentiment,
                    "is_toxic": entry["response"].is_toxic,
                },
            )

            # success, return the result
            entry["count"] -= 1
            if entry["count"] == 0:
                results.pop(request_id)
            else:
                results[request_id] = entry
            return jsonify(data_to_response(entry["response"])), 200

    except Exception as e:
        print(f"Error in handle_query: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/complete", methods=["POST"])
def complete():
    """Handle post request from step5 service"""
    data = request.json
    pipelinedata_requests = data.get('requests')                                # a list of dict from previous step
    pipelinedata_requests = [
        PipelineData(**r) 
        for r in pipelinedata_requests
    ]  # convert dict to PipelineData

    # put to sentiment and toxicity queues
    for pipelinedata in pipelinedata_requests:
        sentiment_queue.put(pipelinedata)
        toxicity_queue.put(pipelinedata)

    return jsonify({}), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return (
        jsonify({"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}),
        200,
    )


def main():
    """
    Start the server on node0
    """
    global manager
    global results
    global results_lock
    assert NODE_NUMBER == 0, "This script should be run on node0 only."

    manager = multiprocessing.Manager()
    results = manager.dict()  # request_id -> {"event": manager.Event(), "response": PipelineData, "count": int (how many requests are waiting for this result)}
    results_lock = manager.Lock()

    print("=" * 60)
    print("NODE0 SERVER STARTING")
    print("=" * 60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")

    # Request queue 
    global request_queue, sentiment_queue, toxicity_queue
    request_queue = multiprocessing.Queue()
    sentiment_queue = multiprocessing.Queue()
    toxicity_queue = multiprocessing.Queue()

    # Start worker processes
    for i in range(EMBEDDING_PROCESSES):
        t = multiprocessing.Process(target=embedding_worker, args=(request_queue,), daemon=True)
        t.start()
    print(f"Started {EMBEDDING_PROCESSES} worker processes")

    for i in range(SENTIMENT_PROCESSES):
        t = multiprocessing.Process(target=sentiment_worker, args=(sentiment_queue, results, results_lock), daemon=True)
        t.start()
    print(f"Started {SENTIMENT_PROCESSES} sentiment worker processes")

    for i in range(TOXICITY_PROCESSES):
        t = multiprocessing.Process(target=toxicity_worker, args=(toxicity_queue, results, results_lock), daemon=True)
        t.start()
    print(f"Started {TOXICITY_PROCESSES} toxicity worker processes")

    hostport = NODE_0_IP

    hostname = hostport.split(":")[0]
    port = int(hostport.split(":")[1]) if ":" in hostport else 8000

    print(f"\nStarting Flask server on {hostname}:{port}")
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()
