import pickle
import random
import time
import gc
import matplotlib.pyplot as plt
from queue import Empty
import multiprocessing

from service import PipelineData
from service import Embedding, FAISS_ANN, DocumentRetrieval, Reranking, ResponseGeneration, SentimentAnalysis, ToxicityDetection 



TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?"
]

def generate_pkl_input(INPUT_LEN):
    """ Generate input pickle files for each stage of the pipeline """
    queries = []
    for _ in range(INPUT_LEN):
        queries.append(random.choice(TEST_QUERIES))
    
    embedding_input = [PipelineData(query=q, request_id=str(i), timestamp=time.time()) for i, q in enumerate(queries)]
    pickle.dump(embedding_input, open("embedding_input.pkl", "wb"))

    pipeline1 = Embedding()
    FAISS_input = pipeline1.process_batch(embedding_input)
    pickle.dump(FAISS_input, open("FAISS_input.pkl", "wb"))
    del pipeline1
    gc.collect()

    pipeline2 = FAISS_ANN()
    retrieval_input = pipeline2.process_batch(FAISS_input)
    pickle.dump(retrieval_input, open("retrieval_input.pkl", "wb"))
    del pipeline2
    gc.collect()

    pipeline3 = DocumentRetrieval()
    reranking_input = pipeline3.process_batch(retrieval_input)
    pickle.dump(reranking_input, open("reranking_input.pkl", "wb"))
    del pipeline3
    gc.collect()

    pipeline4 = Reranking()
    llm_input = pipeline4.process_batch(reranking_input)
    pickle.dump(llm_input, open("llm_input.pkl", "wb"))
    del pipeline4
    gc.collect()

    pipeline5 = ResponseGeneration()
    llm_output = pipeline5.process_batch(llm_input)
    pickle.dump(llm_output, open("llm_output.pkl", "wb"))

def batch_experiment_high_load(SERVICE, MAX_BATCH_SIZE, MIN_BATCH_SIZE=1):
    BATCH_SIZE_RANGE = list(range(MIN_BATCH_SIZE, MAX_BATCH_SIZE + 1))    # range of batch sizes to test

    service_to_input = {
        "embedding": "embedding_input.pkl",
        "faiss": "FAISS_input.pkl",
        "retrieval": "retrieval_input.pkl",
        "reranking": "reranking_input.pkl",
        "llm": "llm_input.pkl",
        "sentiment": "llm_output.pkl",
        "toxicity": "llm_output.pkl",
    }
    input_data = pickle.load(open(service_to_input[SERVICE], "rb"))
    if SERVICE == "llm":
        input_data = input_data[:32]   # limit requests for llm due to cost

    pipeline = None
    if SERVICE == "embedding":
        pipeline = Embedding()
    elif SERVICE == "faiss":
        pipeline = FAISS_ANN()
    elif SERVICE == "retrieval":
        pipeline = DocumentRetrieval()
    elif SERVICE == "reranking":
        pipeline = Reranking()
    elif SERVICE == "llm":
        pipeline = ResponseGeneration()
    elif SERVICE == "sentiment":
        pipeline = SentimentAnalysis()
    elif SERVICE == "toxicity":
        pipeline = ToxicityDetection()
    else:
        print("Unknown service")
        return
    
    # test different batch sizes
    throughputs = []          # requests per minute
    latency_95 = []           # 95th percentile latency
    for batch_size in BATCH_SIZE_RANGE:
        start_time = time.time()          # start time of the experiment
        req_start_time = [time.time() for _ in range(len(input_data)) ]
        req_end_time = []

        current_index = 0
        while current_index < len(input_data):
            # try get batch_size items from input_data
            end_index = min(current_index + batch_size, len(input_data))
            req = input_data[current_index:end_index]
            current_index = end_index
            
            # run pipeline
            # batch_start_time = [time.time() for _ in range(len(req))]
            pipeline.process_batch(req)
            batch_end_time = [time.time() for _ in range(len(req))]

            # record end time of batch
            # req_start_time.extend(batch_start_time)
            req_end_time.extend(batch_end_time)
        
        # throughput
        end_time = time.time()            # end time of the experiment
        total_time = end_time - start_time
        throughput = len(input_data) / total_time * 60  # requests per minute
        throughputs.append(throughput)

        # latency 95th percentile
        latencies = [req_end_time[i] - req_start_time[i] for i in range(len(input_data))]
        latencies.sort()
        p95_latency = latencies[int(0.95 * len(latencies) - 1)]
        latency_95.append(p95_latency)

        pickle.dump(throughputs, open(f"batch_experiment_figure/throughputs_{SERVICE}_2.pkl", "wb"))
        pickle.dump(latency_95, open(f"batch_experiment_figure/latency_95_{SERVICE}_2.pkl", "wb"))
    
    # plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(BATCH_SIZE_RANGE, throughputs, marker='o')
    plt.title(f'Throughput vs Batch Size for {SERVICE}')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (requests/min)')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(BATCH_SIZE_RANGE, latency_95, marker='o', color='orange')
    plt.title(f'95th Percentile Latency vs Batch Size for {SERVICE}')
    plt.xlabel('Batch Size')
    plt.ylabel('95th Percentile Latency (seconds)')
    plt.grid()
    plt.tight_layout()
    
    # save figure
    plt.savefig(f'batch_experiment_figure/{SERVICE}.png')


def main():
    # generate_pkl_input(INPUT_LEN=100)
    batch_experiment_high_load(SERVICE="llm", MAX_BATCH_SIZE=32, MIN_BATCH_SIZE=13)



if __name__ == "__main__":
    main()