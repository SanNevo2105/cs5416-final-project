import pickle
import random
import time

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

def generate_pkl_input():
    queries = []
    for _ in range(100):
        queries.append(random.choice(TEST_QUERIES))
    
    embedding_input = [PipelineData(query=q, request_id=str(i), timestamp=time.time()) for i, q in enumerate(queries)]
    pickle.dump(embedding_input, open("embedding_input.pkl", "wb"))

    pipeline1 = Embedding()
    FAISS_input = pipeline1.process_batch(embedding_input)
    pickle.dump(FAISS_input, open("FAISS_input.pkl", "wb"))

    pipeline2 = FAISS_ANN()
    retrieval_input = pipeline2.process_batch(FAISS_input)
    pickle.dump(retrieval_input, open("retrieval_input.pkl", "wb"))

    pipeline3 = DocumentRetrieval()
    reranking_input = pipeline3.process_batch(retrieval_input)
    pickle.dump(reranking_input, open("reranking_input.pkl", "wb"))

    pipeline4 = Reranking()
    llm_input = pipeline4.process_batch(reranking_input)
    pickle.dump(llm_input, open("llm_input.pkl", "wb"))

    pipeline5 = ResponseGeneration()
    llm_output = pipeline5.process_batch(llm_input)
    pickle.dump(llm_output, open("llm_output.pkl", "wb"))


def main():
    SERVICE = "embedding"  # change this to test different services
    BATCH_SIZE = 4
    BATCH_WAIT_SECONDS = 1

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
    
    request_queue = multiprocessing.Queue()
    for item in input_data:
        request_queue.put(item)
    
    



if __name__ == "__main__":
    main()