from typing import List
from llm import llm, get_stable_seed
import re
from collections import Counter
import string
import random


def vectorize_string_normalized(text):
    text = text.lower()
    counter = Counter(c for c in text if c.isalnum())
    chars = string.ascii_lowercase + string.digits
    
    vector = [counter.get(c, 0) for c in chars]
    total = sum(vector)
    
    if total > 0:
        vector = [v / total for v in vector]
    
    return vector

class SimpleRAG:
    def __init__(self):
        """
        Initialize your Vector Store (in-memory or simple DB) here.
        """
        self.vector_store = []

        self.texts = []
        self.chunk_size = 2
        self.overlap = 1 # overlap sentences for better coherence
        self.embedding_size = 64
        self.verbose = True
        pass

    def ingest(self, text_content: str) -> None:
        """
        1. Split text into chunks (Bonus: respect sentence boundaries).
        2. Vectorize chunks (can be mocked).
        3. Store vectors and text.
        """

        # 1. Split text into chunks
        sentences = re.split(r'(?<=[.!?])\s+', text_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(sentences), step):
            chunks.append(' '.join(sentences[i: i + self.chunk_size]))
        

        # 2. Vectorize (mock using letter counts)
        for chunk in chunks:
            seed = get_stable_seed(chunk)
            random.seed(seed)
            vector = [random.randint(0, 10) for _ in range(self.embedding_size)]
            self.vector_store.append(vector)

        self.texts.extend(chunks)


    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the top-k most relevant chunks for the given query.
        (Use cosine similarity or a keyword search if no embeddings available).
        """
        
        # vectorize query (mock)
        seed = get_stable_seed(query)
        random.seed(seed)
        q_vector = [random.randint(0, 10) for _ in range(self.embedding_size)]

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(np.array(q_vector).reshape(1, -1), np.array(self.vector_store))[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
        except:
            similarities = []
            q_norm = sum([x * x for x in q_vector]) ** 0.5
            for k_vector in self.vector_store:
                if len(q_vector) != len(k_vector):
                    print(f"Warning: dimension mismatch: {len(q_vector)} vs {len(k_vector)}")
                    continue
                product = sum([x * y for x, y in zip(q_vector, k_vector)])
                k_norm = sum([x * x for x in k_vector]) ** 0.5
                if q_norm == 0 or k_norm == 0:
                    print(f"Warning: zero vector")
                    similarities.append(0.0)
                else:
                    similarities.append(product / ( q_norm * k_norm))
            top_indices = [i for i, _ in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:k]]
        
        return [self.texts[i] for i in top_indices]
    


    def generate_response(self, query: str) -> str:
        """
        1. Retrieve context based on query.
        2. Simulate an LLM generation using that context.
        """
        # Example logic:
        # context = self.retrieve(query)
        # prompt = f"Context: {context}\nQuestion: {query}"
        # return llm_call(prompt)
        context_list = self.retrieve(query)
        context = "\n".join(
            [f"({i + 1}): {cxt}" for i, cxt in enumerate(context_list)]
        )
        prompt = f"Context:\n{context}\nQuestion: {query}"
        if self.verbose:
            print(f"Prompt:\n{prompt}\n\n\n")
        return llm(prompt)

# --- Test Execution ---
if __name__ == "__main__":
    # Example Data
    policy_text = """
    Employees can work remotely 2 days a week. 
    Expenses are refunded up to 25 CHF for lunch. 
    Security requires all data to stay in Switzerland.
    """

    rag = SimpleRAG()
    
    print("--- Ingesting ---")
    try:
        rag.ingest(policy_text)
        print("Ingestion successful (if implemented)")
    except NotImplementedError:
        print("Ingest method not implemented yet.")
    
    query = "What is the remote work policy?"
    print(f"\n--- Querying: {query} ---")
    try:
        response = rag.generate_response(query)
        print(f"Result: {response}")
    except NotImplementedError:
        print("Generate Response method not implemented yet.")
