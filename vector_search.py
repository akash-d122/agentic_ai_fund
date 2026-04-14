from google import genai
from dotenv import load_dotenv
import os
import math

# Load API key
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client()

def cosine_similarity(vec1, vec2):
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

def get_top_k_results(query, sentences, embeddings, k=2, threshold=0.65):
    query_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )
    
    query_vector = query_result.embeddings[0].values

    scores = []

    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_vector, emb.values)
        scores.append((sentences[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nAll scores:")
    for text, score in scores:
        print(f"{score:.4f} | {text}")

    filtered = [item for item in scores if item[1] >= threshold]

    return filtered[:k]



sentences = [
    "The quick brown fox jumps over the lazy dog.", 
    "The lazy dog is sleeping.",
    "The fox is quick and clever.",
    "The dog is lazy but loyal.",
    "The fox and the dog are friends."
]

queries = [
    "fast fox",
    "sleeping dog",
    "loyal animal",
    "apple"
]

results = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences
)

embeddings = results.embeddings

for query in queries:
    result = get_top_k_results(query, sentences, embeddings)
    print(f"\nQuery: {query}\n")
    if not results:
        print("No relevant results found.")
    else:
        for text, score in result:
            print(f"{score:.4f} | {text}")
