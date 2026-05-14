from google import genai
from dotenv import load_dotenv
import os
import math

# ------------------ SETUP ------------------

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client()

# ------------------ COSINE SIMILARITY ------------------

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


# ------------------ RETRIEVAL ------------------

def get_top_k_results(query, sentences, embeddings, k=3, threshold=0.60):
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


# ------------------ RERANKING ------------------

def rerank(query, results):
    boosted = []

    for text, score in results:
        bonus = 0

        # simple rule-based reranking
        if "color" in query.lower() and "brown" in text.lower():
            bonus += 0.1

        boosted.append((text, score + bonus))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


# ------------------ GENERATION ------------------

def generate_response(query, context_chunks):
    context = "\n\n".join([f"Source {i+1}: {text}" for i, text in enumerate(context_chunks)])

    prompt = f"""
You are an AI assistant.

STRICT RULES:
- Extract ALL relevant behaviors from the context
- Combine them into ONE clear sentence
- Do NOT invent new information
- Only include attributes present in the context

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text.strip()


# ------------------ VALIDATION ------------------

def extract_keywords(text):
    stopwords = {"the", "is", "a", "of", "and", "to", "in", "source"}
    return [w for w in text.lower().split() if w not in stopwords]


def validate_answer(answer, context_chunks):
    context = " ".join(context_chunks).lower()
    keywords = extract_keywords(answer)

    for word in keywords:
        if word in context:
            return True

    return False

#-------------intent classifier------------------

def detect_query_type(query):
    query = query.lower()

    multi_keywords = ["behavior", "describe", "tell me", "about", "explain"]

    for word in multi_keywords:
        if word in query:
            return "multi"

    return "single"



# ------------------ MAIN PIPELINE ------------------

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy dog is sleeping.",
    "The fox is quick and clever.",
    "The dog is lazy but loyal.",
    "The fox and the dog are friends."
]

queries = [
    # "fast fox",
    # "sleeping dog",
    # "loyal animal",
    # "apple",
    # "what is the color of the fox?",
    # "who is clever?",
    # "are fox and dog enemies?",
    # "tell me about tiger"
    "describe the dog",
"what does the dog do"
"tell me about the dog"
]

# precompute embeddings
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences
)

embeddings = result.embeddings


for query in queries:
    print(f"\nQuery: {query}\n")

    results = get_top_k_results(query, sentences, embeddings)

    if not results:
        print("No relevant results found.")
        continue

    # rerank results
    results = rerank(query, results)

    context_chunks = [text for text, _ in results]


    print("\nContext used:")
    for c in context_chunks:
        print("-", c)

    answer = generate_response(query, context_chunks)

    if not validate_answer(answer, context_chunks):
        print("\nAnswer rejected due to inconsistency")
    else:
        print("\nFinal Answer:")
        print(answer)