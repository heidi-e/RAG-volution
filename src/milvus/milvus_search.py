from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import ollama

# Connect to Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Define the embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define Milvus collection name
COLLECTION_NAME = "ds4300_course_notes"

# Load the Milvus collection
collection = Collection(COLLECTION_NAME)
collection.load()

# Convert text to embeddings
def encode_text(info):
    return embed_model.encode(info).tolist()

# Search for similar embeddings in Milvus
def search_embeddings(query, top_k=3):
    query_embedding = encode_text(query)

    # Perform ANN search in Milvus
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "chunk_size"]
    )

    # Check if results exist
    if not results or len(results[0]) == 0:
        print("No relevant documents found.")
        return []

    # Extract documents and metadata
    obtained_documents = []
    for hit in results[0]:
        text = hit.entity.get("text", "Unknown Text")
        chunk_size = hit.entity.get("chunk_size", "Unknown Size")
        obtained_documents.append(f"(Chunk Size: {chunk_size}) {text}")

    return obtained_documents

#Do not change this! All of our models have the same structure as this function.
def generate_rag_response(query, context_results):

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Interactive user search
def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == 'exit':
            break

        # Retrieve relevant documents
        context_results = search_embeddings(query)

        # Generate response
        rag_response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(rag_response)

if __name__ == "__main__":
    interactive_search()
