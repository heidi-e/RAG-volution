import ollama
import json
from sentence_transformers import SentenceTransformer








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



#Used to set up the conversation between user and AI
def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    while True:
        query = input("\nEnter your search query: ")

        #Exit case:
        if query.lower() == 'exit':
            break

        #Otherwise, generate the RAG response:
        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        rag_response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(rag_response)



if __name__ == "__main__":
    interactive_search()
