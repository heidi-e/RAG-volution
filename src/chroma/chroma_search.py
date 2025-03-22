import chromadb
from sentence_transformers import SentenceTransformer
import ollama

#Use a specified model:
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#Create the Chroma client; need PersistentClient and not Client because we do not want
#data disappearing
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#Create collection to index; pulling from existing preprocesed data
stored_collection = chroma_client.get_or_create_collection(name="ds4300_course_notes")


#Create an embedding for a query:
def encode_text(info):

    return embed_model.encode(info).tolist()


#Begin searching the embeddings
def search_embeddings(query, top_k=3):

    #Encode
    encode_query = encode_text(query)

    #Results will be stored:
    query_results = stored_collection.query(query_embeddings=[encode_query], n_results=top_k)

    #Answer to if there are no answers to the query:
    if not query_results or "documents" not in query_results or not query_results["documents"]:
        return []

    #Get documents and necessary metadata (similar process as redis)
    obtained_documents = query_results["documents"][0]
    obtained_meta = query_results.get("metadatas", [[]])


    if isinstance(obtained_meta, list) and len(obtained_meta) > 0 and isinstance(obtained_meta[0], list):
        obtained_meta = obtained_meta[0]


    if not isinstance(obtained_meta, list) and len(obtained_meta) > 0 and isinstance(obtained_meta[0], list):
        obtained_meta = [{} for _ in obtained_documents]

   

    #print(f'Obtained meta: {obtained_meta} (type: {type(obtained_meta)})')


    for i, pdf in enumerate(obtained_documents):
        
        chunk_size = obtained_meta[i].get("chunk_size", "Unknown Size") if i < len(obtained_meta) and isinstance(obtained_meta[i], dict) else "Unknown Size"
        #print(f'{i+1}. (Chunk Size: {chunk_size}) {pdf}')

    return obtained_documents







def generate_rag_response(query, context_results):

    # Prepare context string
    context_str = "\n".join(
        [
            f"Chunk {i+1}: {query_result}" for i, query_result, in enumerate(context_results)
        ]
    )

    #print(f"context_str: {context_str}")

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
