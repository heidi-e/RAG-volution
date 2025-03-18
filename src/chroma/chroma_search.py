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
    query_results = stored_collection.query(query_embeddings=[query], n_results=top_k)


    for i, pdf in enumerate(query_results["documents"][0]):
        print(f'{i+1}. {pdf[:100]}')

    return query_results["documents"]







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





if __name__ == "__main__":
    search_embeddings("What is NoSQL?")