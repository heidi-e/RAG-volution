import chromadb

#Create the Chroma client; need PersistentClient and not Client because we do not want
#data disappearing
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#Create collection to index; pulling from existing preprocesed data
stored_collection = chroma_client.get_or_create_collection(name="ds4300_course_notes")


#Begin searching the embeddings
def search_embeddings(query, top_k=3):

    #Results will be stored:
    query_results = stored_collection.query(query_text=[query], n_results=top_k)


    for i, pdf in enumerate(query_results["documents"][0]):
        print(f'{i+1}. {pdf[:100]}')

    return query_results["documents"]


if __name__ == "__main__":
    search_embeddings("What is NoSQL?")