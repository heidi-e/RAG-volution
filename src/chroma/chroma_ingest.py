import chromadb
import json
from sentence_transformers import SentenceTransformer


#Decide on the model to use:
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#Create an embedding for a query:
def encode_text(info):

    return embedding_model.encode(info).tolist()

#Create the Chroma client; need PersistentClient and not Client because we do not want
#data disappearing
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#Create collection to index; pulling from existing preprocesed data
stored_collection = chroma_client.get_or_create_collection(name="ds4300_course_notes")


#Going to obtain embeddings, create ids for them, and proceed to store them
def store_embedding(info):
    information = encode_text(info)

    #Create an id and then add it to the stored collection:
    id_gen = str(hash(info))
    stored_collection.add(documents=[info], embeddings=[information], ids=[id_gen])
    print(f'Values have been stored: {info[:50]}')


#Pull from json database:
def pull_from_json(path):

    #Open and process the data
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    
    #Loop through the processed pdfs
    for i in data["processed_pdfs"]: 
        for chunk_size, chunked_material in i["chunked_content"].items():
            for i in chunked_material:
                store_embedding(i)



#Get path to json data
def main():
    path = "data/text_preprocessing_and_chunking/processed_json/ds4300_course_notes.json"
    pull_from_json(path)

if __name__ == "__main__":
    main()