import chromadb
from embeddings.embedder import get_embedding

#Create the Chroma client; need PersistentClient and not Client because we do not want
#data disappearing
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#Create collection to index; pulling from existing preprocesed data
stored_collection = chroma_client.get_or_create_collection(name="ds4300_course_notes")


def store_embedding(x):
    information = get_embedding(x, "nomic-embed-text")
    stored_collection.add(documents=[x], embeddings=[information])


def main():
    display_text = "ChromaDB storage"
    store_embedding(display_text)


if __name__ == "__main__":
    main()
