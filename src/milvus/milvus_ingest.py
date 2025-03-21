import json
import uuid
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# Connect to Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Define the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define Milvus collection name
COLLECTION_NAME = "ds4300_course_notes"


# Define the collection schema
def create_milvus_collection():
    if COLLECTION_NAME in utility.list_collections():
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False),
        FieldSchema(name="chunk_size", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=embedding_model.get_sentence_embedding_dimension()),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096)
    ]

    schema = CollectionSchema(fields, description="Embedding storage for DS4300 course notes")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create an index for fast searching
    collection.create_index(field_name="embedding",
                            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})

    print(f"Collection '{COLLECTION_NAME}' created successfully.")


create_milvus_collection()

# Load the collection
collection = Collection(COLLECTION_NAME)
collection.load()


# Convert text to embeddings
def encode_text(info):
    return embedding_model.encode(info).tolist()


# Store embeddings in Milvus
def store_embedding(info, chunk_size):
    embedding = encode_text(info)
    unique_id = str(uuid.uuid4())  # Generate a unique UUID

    # Check if the embedding already exists
    search_results = collection.query(
        expr=f'text == "{info}"',
        output_fields=["id"]
    )

    if search_results:
        print("Skipping already-stored text")
        return

    # Insert data into Milvus
    data = [[unique_id], [chunk_size], [embedding], [info]]
    collection.insert(data)
    print(f'Stored: {info[:50]} (Size: {chunk_size})')


# Pull from json database
def pull_from_json(path):
    try:
        # Open and process the data
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Loop through the processed pdfs
        for pdf in data.get("processed_pdfs", []):
            title = pdf.get("title", "Unknown Title")
            print(f'Processing: {title}')

            for chunk_size, chunked_material in pdf.get("chunked_content", {}).items():
                for text_chunk in chunked_material:
                    store_embedding(text_chunk, chunk_size)

        print("Embeddings successfully stored.")
    except Exception as e:
        print(f'Error reading JSON: {e}')


# Main function to run the script
def main():
    path = "data/text_preprocessing_and_chunking/processed_json/ds4300_course_notes.json"
    pull_from_json(path)


if __name__ == "__main__":
    main()
