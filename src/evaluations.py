import csv
import time
from sentence_transformers import SentenceTransformer 
from memory_profiler import memory_usage
from chroma.chroma_ingest import 

def results(data):
    with open('testing/results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def run_experiment(embedding_model_name, vector_db_name, llm_name, documents):
    embedding_model = SentenceTransformer(embedding_model_name)
    

def main():
    with open("data/text_preprocessing_and_chunking/processed_json/ds4300_course_notes.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = [{'id': idx, 'text': doc['text'], 'metadata': doc['metadata']} for idx, doc in enumerate(data["processed_pdfs"])]

    embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    vector_dbs = ['chroma', 'redis', 'milvus']
    llms = ['llama2:7b', 'mistral7b']

    for embedding_model in embedding_models:
        for vector_db in vector_dbs:
            for llm in llms:
                print(f"Running experiment with {embedding_model}, {vector_db}, {llm}")
                run_experiment(embedding_model, vector_db, llm, documents)
    
