import ollama
import json
from sentence_transformers import SentenceTransformer

#Decide on the model to use:
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#Create an embedding for a query:
def encode_text(info):

    return embedding_model.encode(info).tolist()