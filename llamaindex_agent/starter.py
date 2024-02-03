import logging
import sys
from llama_index import VectorStoreIndex, SimpleDirectoryReader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What are the areas of CVA Preparedness?")
print(response)