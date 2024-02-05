# System
from pathlib import Path
# Logging
import logging
import sys
# Common
from llama_index import ServiceContext
# Feature Pipeline
from llama_index import VectorStoreIndex, set_global_tokenizer, download_loader
import tiktoken
# Training Pipeline
# Empty
# Inference Pipeline
from llama_index.llms import OpenAI, ChatMessage

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

LLM_MODEL = "gpt-3.5-turbo"
#LLM_MODEL = "gpt-4"
llm = OpenAI(temperature=0.1, model=LLM_MODEL)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

set_global_tokenizer(tiktoken.encoding_for_model(LLM_MODEL).encode)

# Feature Pipeline
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('./data/CVAP_Guidance.pdf'))
index = VectorStoreIndex(documents, service_context=service_context)

# Inference Pipeline
query_engine = index.as_query_engine()
response = query_engine.query("What are the areas of CVA Preparedness?")
print(response)

#CVAROLE = "You are a Cash and Voucher Assistance expert from the Red Cross Red Crescent Movement."
#
#messages = [
#    ChatMessage(role="system", content=CVAROLE),
#    ChatMessage(role="user", content="What are the areas of CVA Preparedness?"),
#]
#
#response = llm.stream_chat(messages)
#
#for r in response:
#    print(r.delta, end="")