# System
from pathlib import Path
# Logging
import logging
import sys
# Common
from llama_index import ServiceContext, set_global_service_context, set_global_tokenizer
# Feature Pipeline
from llama_index import VectorStoreIndex, download_loader
import tiktoken
# Training Pipeline
# Empty
# Inference Pipeline
from llama_index.llms import OpenAI, ChatMessage
from llama_index.prompts import PromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#LLM_MODEL = "gpt-3.5-turbo"
LLM_MODEL = "gpt-3.5-turbo-0125"
#LLM_MODEL = "gpt-4"
#EMBED_MODEL = "local"
#EMBED_MODEL = "text-embedding-ada-002"
#EMBED_MODEL = "text-embedding-3-small"
#EMBED_MODEL = "text-embedding-3-large"
EMBED_MODEL = "local:BAAI/bge-small-en-v1.5"

######### INIT SYSTEM #########

llm = OpenAI(temperature=0.1, model=LLM_MODEL)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=EMBED_MODEL)
set_global_service_context(service_context)
set_global_tokenizer(tiktoken.encoding_for_model(LLM_MODEL).encode)

######### Feature Pipeline ##########
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('./data/CVAP_Guidance.pdf'))
index = VectorStoreIndex.from_documents(documents)

######## Inference Pipeline ##########

template = (
    "You are a Cash and Voucher Assistance expert from the Red Cross Red Crescent Movement.\n"
    "We have provided context information below. \n"
    "-------------------------\n"
    "{context_str}\n"
    "-------------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

qa_prompt_template = PromptTemplate(template)

query_engine = index.as_query_engine()
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_template})
query = "What are the areas of CVA Preparedness?"
response = query_engine.query(query)
print(response.get_formatted_sources())
print("query was:", query)
print(f"response was {response}")

#CVAROLE = "You are a Cash and Voucher Assistance expert from the Red Cross Red Crescent Movement."
#
#messages = [
#    ChatMessage(role="system", content=CVAROLE),
#    ChatMessage(role="user", content="What are the areas of CVA Preparedness?"),
#]
#
#response = query_engine.stream_chat(messages)
#
#for r in response:
#    print(r.delta, end="")