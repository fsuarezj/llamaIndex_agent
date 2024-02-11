# System
from pathlib import Path
# Logging
import logging
import sys
# Common
from llama_index import ServiceContext, set_global_service_context, set_global_tokenizer
from global_conf import LLM, GPT_MODEL, HF_MODEL, EMBED_MODEL, MODE
# Feature Pipeline
from llama_index import VectorStoreIndex, download_loader
import tiktoken
from transformers import AutoTokenizer
# Training Pipeline
# Empty
# Inference Pipeline
from inference import InferencePipeline
from llama_index.llms import OpenAI, ChatMessage, HuggingFaceLLM, MessageRole
from llama_index.prompts import PromptTemplate, ChatPromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

######### INIT SYSTEM #########

if LLM == "gpt":
    llm = OpenAI(temperature=0.1, model=GPT_MODEL)
    set_global_tokenizer(tiktoken.encoding_for_model(GPT_MODEL).encode)
elif LLM == "huggingface":
    llm = HuggingFaceLLM(model_name=HF_MODEL)
    set_global_tokenizer(AutoTokenizer.from_pretrained(HF_MODEL).encode)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=EMBED_MODEL)
set_global_service_context(service_context)

######### Feature Pipeline ###########
## 1. Load ##
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('./data/CVAP_Guidance.pdf'))
## 2. Transform ##
## 3. Index ##
index = VectorStoreIndex.from_documents(documents)

######### Inference ##########
ip = InferencePipeline(index, llm)
ip.run()