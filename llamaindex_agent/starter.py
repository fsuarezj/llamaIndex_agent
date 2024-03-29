# System
from pathlib import Path
# Logging
import logging
import sys
# Common
from llama_index.core import Settings, set_global_service_context, set_global_tokenizer, set_global_handler
from global_conf import LLM, GPT_MODEL, HF_MODEL, CPP_MODEL, EMBED_MODEL, MODE
# Feature Pipeline
from llama_index.core import VectorStoreIndex, download_loader
import tiktoken
from transformers import AutoTokenizer
# Training Pipeline
# Empty
# Inference Pipeline
from inference import InferencePipeline
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate, ChatPromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

######### INIT SYSTEM #########

if LLM == "gpt":
    llm = OpenAI(temperature=0.1, model=GPT_MODEL, verbose=True)
    set_global_tokenizer(tiktoken.encoding_for_model(GPT_MODEL).encode)
elif LLM == "huggingface":
    llm = HuggingFaceLLM(model_name=HF_MODEL)
    set_global_tokenizer(AutoTokenizer.from_pretrained(HF_MODEL).encode)
elif LLM == "llamacpp":
    print("Hola")
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
    llm = LlamaCPP(model_url=CPP_MODEL, temperature=0.1, max_new_tokens=256,
                   context_window=3900, generate_kwargs={}, model_kwargs={"n_gpu_layers":1},
                   messages_to_prompt=messages_to_prompt, completion_to_prompt=completion_to_prompt, verbose=True)

Settings.llm = llm
Settings.embed_model = EMBED_MODEL
#service_context = ServiceContext.from_defaults(llm=llm, embed_model=EMBED_MODEL)
#set_global_service_context(service_context)

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