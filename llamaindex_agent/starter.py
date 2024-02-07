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
from transformers import AutoTokenizer
# Training Pipeline
# Empty
# Inference Pipeline
from llama_index.llms import OpenAI, ChatMessage, HuggingFaceLLM, MessageRole
from llama_index.prompts import PromptTemplate, ChatPromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

LLM = "gpt"
#LLM = "huggingface"
#GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL = "gpt-3.5-turbo-0125"
#GPT_MODEL = "gpt-4"
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
#EMBED_MODEL = "local"
#EMBED_MODEL = "text-embedding-ada-002"
#EMBED_MODEL = "text-embedding-3-small"
#EMBED_MODEL = "text-embedding-3-large"
EMBED_MODEL = "local:BAAI/bge-small-en-v1.5"
#MODE = "query"
MODE = "chat"

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

######## Inference Pipeline ##########

user_template = (
    "We have provided context information below. \n"
    "-------------------------\n"
    "{context_str}\n"
    "-------------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

text_qa_template = PromptTemplate(user_template)

query = "What are the areas of CVA Preparedness?"

if MODE == "query":
    qa_prompt_template = PromptTemplate(user_template)
    query_engine = index.as_query_engine()
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_template})
    response = query_engine.query(query)
    print(response.get_formatted_sources())
    print("Question:", query)
    print(f"Response was {response}")
elif MODE == "chat":
    #CVAROLE = "You are a Cash and Voucher Assistance expert from the Red Cross Red Crescent Movement. You base your answers only in the context"
    CVAROLE = "You are a Cash and Voucher Assistance that speaks in Shakespeare style and always finish its responses saying 'Goat Bless You'"
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=CVAROLE)
        ,
        ChatMessage(
            role=MessageRole.USER,
            content=user_template
        ),
    ]
    text_qa_template = ChatPromptTemplate(messages)

    chat_engine = index.as_chat_engine(verbose=True, text_qa_template=text_qa_template)
    chat_engine.reset()
    streaming_response = chat_engine.stream_chat(query)
    for token in streaming_response.response_gen:
        print(token, end="")
    
    chat_engine.chat_repl()