# Inference Pipeline
from llama_index.llms import OpenAI, ChatMessage, HuggingFaceLLM, MessageRole
from llama_index.prompts import PromptTemplate, ChatPromptTemplate
from global_conf import MODE

class InferencePipeline():
    """This class is the Inference Pipeline"""

    def __init__(self, index):
        """Creates the inference pipeline
        
        :param index: A vector index to use in the inference"""
        self.index = index

    def run(self):
        """Basic run of the Inference Pipeline"""
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
            query_engine = self.index.as_query_engine()
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

            chat_engine = self.index.as_chat_engine(verbose=True, text_qa_template=text_qa_template)
            chat_engine.reset()
            streaming_response = chat_engine.stream_chat(query)
            for token in streaming_response.response_gen:
                print(token, end="")

            chat_engine.chat_repl()