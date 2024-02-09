# Inference Pipeline
from llama_index.llms import ChatMessage, MessageRole, OpenAI
from llama_index.prompts import PromptTemplate, ChatPromptTemplate
from llama_index.tools import BaseTool, FunctionTool
from global_conf import MODE

class XlsFormsAgent:
    """This is the Agent to help creating XlsForms"""
    
    def __init__(
            self,
            tools: Sequence[BaseTool] = [],
            chat_history: List[ChatMessage] = [],
            llm: OpenAI = None
    ) -> None:
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

        #Defining tools
        self._llm = llm
        self._multiply_tool - FunctionTool.from_defaults(fn=self._multiply)
        self._add_tool - FunctionTool.from_defaults(fn=self._add)

    def _multiply(a: int, b: int) -> int:
        """Multiple two integers and returns the result integer"""
        return a * b

    def _add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    def reset(self) -> None:
        self._chat_history = []
    
    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role=MessageRole.USER, content=message))
        tools = [tool.metadata.to_openai_tool() for _, tool in self._tools.items()]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = ai_message.additional_kwargs.get("tools_calls", None)
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)
        
        return ai_message.content




class InferencePipeline():
    """This class is the Inference Pipeline"""

    def __init__(self, index, llm) -> None:
        """Creates the inference pipeline
        
        :param index: A vector index to use in the inference"""
        self._index = index
        self._llm = llm

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
            query_engine = self._index.as_query_engine()
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

            chat_engine = self._index.as_chat_engine(verbose=True, text_qa_template=text_qa_template)
            chat_engine.reset()
            streaming_response = chat_engine.stream_chat(query)
            for token in streaming_response.response_gen:
                print(token, end="")

            chat_engine.chat_repl()