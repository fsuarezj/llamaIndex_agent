from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import ChatPromptTemplate
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from global_conf import GPT_MODEL

class FormInfo:
    def __init__(self):
        self.country = None

form = FormInfo()

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

def set_country(country_field) -> str:
    """
    Set the country if the user gives that information
    """
    form.country = country_field
    return "The country has been set"

set_country_tool = FunctionTool.from_defaults(fn=set_country)

def get_registration_form_info() -> str:
    """
    Provides the basic information to create a registration form
    """
    if form.country == None:
        result = "There is no information about the country. Ask the user about the country."
    else:
        result = f"The form should include first name, last name, select of the main regions of {form.country} and phone number"
    return result

get_form_info_tool = FunctionTool.from_defaults(fn=get_registration_form_info)

# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are an expert system aimed to create registration forms for the 121 platform. \
            The 121 platform is a system to manage cash projects developed by 510, the data and \
            digital unit of Netherlands Red Cross."
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            f"Your objective is to guide the user to create a registration form. So first welcome the \
            user explain who are you and how you can help. Then ask if they have already a form that \
            want to use for the registration.\
                - In case they have a form, say that you have not yet implemented the functionality to \
                import forms and finish the conversation.\
                - In case they don't have a form offer proceed to create one.\
            You can use knowledge from the Red Cross Red Crescent Movement or the humanitarian world, \
            But if the user asks anything not related to the registration form you are doing, you will \
            respond that you have been created only for this purpose.\
            Once you have a form, create the xlsForm."
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and prior knowledge, "
            "answer the question: {query_str}\n"
            "Finish the response saying 'RACATUMBA'"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

llm = OpenAI(model=GPT_MODEL)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

#agent = ReActAgent.from_tools([multiply_tool, add_tool, set_country_tool, get_form_info_tool], llm=llm, verbose=True)
#agent = ReActAgent.from_tools([multiply_tool, add_tool, set_country_tool, get_form_info_tool], llm=llm, verbose=True, prefix_messages=chat_text_qa_msgs,callback_manager=callback_manager)
agent = OpenAIAgent.from_tools([multiply_tool, add_tool, set_country_tool, get_form_info_tool], llm=llm, verbose=True, prefix_messages=chat_text_qa_msgs,callback_manager=callback_manager)
#agent.update_prompts({"text_qa_template": text_qa_template})
response = agent.chat("Hi, I'm a new user")
print(f"Agent: {str(response)}")

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {str(response)}")


prompt_dict = agent.get_prompts()
print(prompt_dict)
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")

print("----------------------")
llm_pairs = llama_debug.get_llm_inputs_outputs()
for i in llm_pairs:
    for j in i:
        if "messages" in j.payload.keys():
            print("MESSAGES")
            for h in j.payload["messages"]:
                print(h)
        if "response" in j.payload.keys():
            print("RESPONSE")
            print(j.payload["response"])
