from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import ChatPromptTemplate
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from global_conf import GPT_MODEL
from utils import CaptureStderr
from io import StringIO
import pandas as pd

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

def get_country() -> str:
    """
    Provides the country where the registration eorm
    """
    if form.country == None:
        result = "There is no information about the country. Ask the user about the country."
    else:
#        result = f"The form should include first name, last name, select of the main regions of {form.country} and phone number"
        result = f"The country is  {form.country}"
    return result

get_country_tool = FunctionTool.from_defaults(fn=get_country)

def get_registration_form_info() -> str:
    """
    Provides the basic information to create a registration form
    """
    if form.country == None:
        result = "There is no information about the country. Ask the user about the country."
    else:
#        result = f"The form should include first name, last name, select of the main regions of {form.country} and phone number"
        result = f"The country is  {form.country}"
    return result

get_form_info_tool = FunctionTool.from_defaults(fn=get_registration_form_info)

def create_csv_file(csv: str) -> str:
    """
    Creates a csv file, gets only one argument with the csv content
    """
    with CaptureStderr() as output:
        df = pd.read_csv(StringIO(csv), sep=";")
    print("HEYYYYY: ", output)
    with CaptureStderr() as output:
        df.to_excel("output/probando.xlsx")
    print("HOLAAAAAA")
    print(output)

create_csv_file_tool = FunctionTool.from_defaults(fn=create_csv_file)

# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            f"You are an expert system aimed to create registration forms for the 121 platform. \
            The 121 platform is a system to manage cash projects developed by 510, the data and \
            digital unit of Netherlands Red Cross.\
            Your objective is to guide the user to create a registration form. To do that, follow these steps:\
            1. Welcome the user explain who are you and how you can help.\
            2. Ask if they have already a form that want to use for the registration.\n\
                - In case they have a form, say that you have not yet implemented the functionality to \
                import forms and finish the conversation.\n\
                - In case they don't have a form, proceed to create one.\
            If you have any information missing, ask for it.\
            Once you have a form template, respond with the xlsForm of that form in csv format separated by semicolons and create\
            a file with the csv."
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            f"Then create a good registration form, if you have any information missing, ask for it.\n\
            A good registration form should include the following:\n\
            - An introduction that every explaining the project in simple terms to the person registered, \
                this information should include what is the project about, the National Society implementing it, \
                when it is planned to be implemented. It should also manage expectations explaining explaining that\
                the fact of being registered doesn't mean that the person will be included in the project.\n\
            - A consent question, explaining the person why their data is collected and what will be done with it.\n\
            - Information needed for the delivery mechanism:\n\
                * If the delivery mechanism is mobile money, the mobile phone should be added\n\
                * If the delivery mechanism is bank transfers, a bank account should be added\n\
                * For any other delivery mechanism, there might be other information needed.\n\
            - Information adapted to type of recipient: \n\
                * If the project is at household level, main information as first name, last \
                    name, gender (including if they prefer not to say) and date of birth should be asked.\n\
                * If the project is at individual level, main information about the person should be asked\n\
            - Information to avoid duplications: it can be id, phone number or other.\n\
            - Information about the place, including village and region from the country where the project is implemented.\n\
            - If it's a household level, it should include also dissaggregated information about household members, \
                meaning the number of members by gender (male or female) and by group of age. The groups of age may vary, so you should \
                ask the user if the ranges are as follows:\n\
                * Female children from 0 to 5 years old\n\
                * Male children from 0 to 5 years old\n\
                * Female children from 6 to 17 years old\n\
                * Male children from 6 to 17 years old\n\
                * Female from 18 to 59 years old\n\
                * Male from 18 to 59 years old\n\
                * Female of 60 or more years old\n\
                * Male of 60 or more years old\n\
            - Information of the selection criteria for the project: if the house was destroyed, livelihoods or others. If you don't\
                have information about the selection criteria, ask for them to include clear questions about them.\n\
            - Information about KYC (Know Your Customer) if necessary.\
            You can use knowledge from the Red Cross Red Crescent Movement or the humanitarian world, \
            But if the user asks anything not related to the registration form you are doing, you will \
            respond that you have been created only for this purpose.\
            The xlsForm should follow the following:\n\
            - All question should be closed questions, except if the option Other is included.\n\
            - All questions should be mandatory\
            - When possible, add a constraint to limit possible responses like negative numbers in members of households\n\
                or dates of birth from more than 120 years ago.\
            - All variables should be in camelCase"
#            "Context information is below.\n"
#            "---------------------\n"
#            "{context_str}\n"
#            "---------------------\n"
#            "Given the context information and prior knowledge, "
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
agent = OpenAIAgent.from_tools([set_country_tool, get_country_tool, create_csv_file_tool], llm=llm, verbose=True, prefix_messages=chat_text_qa_msgs,callback_manager=callback_manager)
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

print(form.country)