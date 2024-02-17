from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from global_conf import GPT_MODEL
import nest_asyncio
nest_asyncio.apply()

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

llm = OpenAI(model=GPT_MODEL)
agent2 = ReActAgent.from_tools([multiply_tool, add_tool, set_country_tool, get_form_info_tool], llm=llm, verbose=True)
agent = OpenAIAgent.from_tools([multiply_tool, add_tool, set_country_tool, get_form_info_tool], llm=llm, verbose=True)

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
