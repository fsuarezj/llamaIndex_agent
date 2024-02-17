from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from global_conf import GPT_MODEL

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenAI(model=GPT_MODEL)
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response_gen = agent.chat("What is 20+(2*4)? Calculate step by step")
prompt_dict = agent.get_prompts()
print(prompt_dict)
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")
