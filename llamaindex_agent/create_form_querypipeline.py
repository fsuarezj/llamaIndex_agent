from llama_index.core import Settings, PromptTemplate
from llama_index.core.agent import QueryPipelineAgentWorker, AgentRunner
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI

from global_conf import GPT_MODEL

import pandas as pd

Settings.llm = OpenAI(model=GPT_MODEL)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

df_form = pd.read_excel("./basic_form.xlsx")

context_str = "You are an expert system aimed to create registration forms for the 121 platform. \
                The aim of the 121 Platform is to make Cash & Voucher Assistance (CVA) easier, \
                safer and faster, and to help people affected by disasters meet their own needs. \
                It includes a portal to assist humanitarian organizations in running a safe CVA program.\
                It creates an overview with real-time updates on: registration / validation / inclusion /\
                 review inclusion / payments / monitoring and evaluation."

prompt_str = "{context_str} \n \
              Improve the following form in xlsform format including the points below. Use only information \
                given in the context or in the form. If you don't know anything, ask the user. The \
                form is as follows: \
                {form}\
                ----------------------------\
                Improve the form including the following points:\
                - Include dissaggregated data per sex and age about the members of the household\
                - Include questions about the vulnerability criteria of the project"

prompt_tmpl = PromptTemplate(prompt_str)

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "prompt_str": prompt_str,
        "llm": Settings.llm
    }
)
p.add_chain(["prompt_str", "llm"])

form_str = df_form.to_string()

#output = p.run(context_str=context_str, form=form_str)

agent_worker = QueryPipelineAgentWorker(p)
agent = AgentRunner(
    agent_worker, callback_manager=callback_manager, verbose=True
)

task = agent.create_task(context_str=context_str, form=form_str)

step_output = agent.run_step(task.task_id)