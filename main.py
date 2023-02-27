# take a text prompt from the command line and use langchain to generate a response. 
# langchain will be set up to use openai or wolfram alpha depending on the text input. 

import argparse
import langchain
import os
from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from openai.error import AuthenticationError, InvalidRequestError, RateLimitError
import datetime
from langchain.agents.agent import AgentExecutor

BUG_FOUND_MSG = "Congratulations, you've found a bug in this application!"
AUTH_ERR_MSG = "Please paste your OpenAI key from openai.com to use this application. "


# use argparse to take a text prompt from the command line
parser = argparse.ArgumentParser()

parser.add_argument("prompt", help="text prompt for langchain to generate a response to, using openai (for general text) or wolfram alpha (for numeric facts and calculations)")
args = parser.parse_args()

# todo why is this not a constant in langchain?
WOLFRAM_LANGCHAIN_TOOL_NAME = "wolfram-alpha"

# langchain has two main elements: a large language model (LLM) and a tool.
# We'll use OpenAI's LLM but we could equally use AI21, Banana, Cohere, or anything else for which there is a wrapper listed in the langchain docs https://langchain.readthedocs.io/en/latest/ecosystem/bananadev.html

# how many tokens should we generate? Arbitrary, but we don't want to generate too many tokens because it will take a long time and we'll get charged for it.
# as a rule of thumb, from OpenAI's docs, 1 token = 4 characters but it depends on language
MAX_OPENAI_TOKENS = 512

def init_langchain_llm():
    """
    Initialise the large language model for langchain. In this demo we use OpenAI.
    """

    # check the OPENAI_API_KEY environment variable is set. If not, exit with error "please set OPENAI_API_KEY environment variable" 
    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("please set OPENAI_API_KEY environment variable")
    # set the LLM to use OpenAI's GPT-3
    # TODO unclear why VSCode produces error "argument missing for parameter 'client'"

    # temperature is essentially a kind of creativity parameter. The higher the temperature, the more creative the output. 0 = try to be as true to the original dataset as possible 
    llm = OpenAI(temperature=0, max_tokens=MAX_OPENAI_TOKENS)
    return llm

def init_langchain() -> AgentExecutor:
    llm = init_langchain_llm()

    # for better or worse, as of 27 Feb 2023 the first parameter (tool_names) 
    # is one of the names from one of these constants defined in load_tools: 
    # _BASE_TOOLS, _LLM_TOOLS, _EXTRA_LLM_TOOLS, _EXTRA_OPTIONAL_TOOLS
    # NOTE you need to have a Wolfram Alpha API key (free for non-commercial use). See https://developer.wolframalpha.com/portal/myapps/index.html
    tools = load_tools([WOLFRAM_LANGCHAIN_TOOL_NAME],llm=llm)

    # TODO we might not need conversational memory for a single call but 
    # "conversational-react-description" might need it so let's use it anyway and remove if it makes no difference

    # TODO not sure the importance of explicitly stating the memory_key here as in the chat-gpt-langchain demo I got it from doesn't reference it anywhere else and it has a sensible default
    memory = ConversationBufferMemory(memory_key="chat_history")

    chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
    return chain

def run_chain(chain:AgentExecutor, prompt):
    response = ""
    error_msg = ""
    try: 
        response = chain.run(prompt)
    except AuthenticationError as ae:
        error_msg = AUTH_ERR_MSG + str(datetime.datetime.now()) + ". " + str(ae)
        print("error_msg", error_msg)
    except RateLimitError as rle:
        error_msg = "\n\nRateLimitError: " + str(rle)
    except ValueError as ve:
        error_msg = "\n\nValueError: " + str(ve)
    except InvalidRequestError as ire:
        error_msg = "\n\nInvalidRequestError: " + str(ire)
    except Exception as e:
        error_msg = "\n\n" + BUG_FOUND_MSG + ":\n\n" + str(e)

    return response, error_msg

# take a text prompt from the command line, initialise langchain, and generate a response calling chain.run
chain = init_langchain()
prompt = args.prompt

response, error_msg = run_chain(chain, prompt)

print("response", response)
print("error_msg", error_msg)

# Try these two prompts. The conversational arbitrator will choose the best response from the two tools (OpenAI and Wolfram Alpha) depending on the prompt.
# "How many ping pong balls fit into a jumbo jet" -- classic Google interview question
# "Write a poem about ping pong balls and jumbo jets"
