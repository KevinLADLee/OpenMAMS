import os
import re
import sys
import traceback
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph.message import add_messages

sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper

from remembr.memory.memory import Memory

from remembr.agents.agent import Agent, AgentOutput
from remembr.config.config_loader import Config, get_config
from remembr.factories import load_llm, load_embeddings




def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

def try_except_continue(state, func):
    while True:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue

class ReMEmbRAgent(Agent):

    def __init__(
        self,
        llm_type: Optional[str] = None,
        num_ctx: int = 8192,
        temperature: float = 0,
        config: Optional[Config] = None
    ):
        """
        Initialize ReMEmbR Agent.

        Args:
            llm_type: LLM model name (deprecated, use config instead)
            num_ctx: Context window size
            temperature: Sampling temperature
            config: Configuration object (if None, will auto-load from config.yaml)

        Examples:
            # New config-based approach (recommended)
            >>> from remembr.config.config_loader import load_config
            >>> config = load_config()
            >>> agent = ReMEmbRAgent(config=config)

            # Legacy approach (backward compatible)
            >>> agent = ReMEmbRAgent(llm_type='qwen3:8b')
        """
        # Load config if not provided
        if config is None:
            try:
                config = get_config()
            except:
                # If config not loaded, create minimal config for backward compatibility
                from remembr.config.defaults import get_default_config
                config = Config(get_default_config())

        self.config = config

        # Support legacy parameters with config fallback
        if llm_type is None:
            llm_type = config.get('llm.backend', 'qwen3:8b')

        self.llm_type = llm_type
        self.num_ctx = num_ctx
        self.temperature = temperature

        # Load LLM using factory (with parameter overrides)
        llm = load_llm(
            config,
            llm_type=llm_type,
            temperature=temperature,
            num_ctx=num_ctx
        )
        chat = FunctionsWrapper(llm)
        self.chat = chat

        # Load embeddings using factory
        self.embeddings = load_embeddings(config)

        # Load prompts
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.agent_prompt = file_to_string(top_level_path+'prompts/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+'prompts/agent_gen_system_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0

        self.chat_history = ChatMessageHistory()




    def set_memory(self, memory: Memory):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()



    def create_tools(self, memory):

        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_text(x),
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: tuple[float, float, float] = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                                                    The query must be an (x,y,z) array with floating point values \
                                                    Based on the question and your context, decide what position to search for in the database. \
                                                    This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. \
                                                    The query will then search your memories for you.")
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                                The query must be a string containing only time. \
                                Based on the question and your context, decide what time to search for in the database. \
                                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        # position-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_time(x),
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an H:M:S time.",
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    ### Nodes

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.chat


        # limit to 5 tool calls.
        if self.agent_call_count < 3:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt


        agent_prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", prompt),
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),

            ]
        )


        model = agent_prompt | model

        question = f"The question is: {messages[0]}"

        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        # if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
        #     for i in range(len(messages)):
        #         if type(messages[i]) == ToolMessage:
        #             messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id


        response = model.invoke({"question": question, "chat_history": messages[:]})

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("[{}]", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"

        self.agent_call_count += 1


        return {"messages": [response]}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please responsed in the desired format."
        last_message = messages[-1]


        docs = last_message.content

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})


        gen_prompt = ChatPromptTemplate.from_messages(
            [
                # ("human", "What do you do?"),
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                # ("ai", filled_prompt.text),
                ("human", "{question}"),

            ]
        )

        model = gen_prompt | self.chat

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = eval(response) 
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["time", "text", "binary", "position", "duration"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")
                
            if type(parsed['position']) == str:
                parsed['position'] = eval(parsed['position'])
            
            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except:
            raise ValueError("Generate call failed. Retrying...")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}



    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        # retrieve = ToolNode([self.retriever_tool])
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)
        # workflow.add_node("action", lambda state: try_except_continue(state, tool_node))


        # workflow.add_node("action", self.call_tool)

        workflow.add_node(
            "generate", lambda state: try_except_continue(state, self.generate)
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not


        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )


        workflow.add_edge('action', 'agent')

        workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()


    def query(self, question: str):

        inputs = { "messages": [
                                (("user", question)),
            ]
        }

        # print("inputs", inputs)

        out = self.graph.invoke(inputs)

        # print("Raw output:")
        # print(out)

        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        # print("Raw response:")
        # print(response)

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = eval(response) 
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)
        print(response)

        return response
