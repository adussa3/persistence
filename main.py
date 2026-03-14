############################
# HUMAN IN THE LOOP MEMORY #
############################

# load_dotenv takes all the environment variables from the .env file and adds them to the environment variables
# of the system. This allows us to access these variables in our code using os.environ or other similar methods.
from dotenv import load_dotenv
load_dotenv()

# We're importing TypeDict for our state
from typing import TypedDict

# the StateGraph is the main class for building stateful LangGraph graphs
# When we create our workflow (which descriibes the execution of nodes and edges of our agentic flow)
# we need to provide the flow with our state
#
# The state is simply going to be a data structure (usually a dictionary or pydantic class) which is maintained
# for the entire execution, and it holds the information of the execution
#
# We can store their intermediate results, LLM responses, and basically everything we can think of we can store
# Every node we run will have access to this state (it's the input for every node)
# and the nodes can also modify and update the state (by returning a new state or modifying the existing state)
#
# START is a constant that holds __start__, whihs is the key for LangGraph's default starting node
# This is the starting node in the LangGraph execution
#
# END is a constant that holds __end__, which is the key for LangGraph's default ending node
# When we reach the END node with this key, the LangGraph stops execution
from langgraph.graph import StateGraph, START, END

# MemorySaver is a checkpoint that stores the graph's state after each node execution
# but, it stores the state in memory! This storage type is emphemeral so it'll be gone upon each run of the graph's execution!
#
# However this is a good starting point to get started and have a feeling of the state objects that are checkpointed 
from langgraph.checkpoint.memory import MemorySaver

# Instead of using MemorySaver, which is empemeral, in-memory, and loses the information after each run of the program
# We'll use SQLite Saver which persists our state in storage ans into an SQLite database
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3


#######################
# DEFINE STATE SCHEMA #
#######################

class State(TypedDict):
  # User input
  input: str

  # User feedback which will be collected during the execution of the graph
  user_feedback: str

#########
# NODES #
#########

# NOTE: Were not going to the call the llm! We're just testing the human-in-the-loop functionality
def step_1_node(state: State) -> None:
  print("---Step 1---")

def human_feedback_node(state: State) -> None:
  print("---human feedback---")

def step_3_node(state: State) -> None:
  print("---Step 3---")

#########
# GRAPH #
#########

builder = StateGraph(state_schema=State)

# Nodes
builder.add_node("step_1", step_1_node)
builder.add_node("human_feedback", human_feedback_node)
builder.add_node("step_3", step_3_node)

# Edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

################
# MEMORY SAVER #
################

### Temporary

# We give the graph a memory saver when we compile it
# It's responsible for persisting the memory of the graph's state upon each graph's execution
# memory = MemorySaver()

### Permenant

# This stores the current graph state inside the database
# checkpoints.sqlite tells SQLite to run a local database in our file system/local disk instead of an external database
#
# check_same_thread=False allows us to make database operations even though we're going to run in different threads
# because in our main files, we're going to run the graph (starts the thread), stop the execution (ends the thread),
# then we're going to rerun it again (starts a different thread)
#
# If we don't mark check_same_thread as False, we get an error because we tried to edit the database through different threads
connection = sqlite3.connect(database="checkpoints.sqlite", check_same_thread=False)

# option 1
memory=SqliteSaver(connection)

# option 2
# checkpoints.sqlite tells SQLite to run a local database in our file system/local disk instead of an external database
#memory = SqliteSaver.from_conn_string("checkpoints.sqlite")

# When we compile the builder in the graph, in the "checkpointer" parameter, we pass in our memory saver
#
# The interrupt_before parameter makes it so that before we execute the "human_feedback" node, it'll stop the graph's execution
# Because we checkpointed the graph's state, we can get a user input (human feedback), and then resume our graph execution starting from where we stopped
# 
# This is because the checkpointer helps us remember where we stopped and the graph's state
# 
# The use case here is when we have user-facing applications and we want to integrate human feedback from our agent
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

###################
# VISUALIZE GRAPH #
###################

# This prints the graph visualization in the console using mermaid syntax
# We can paste this mermaid code in the mermaid live editor (https://mermaid.live/) to see the graph visualization
print(graph.get_graph().draw_mermaid())

# draw_mermaid_png(output_file_path=...) also saves the graph visualization as a png file
graph.get_graph().draw_mermaid_png(output_file_path="Persistence_Mermaid.png")

# This prints the graph visualization in the console using ascii characters
# Note: you need to install Gandalf to view the graph visualization
# print(graph.get_graph().draw_ascii())

# This saves the graph visualization as a png file
with open("Persistence.png", "wb") as f:
  f.write(graph.get_graph().draw_mermaid_png())

if __name__ == "__main__":
  # We can think of the thread ID as a session ID or a converstation ID, and this helps us differentiate between runs of our graph
  # In cases where we have different users or event different conversations with the same user using the agent
  thread = {"configurable": {"thread_id": 777}}

  initial_input = {"input": "hello world"}

  # We want to stream our graph events, so we'll use graph.stream which is a method which receives the initial input,
  # The thread (which helps us differentiate between runs), and we'll give it the stream_mode of values
  for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

  # print the next node
  print(graph.get_state(thread).next)

  # Let's get the human input
  user_input = input("Tell me how you want to update the state: ")

  # Now update the graph's state with the user input
  # We're updating the current thread (thread_id = 1)
  # We update the graph's state values with the "values" named parameter
  # Setting "as_node" to "human_feedback" will update as if the "human_feedback" node ran and updated the value in the execution
  graph.update_state(thread, values={"user_feedback": user_input}, as_node="human_feedback")

  print("--State after update--")
  print(graph.get_state(thread))

  print(graph.get_state(thread).next)

  # Here we're passing in None as the input
  for event in graph.stream(None, thread, stream_mode="values"):
    print(event)