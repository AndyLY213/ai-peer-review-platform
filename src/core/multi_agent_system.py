import os
import autogen
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")

# Configuration for Ollama model
config_list = [
    {
        "model": os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b"),
        "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        "api_type": "ollama"
    }
]

# You can customize these parameters based on your requirements
llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

# Define agent systems
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human user that interacts with the AI system, providing tasks, feedback, and approving or rejecting plans.",
    human_input_mode="ALWAYS",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "workspace"
    },
)

assistant = autogen.AssistantAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant that can understand and execute code, solve problems, and help with various tasks.",
    llm_config=llm_config,
)

# Optional specialized agents
code_executor = autogen.AssistantAgent(
    name="Code_Executor",
    system_message="You specialize in executing code, debugging issues, and explaining the execution results.",
    llm_config=llm_config,
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="You specialize in breaking down complex tasks into smaller subtasks and planning the execution steps.",
    llm_config=llm_config,
)

# Create a group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, code_executor, planner],
    messages=[],
    max_round=50
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# Example execution
if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    print("🤖 Multi-Agent System is ready! Type your task and press Enter.")
    print("Type 'exit' to end the conversation.")
    
    # Start with a task
    user_proxy.initiate_chat(
        manager,
        message="Hello, I'm ready to collaborate on a task. What can you help me with today?"
    ) 