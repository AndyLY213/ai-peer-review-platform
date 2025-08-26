import os
import autogen
from dotenv import load_dotenv
from agent_factory import AgentFactory
from src.core.llm_config import get_llm_config, get_llm_provider_info

# Load environment variables
load_dotenv("config.env")

def setup_agent_system():
    """
    Set up the multi-agent system.
    
    Returns:
        Tuple of (user_proxy, manager) for starting conversations
    """
    # Create LLM configuration using centralized system
    llm_config = get_llm_config()
    
    # Print provider info for debugging
    provider_info = get_llm_provider_info()
    print(f"Enhanced Multi-Agent System using LLM Provider: {provider_info['provider']} - Model: {provider_info['model']}")
    
    # Create agent factory
    factory = AgentFactory(llm_config)
    
    # Create User Proxy Agent
    user_proxy = factory.create_user_proxy(
        name="User_Proxy",
        system_message="A human user that interacts with the AI system, providing tasks, feedback, and approving or rejecting plans.",
        human_input_mode="ALWAYS",
        code_execution_config={
            "last_n_messages": 2,
            "work_dir": "workspace"
        }
    )
    
    # Create Assistant Agents with different specializations
    factory.create_assistant(
        name="Assistant",
        system_message="You are a helpful AI assistant that can understand and execute code, solve problems, and help with various tasks."
    )
    
    factory.create_assistant(
        name="Code_Executor",
        system_message="You specialize in executing code, debugging issues, and explaining the execution results."
    )
    
    factory.create_assistant(
        name="Planner",
        system_message="You specialize in breaking down complex tasks into smaller subtasks and planning the execution steps."
    )
    
    factory.create_assistant(
        name="Researcher",
        system_message="You specialize in researching information, finding relevant data, and providing comprehensive answers."
    )
    
    # Create group chat with all agents
    groupchat = factory.create_group_chat()
    
    # Create group chat manager
    manager = factory.create_group_chat_manager(groupchat)
    
    return user_proxy, manager

def main():
    """
    Main function to run the multi-agent system.
    """
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    # Set up agent system
    user_proxy, manager = setup_agent_system()
    
    print("🤖 Enhanced Multi-Agent System is ready!")
    print("Type 'exit' at any time to end the conversation.")
    
    # Start the conversation
    initial_message = input("Enter your task: ")
    
    if initial_message.lower() != "exit":
        user_proxy.initiate_chat(
            manager,
            message=initial_message
        )

if __name__ == "__main__":
    main() 