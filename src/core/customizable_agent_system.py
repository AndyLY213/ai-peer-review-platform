import os
import autogen
from dotenv import load_dotenv
from agent_factory import AgentFactory
from agent_templates import get_template, list_templates

# Load environment variables
load_dotenv("config.env")

def create_ollama_config():
    """
    Create configuration for Ollama model.
    
    Returns:
        Dictionary with LLM configuration
    """
    config_list = [
        {
            "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            "api_type": "ollama"
        }
    ]

    return {
        "config_list": config_list,
        "temperature": 0.7,
        "timeout": 120,
    }

class MultiAgentSystem:
    """
    Customizable multi-agent system using templates.
    """
    
    def __init__(self):
        """Initialize the multi-agent system."""
        # Create LLM configuration
        self.llm_config = create_ollama_config()
        
        # Create agent factory
        self.factory = AgentFactory(self.llm_config)
        
        # Create workspace directory if it doesn't exist
        os.makedirs("workspace", exist_ok=True)
        
        # Initialize user proxy
        self._create_user_proxy()
        
        # Track if group chat is created
        self.groupchat = None
        self.manager = None
    
    def _create_user_proxy(self):
        """Create the default user proxy agent."""
        template = get_template("user_proxy")
        self.user_proxy = self.factory.create_user_proxy(
            name=template["name"],
            system_message=template["system_message"],
            human_input_mode=template["human_input_mode"],
            code_execution_config=template["code_execution_config"]
        )
    
    def add_agent_from_template(self, template_name):
        """
        Add an agent from a template.
        
        Args:
            template_name: Name of the template to use
            
        Returns:
            True if agent was created, False otherwise
        """
        template = get_template(template_name)
        if not template:
            print(f"Template '{template_name}' not found.")
            return False
        
        # Skip user_proxy template as it's already created
        if template_name.lower() == "user_proxy":
            print("User proxy is already created.")
            return True
        
        # Create assistant agent
        self.factory.create_assistant(
            name=template["name"],
            system_message=template["system_message"]
        )
        
        print(f"Added agent: {template['name']}")
        return True
    
    def create_chat(self):
        """
        Create a group chat with all agents.
        
        Returns:
            Group chat manager
        """
        self.groupchat = self.factory.create_group_chat()
        self.manager = self.factory.create_group_chat_manager(self.groupchat)
        return self.manager
    
    def start_chat(self, initial_message):
        """
        Start a chat with the given message.
        
        Args:
            initial_message: Initial message to start the chat with
        """
        if not self.manager:
            self.create_chat()
        
        self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message
        )
    
    def list_available_templates(self):
        """
        List all available agent templates.
        
        Returns:
            List of template names
        """
        return list_templates()
    
    def list_current_agents(self):
        """
        List all current agents in the system.
        
        Returns:
            List of agent names
        """
        return self.factory.list_agents()

def interactive_setup():
    """
    Interactively set up the multi-agent system.
    
    Returns:
        Configured multi-agent system
    """
    system = MultiAgentSystem()
    
    print("Welcome to the Multi-Agent System Setup!")
    print("\nAvailable agent templates:")
    for template in system.list_available_templates():
        print(f"- {template}")
    
    print("\nThe User Proxy agent is added by default.")
    
    # Add agents
    while True:
        template = input("\nEnter a template name to add (or 'done' to finish): ")
        if template.lower() == "done":
            break
        
        system.add_agent_from_template(template)
    
    # Create the group chat
    system.create_chat()
    
    # Show configured agents
    print("\nConfigured Agents:")
    for agent_name in system.list_current_agents():
        print(f"- {agent_name}")
    
    return system

def main():
    """Main function to run the multi-agent system."""
    print("🤖 Customizable Multi-Agent System")
    print("----------------------------------")
    
    # Set up the system
    system = interactive_setup()
    
    print("\n🤖 Multi-Agent System is ready!")
    print("Type 'exit' at any time to end the conversation.")
    
    # Start the conversation
    initial_message = input("\nEnter your task: ")
    
    if initial_message.lower() != "exit":
        system.start_chat(initial_message)

if __name__ == "__main__":
    main() 