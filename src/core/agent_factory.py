import os
import autogen
from typing import Dict, List, Any, Optional

class AgentFactory:
    """
    Factory class for creating and managing different types of agents.
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize the AgentFactory with the LLM configuration.
        
        Args:
            llm_config: Configuration for the language model
        """
        self.llm_config = llm_config
        self.agents = {}
    
    def create_user_proxy(
        self,
        name: str = "User_Proxy",
        system_message: str = "A human user that interacts with the AI system.",
        human_input_mode: str = "ALWAYS",
        code_execution_config: Optional[Dict[str, Any]] = None
    ) -> autogen.UserProxyAgent:
        """
        Create a user proxy agent.
        
        Args:
            name: Name of the agent
            system_message: System message describing the agent's role
            human_input_mode: Mode for human input ('ALWAYS', 'NEVER', or 'TERMINATE')
            code_execution_config: Configuration for code execution
            
        Returns:
            The created UserProxyAgent
        """
        if code_execution_config is None:
            code_execution_config = {
                "last_n_messages": 2,
                "work_dir": "workspace"
            }
            
        agent = autogen.UserProxyAgent(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config
        )
        
        self.agents[name] = agent
        return agent
    
    def create_assistant(
        self,
        name: str,
        system_message: str,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> autogen.AssistantAgent:
        """
        Create an assistant agent.
        
        Args:
            name: Name of the agent
            system_message: System message describing the agent's role
            llm_config: Custom LLM configuration (uses default if None)
            
        Returns:
            The created AssistantAgent
        """
        agent_llm_config = llm_config if llm_config is not None else self.llm_config
        
        agent = autogen.AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=agent_llm_config
        )
        
        self.agents[name] = agent
        return agent
    
    def create_group_chat(
        self,
        agents: Optional[List[autogen.Agent]] = None,
        agent_names: Optional[List[str]] = None,
        max_round: int = 50
    ) -> autogen.GroupChat:
        """
        Create a group chat with the specified agents.
        
        Args:
            agents: List of agents to include
            agent_names: List of agent names to include (if agents not provided)
            max_round: Maximum number of chat rounds
            
        Returns:
            The created GroupChat
        """
        if agents is None:
            if agent_names is None:
                # Use all created agents if none specified
                agents = list(self.agents.values())
            else:
                # Use specified agent names
                agents = [self.agents[name] for name in agent_names if name in self.agents]
        
        return autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round
        )
    
    def create_group_chat_manager(
        self,
        groupchat: autogen.GroupChat,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> autogen.GroupChatManager:
        """
        Create a group chat manager.
        
        Args:
            groupchat: The group chat to manage
            llm_config: Custom LLM configuration (uses default if None)
            
        Returns:
            The created GroupChatManager
        """
        manager_llm_config = llm_config if llm_config is not None else self.llm_config
        
        return autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=manager_llm_config
        )
    
    def get_agent(self, name: str) -> Optional[autogen.Agent]:
        """
        Get an agent by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            The agent if found, None otherwise
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        List all created agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys()) 