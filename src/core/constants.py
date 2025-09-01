"""
Configuration constants for the Peer Review Simulation System.

This module centralizes all configuration constants, magic numbers,
and default values used throughout the application.
"""

# Token System Constants
DEFAULT_INITIAL_TOKENS = 100
MAX_TOKEN_AMOUNT = 10000
MIN_TOKEN_AMOUNT = 0
MAX_TOKEN_TRANSFER_REASON_LENGTH = 200

# Paper Database Constants
MAX_PAPER_TITLE_LENGTH = 500
MIN_PAPER_TITLE_LENGTH = 5
MAX_PAPER_ABSTRACT_LENGTH = 5000
MAX_AUTHOR_NAME_LENGTH = 100
MAX_PAPERS_PER_LOAD = 1000
DEFAULT_PAPERS_PER_LOAD = 100

# File Operation Constants
MAX_FILE_SIZE_MB = 100
BACKUP_COUNT = 5
LOG_FILE_MAX_SIZE_MB = 10
ERROR_LOG_MAX_SIZE_MB = 5

# Simulation Constants
MIN_SIMULATION_ROUNDS = 1
MAX_SIMULATION_ROUNDS = 100
MIN_INTERACTIONS_PER_ROUND = 1
MAX_INTERACTIONS_PER_ROUND = 1000
DEFAULT_SIMULATION_ROUNDS = 10
DEFAULT_INTERACTIONS_PER_ROUND = 5
MIN_RESEARCHERS_FOR_SIMULATION = 2

# LLM Configuration Constants
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_TIMEOUT = 120
MIN_LLM_TEMPERATURE = 0.0
MAX_LLM_TEMPERATURE = 2.0
MIN_LLM_TIMEOUT = 10
MAX_LLM_TIMEOUT = 600

# Validation Constants
MAX_STRING_LENGTH = 1000
MAX_RESEARCHER_ID_LENGTH = 50
MAX_PAPER_ID_LENGTH = 50
VALID_ID_PATTERN = r'^[a-zA-Z0-9_.-]+$'

# Paper Status Constants
VALID_PAPER_STATUSES = {
    "draft", "submitted", "in_review", "published", "rejected"
}

# Research Field Constants
VALID_RESEARCH_FIELDS = {
    "AI", "Artificial Intelligence",
    "NLP", "Natural Language Processing", 
    "CV", "Computer Vision",
    "Robotics", "Robotics and Control Systems",
    "Theory", "Theoretical Computer Science",
    "Ethics", "AI Ethics and Fairness",
    "Systems", "Computer Systems and Architecture",
    "HCI", "Human-Computer Interaction",
    "Security", "Cybersecurity and Privacy",
    "Data Science", "Data Science and Analytics",
    "Unknown"
}

# Specialty Compatibility Matrix
SPECIALTY_COMPATIBILITY = {
    "Artificial Intelligence": [
        "Artificial Intelligence", "Natural Language Processing", 
        "Computer Vision", "Data Science and Analytics"
    ],
    "Natural Language Processing": [
        "Natural Language Processing", "Artificial Intelligence", 
        "Data Science and Analytics"
    ],
    "Computer Vision": [
        "Computer Vision", "Artificial Intelligence", 
        "Data Science and Analytics"
    ],
    "Robotics and Control Systems": [
        "Robotics and Control Systems", "Artificial Intelligence", 
        "Computer Systems and Architecture"
    ],
    "Theoretical Computer Science": [
        "Theoretical Computer Science", "Artificial Intelligence"
    ],
    "AI Ethics and Fairness": [
        "AI Ethics and Fairness", "Artificial Intelligence", 
        "Human-Computer Interaction"
    ],
    "Computer Systems and Architecture": [
        "Computer Systems and Architecture", "Cybersecurity and Privacy", 
        "Robotics and Control Systems"
    ],
    "Human-Computer Interaction": [
        "Human-Computer Interaction", "AI Ethics and Fairness", 
        "Artificial Intelligence"
    ],
    "Cybersecurity and Privacy": [
        "Cybersecurity and Privacy", "Computer Systems and Architecture", 
        "AI Ethics and Fairness"
    ],
    "Data Science and Analytics": [
        "Data Science and Analytics", "Artificial Intelligence", 
        "Natural Language Processing", "Computer Vision"
    ]
}

# File Path Constants
DEFAULT_WORKSPACE_DIR = "peer_review_workspace"
DEFAULT_PAPERS_FILE = "papers.json"
DEFAULT_TOKENS_FILE = "tokens.json"
DEFAULT_RESULTS_FILE = "simulation_results.json"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CONFIG_FILE = "config.env"

# Dataset Constants
PEERREAD_FIELD_MAPPING = {
    "iclr": "AI", "icml": "AI", "nips": "AI", "neurips": "AI",
    "aaai": "AI", "ijcai": "AI", "aistats": "AI", "uai": "AI",
    "cvpr": "Vision", "eccv": "Vision", "iccv": "Vision", 
    "wacv": "Vision", "bmvc": "Vision",
    "acl": "NLP", "emnlp": "NLP", "naacl": "NLP", "coling": "NLP",
    "conll": "NLP", "eacl": "NLP", "tacl": "NLP", "cl": "NLP",
    "arxiv.cs.cl": "NLP",
    "sigmod": "Databases", "vldb": "Databases", "icde": "Databases",
    "kdd": "Data Mining",
    "chi": "HCI", "uist": "HCI", "cscw": "HCI",
    "siggraph": "Graphics", "eurographics": "Graphics",
    "www": "Web", "wsdm": "Web",
    "sigcomm": "Networks", "imc": "Networks", "infocom": "Networks"
}

# Review Constants
MIN_REVIEW_LENGTH = 10
MAX_REVIEW_LENGTH = 5000
DEFAULT_REVIEW_RATING_SCALE = (1, 10)

# Token Economics Constants
REVIEW_REQUEST_TOKEN_RANGE = (10, 50)  # Min and max tokens for review requests
REVIEW_COMPLETION_BONUS = 0  # Additional tokens for completing reviews
TOKEN_DECAY_RATE = 0.0  # Rate at which unused tokens decay (0 = no decay)

# Progress Reporting Constants
PROGRESS_REPORT_INTERVAL = 10  # Report progress every N items
STATISTICS_DISPLAY_LIMIT = 5  # Show top N items in statistics

# Error Handling Constants
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
CONNECTION_TIMEOUT_SECONDS = 30

# AutoGen Configuration Constants
DEFAULT_MAX_ROUNDS = 50
DEFAULT_GROUP_CHAT_TIMEOUT = 300
DEFAULT_HUMAN_INPUT_MODE = "ALWAYS"

# Logging Format Constants
LOG_FORMAT_DETAILED = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
LOG_FORMAT_SIMPLE = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Environment Variable Names
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_TIMEOUT = "LLM_TIMEOUT"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_LOG_DIR = "LOG_DIR"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"
ENV_OLLAMA_API_BASE = "OLLAMA_API_BASE"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_ANTHROPIC_MODEL = "ANTHROPIC_MODEL"

# Default Model Names
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"

# API Endpoints
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_AZURE_API_VERSION = "2023-12-01-preview"