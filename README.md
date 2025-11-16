# AI Peer Review Platform

A comprehensive multi-agent simulation of the academic peer review process using large language models (GPT-4o-mini via OpenAI API).

## 🎯 Overview

This platform simulates a realistic academic peer review ecosystem where AI-powered researchers:
- **Publish papers** in their areas of expertise
- **Review papers** from other researchers
- **Participate in a token-based economy** (spend tokens to request reviews, earn tokens by reviewing)
- **Exhibit realistic biases and behaviors** based on their specialties and career stages
- **Navigate complex academic dynamics** including funding, career progression, and institutional pressures

## ✨ Key Features

- **🤖 Multi-Agent System**: 10+ specialized researcher agents with distinct personalities and biases
- **🧠 AI-Powered Reviews**: Realistic peer reviews generated using GPT-4o-mini via OpenAI API
- **💰 Token Economy**: Economic simulation of review requests and completions
- **📊 Advanced Analytics**: Comprehensive metrics and performance tracking
- **🎭 Bias Simulation**: Models real academic biases (confirmation bias, halo effect, etc.)
- **🏛️ Institutional Dynamics**: Career progression, funding systems, and venue prestige
- **📈 Network Effects**: Citation networks, collaboration patterns, and academic communities
- **🔄 Reproducibility Tracking**: Research reproducibility and validation systems

## New Feature: PeerRead Dataset Integration

The simulation now supports loading real research papers from the [PeerRead dataset](https://github.com/allenai/PeerRead), a dataset of scientific papers and their reviews from various computer science conferences.

### Key Features

- Load real papers and reviews from the PeerRead dataset
- Process paper metadata (title, abstract, authors, etc.)
- Extract content from paper sections
- Link papers with their reviews
- Support for smaller test datasets for faster loading

## Directory Structure

```
PeerReview/
├── config/                  # Configuration files
├── src/                     # Source code
│   ├── agents/              # Agent implementations
│   ├── core/                # Core system components
│   ├── data/                # Data management
│   │   ├── paper_database.py     # Paper database with PeerRead support
│   │   ├── create_test_dataset.py # Script to create test dataset
│   │   ├── test_paper_database.py # Test script for PaperDatabase
│   │   └── token_system.py        # Token economy system
│   └── simulation/          # Simulation logic
│       └── peer_review_simulation.py # Main simulation system
├── test_dataset/            # Smaller dataset for testing (generated)
├── main.py                  # Entry point
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (recommended: Python 3.10+)
- **OpenAI API Key** (get one at [OpenAI Platform](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ai-peer-review-platform.git
cd ai-peer-review-platform
```

2. **Create and activate a virtual environment**:
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
# Copy the example environment file
cp .env.example .env.local

# Edit .env.local and add your OpenAI API key
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_actual_api_key_here
# OPENAI_MODEL=gpt-4o-mini
```

5. **Run the simulation**:
```bash
python main.py
```

### Alternative Installation Methods

**Using pip (development install)**:
```bash
pip install -e .
```

**With optional dependencies**:
```bash
pip install -e ".[dev,viz,analysis]"
```

## 🎮 Running Simulations

### Basic Usage

```bash
python main.py
```

The platform offers three simulation modes:

1. **🤖 Automated Mode**: Run multiple rounds of interactions automatically
2. **💬 Interactive Mode**: Chat directly with researcher agents
3. **🔄 Hybrid Mode**: Set up researchers then interact manually

### Example Simulation Output

```
🔬 Peer Review Simulation System
--------------------------------
Simulation Modes:
1. Interactive Mode (chat with researchers)
2. Automated Simulation (run rounds of interactions)
3. Hybrid Mode (setup researchers and then interact)

Enter simulation mode (1-3): 2
Enter number of simulation rounds: 3
Enter number of interactions per round: 15

Creating researcher agents...
✓ Added AI_Researcher (Artificial Intelligence)
✓ Added NLP_Researcher (Natural Language Processing)
✓ Added CV_Researcher (Computer Vision)
...

Final Results:
- Total Reviews Completed: 42
- Total Tokens Exchanged: 1,250
- Top Performer: AI_Researcher (350 tokens)
```

### Testing the Paper Database

To test the PaperDatabase with the PeerRead integration:

```bash
python src/data/test_paper_database.py
```

This will:
1. Initialize a test database
2. Load papers from the test dataset
3. Print statistics about the loaded papers
4. Test basic functionality like searching and filtering

## Working with the PeerRead Dataset

### Manually Loading Papers

You can manually load papers from the PeerRead dataset:

```python
from src.data.paper_database import PaperDatabase

# Initialize database
db = PaperDatabase()

# Load from the test dataset
db.load_peerread_dataset(use_test_dataset=True)

# Or load from the full dataset with a limit
db.load_peerread_dataset(use_test_dataset=False, limit=100)

# Get paper statistics
papers = db.get_all_papers()
print(f"Loaded {len(papers)} papers")
```

### Creating a Custom Test Dataset

You can modify the `create_test_dataset.py` script to create a custom test dataset:

```python
# Configuration
SOURCE_DIR = "../../PeerRead/data"
TARGET_DIR = "../../PeerReview/test_dataset"
PAPERS_PER_CONFERENCE = 10  # Adjust number of papers per conference
INCLUDE_REVIEWS = True
```

## 🏗️ Architecture

```
ai-peer-review-platform/
├── src/                     # Source code
│   ├── agents/              # Researcher agents and templates
│   ├── core/                # Core systems (tokens, LLM clients)
│   ├── data/                # Data models and database
│   ├── enhancements/        # Advanced features (bias, networks, etc.)
│   └── simulation/          # Main simulation engine
├── tests/                   # Organized test suites
│   ├── unit/                # Unit tests for individual components
│   ├── integration/         # Integration and end-to-end tests
│   └── debug/               # Debug and development test scripts
├── peer_review_workspace/   # Generated simulation data
├── logs/                    # Application logs
├── .env.example            # Environment configuration template
├── requirements.txt        # Python dependencies
├── pytest.ini             # Test configuration
├── run_tests.py           # Test runner script
└── main.py                # Application entry point
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env.local`:

```bash
# AI Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=120

# Simulation Settings
DEFAULT_RESEARCHERS=10
DEFAULT_TOKENS=100
ENABLE_BIAS_SIMULATION=true

# Performance
MAX_CONCURRENT_CALLS=5
REQUEST_TIMEOUT=30
```

### Researcher Personalities

Each researcher has unique characteristics:
- **AI_Researcher**: Prefers practical applications
- **Theory_Researcher**: Values mathematical rigor
- **Ethics_Researcher**: Emphasizes societal impact
- **Security_Researcher**: Focuses on threat models
- And 6 more specialized researchers...

## 🧪 Testing

The project includes a comprehensive test suite organized by type:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run only integration tests  
python run_tests.py --type integration

# Quick test (just basic functionality)
python run_tests.py --quick

# Run with coverage report
python run_tests.py --coverage

# Using pytest directly
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/ -v            # All tests with verbose output
```

### Test Structure
- **`tests/unit/`** - Fast unit tests for individual components
- **`tests/integration/`** - End-to-end integration tests
- **`tests/debug/`** - Development and debugging scripts

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Submit a pull request**

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o-mini API enabling realistic agent behavior
- **PeerRead Dataset** for real academic paper data
- The academic community for inspiration and validation
- Multi-agent systems research community for foundational concepts

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-peer-review-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-peer-review-platform/discussions)
- **Email**: contact@example.com

---

**⭐ Star this repository if you find it useful!** 