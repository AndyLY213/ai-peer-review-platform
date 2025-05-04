# Peer Review Simulation System

A multi-agent simulation of the academic peer review process using AutoGen and Ollama.

## Overview

This system simulates a token-based peer review economy where AI researchers publish papers and review each other's work. Researchers spend tokens to request reviews and earn tokens by completing reviews, creating an economic cycle.

The system uses AutoGen to create a multi-agent environment with specialized researcher agents who have different areas of expertise and biases.

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

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the PeerRead dataset:

```bash
git clone https://github.com/allenai/PeerRead.git
```

3. Create a smaller test dataset for faster testing:

```bash
cd PeerReview
python src/data/create_test_dataset.py
```

4. Configure your Ollama model in `config.env`:

```
OLLAMA_MODEL=qwen3:4b
OLLAMA_API_BASE=http://localhost:11434
```

## Running the Simulation

Run the main simulation script:

```bash
python main.py
```

The script will:
1. Load papers from the test dataset (or full dataset if test dataset not found)
2. Initialize researcher agents with specialties
3. Provide options for interactive mode, automated simulation, or hybrid mode

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

## Contribution

Feel free to contribute to this project by:
- Adding new features to the simulation
- Improving agent behaviors
- Enhancing the PeerRead integration
- Adding support for other datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details. 