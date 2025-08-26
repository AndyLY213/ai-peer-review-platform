# Installation Guide

This guide provides detailed installation instructions for the AI Peer Review Platform.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for AI API calls

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB for large simulations
- **Storage**: 5GB for datasets and logs
- **CPU**: Multi-core for parallel processing

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ai-peer-review-platform.git
cd ai-peer-review-platform
```

2. **Create virtual environment**:
```bash
# Using venv (recommended)
python -m venv .venv

# Or using conda
conda create -n peer-review python=3.10
conda activate peer-review
```

3. **Activate virtual environment**:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Install in development mode with all extras
pip install -e ".[dev,viz,analysis]"
```

### Method 3: Docker Installation (Coming Soon)

```bash
# Build and run with Docker
docker build -t ai-peer-review .
docker run -it --env-file .env.local ai-peer-review
```

## Configuration Setup

### 1. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env.local
```

Edit `.env.local` with your settings:
```bash
# Required: Get your API key from https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_actual_api_key_here

# Optional: Adjust model settings
GEMINI_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.7
```

### 2. Verify Installation

Test your installation:
```bash
python -c "import src.core.gemini_client; print('✓ Installation successful')"
```

Run a quick test:
```bash
python test_simple.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Make sure you're in the project root directory
cd ai-peer-review-platform
python main.py
```

#### 2. API Key Issues
```bash
# Error: Invalid API key
# Solution: Check your .env.local file
cat .env.local | grep GEMINI_API_KEY
```

#### 3. Permission Errors
```bash
# Error: Permission denied
# Solution: Check file permissions
chmod +x main.py
```

#### 4. Virtual Environment Issues
```bash
# Deactivate and recreate virtual environment
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Platform-Specific Issues

#### Windows
- Use `python` instead of `python3`
- Use backslashes in paths: `.venv\Scripts\activate`
- Install Visual C++ Build Tools if compilation errors occur

#### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Use `python3` if `python` points to Python 2

#### Linux
- Install Python development headers: `sudo apt-get install python3-dev`
- Install build essentials: `sudo apt-get install build-essential`

## Performance Optimization

### 1. Concurrent API Calls
Adjust in `.env.local`:
```bash
MAX_CONCURRENT_CALLS=5  # Increase for faster processing
REQUEST_TIMEOUT=30      # Adjust based on your connection
```

### 2. Memory Usage
For large simulations:
```bash
# Reduce number of researchers
DEFAULT_RESEARCHERS=5

# Disable heavy features
ENABLE_NETWORK_EFFECTS=false
ENABLE_PERFORMANCE_MONITORING=false
```

### 3. Logging
Reduce logging for better performance:
```bash
LOG_LEVEL=WARNING
VERBOSE_LOGGING=false
```

## Updating

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Update from Git
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

### Remove Virtual Environment
```bash
deactivate
rm -rf .venv
```

### Remove Project Files
```bash
cd ..
rm -rf ai-peer-review-platform
```

## Getting Help

If you encounter issues:

1. **Check the logs**: Look in the `logs/` directory
2. **Review the FAQ**: See common issues above
3. **Search existing issues**: [GitHub Issues](https://github.com/yourusername/ai-peer-review-platform/issues)
4. **Create a new issue**: Include your system info and error messages
5. **Join discussions**: [GitHub Discussions](https://github.com/yourusername/ai-peer-review-platform/discussions)

## System Information

To help with troubleshooting, gather this information:

```bash
# Python version
python --version

# Pip version
pip --version

# Installed packages
pip list

# System information
python -c "import platform; print(platform.platform())"
```