# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LeetCode Assistant built with LangGraph that helps analyze and solve LeetCode problems using AI agents. The application uses a multi-agent workflow to break down problem-solving into specialized steps.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,jupyter]"
```

### Testing and Quality
```bash
# Run tests
pytest

# Run tests with async support
pytest -v

# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with mypy
mypy src/
```

### Running the Application
```bash
# CLI interface
leetcode-assistant

# Or directly
python -m src.cli
```

## Architecture

The project follows a multi-agent architecture using LangGraph:

### Core Components

1. **Agents** (`src/agents/`): Specialized AI agents for different aspects of problem solving
   - `problem_analyzer.py`: Analyzes LeetCode problem statements
   - `solution_strategist.py`: Develops solution strategies and approaches
   - `code_generator.py`: Generates actual code implementations
   - `test_runner.py`: Executes and validates generated solutions
   - `explainer.py`: Provides explanations of solutions and concepts

2. **Graph Workflow** (`src/graph/`): LangGraph workflow orchestration
   - `state.py`: Defines the state structure passed between agents
   - `workflow.py`: Orchestrates the multi-agent workflow execution

3. **Utilities** (`src/utils/`): Helper modules
   - `leetcode_parser.py`: Parses LeetCode problem formats and extracts test cases
   - `code_executor.py`: Safe execution environment for running generated code

4. **CLI Interface** (`src/cli.py`): Command-line interface using Click and Rich

### Workflow Flow
The typical execution follows: Problem Analysis → Strategy Development → Code Generation → Testing → Explanation

### Dependencies
- **LangGraph/LangChain**: Multi-agent workflow orchestration
- **OpenAI/Anthropic**: LLM providers for agents
- **Click**: CLI framework
- **Rich**: Enhanced terminal output
- **BeautifulSoup/requests**: Web scraping for LeetCode problems
- **Pydantic**: Data validation and serialization

## Development Notes

- The project uses Black (line length 88) and isort for code formatting
- Type hints are enforced via mypy
- Virtual environment is located in `venv/`
- Jupyter notebooks are available in `notebooks/` for experimentation
- Test files follow pytest conventions in `tests/`