# Autonomous BMC Generator

## Overview

The Autonomous BMC Generator is an AI-powered system that autonomously extracts, validates, and refines Business Model Canvas (BMC) segments from a business idea. It leverages LLMs, retrieval-augmented generation (RAG), and a multi-agent workflow to deliver a polished, investor-ready BMC, validation report, and relevant startup examples.

## Features
- **Business Idea Extraction**: Converts a business idea into structured BMC segments.
- **Validation**: Critically evaluates feasibility and market viability.
- **BMC Draft Generation**: Produces a refined, actionable BMC.
- **Startup Example Retrieval**: Finds relevant real-world company examples for each segment.
- **Version Control**: Maintains a history of BMC drafts and changes.
- **API Access**: FastAPI endpoint for programmatic use.

## Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies.
- An OpenAI API key (for LLM usage)

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY=your-openai-api-key
   # On Windows (cmd):
   set OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

### CLI Example
You can run the main script directly to process a business idea:

```bash
python python/main_agentic_flow.py
```

Edit the `business_idea` variable in the script to try your own idea.

### API Server
To launch the FastAPI server:

```bash
uvicorn python.main_agentic_flow:app --reload
```

#### API Endpoint
- **POST** `/generate-bmc`
  - **Body**: `{ "business_idea": "<your idea here>" }`
  - **Returns**: BMC draft, validation report, examples, version history, and errors (if any)

Example with `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/generate-bmc" -H "Content-Type: application/json" -d '{"business_idea": "A platform connecting local farmers to urban consumers."}'
```

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Project Structure
- `python/main_agentic_flow.py`: Main workflow and API implementation
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Contact
For questions or support, please open an issue or contact the maintainer. 