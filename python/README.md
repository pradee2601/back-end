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
- A Gemini API key (for Google Gemini model usage)
- A Trivily API key (for Trivily service integration)

## Setup
1. **Clone the repository**
2. **(Recommended) Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY=your-openai-api-key
   # On Windows (cmd):
   set OPENAI_API_KEY=your-openai-api-key
   ```
5. **Set your Gemini API key**:
   ```bash
   export GEMINI_API_KEY=your-gemini-api-key
   # On Windows (cmd):
   set GEMINI_API_KEY=your-gemini-api-key
   ```
6. **Set your Trivily API key**:
   ```bash
   export TRIVILY_API_KEY=your-trivily-api-key
   # On Windows (cmd):
   set TRIVILY_API_KEY=your-trivily-api-key
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
The following environment variables are required:
- `OPENAI_API_KEY`: Your OpenAI API key (required for LLM features)
- `GEMINI_API_KEY`: Your Gemini API key (required for Google Gemini model features)
- `TRIVILY_API_KEY`: Your Trivily API key (required for Trivily service integration)

## Project Structure
- `main_agentic_flow.py`: Main workflow and API implementation (FastAPI app, CLI entry point)
- `flask_bmc_api.py`: (Legacy/alternative) Flask-based API implementation (if present)
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Contact
For questions or support, please open an issue or contact the maintainer. 