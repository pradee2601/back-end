# Project Overview

This repository contains a dual-backend system:
- **Backend**: A Node.js/Express server for handling business ideas and user management.
- **Python**: A Flask-based API for additional business model computations or agentic flows.

## Directory Structure

```
back-end/
  Backend/           # Node.js/Express backend
    controllers/     # Express controllers
    models/          # Mongoose models
    routes/          # Express routes
    server.js        # Entry point for Node.js server
    package.json     # Node.js dependencies
  python/            # Python/Flask backend
    flask_bmc_api.py # Flask API
    main_agentic_flow.py # Main agentic flow logic
    requirements.txt # Python dependencies
```

## Setup Instructions

### 1. Node.js Backend

1. Navigate to the Backend directory:
   ```sh
   cd Backend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the server:
   ```sh
   npm start
   # or
   node server.js
   ```

### 2. Python Flask API

1. Navigate to the python directory:
   ```sh
   cd python
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Flask API:
   ```sh
   python flask_bmc_api.py
   ```

## Usage
- The Node.js backend will be available at the port specified in `server.js` (default: 3000).
- The Flask API will be available at the port specified in `flask_bmc_api.py` (default: 5000).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[Specify your license here]

Project Architecture
https://drive.google.com/file/d/1Nt004eQqP3gmGcN3FOwMfA96K-KHuJ-v/view?usp=drive_link