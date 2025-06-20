# Backend API for Autonomous BMC Generator

## Overview
This backend provides RESTful API endpoints for user authentication and business idea management, integrating with an AI-powered BMC (Business Model Canvas) generator. It is built with Node.js, Express, and MongoDB, and communicates with a Python-based AI service for BMC generation and validation.

## Features
- User registration and login (with password hashing)
- Create, retrieve, and manage business ideas
- Integrates with an AI service to generate and validate BMCs
- Version history management for business ideas
- CORS enabled for frontend integration

## Requirements
- Node.js (v14+ recommended)
- MongoDB (local or remote instance)
- Python AI service running at `http://127.0.0.1:8000` (see `/python` folder)

## Setup
1. **Install dependencies:**
   ```bash
   npm install
   ```
2. **Configure MongoDB:**
   - By default, connects to `mongodb://127.0.0.1:27017/bmc` (edit `server.js` if needed).
3. **Start the server:**
   ```bash
   npm run dev
   ```
   The server runs on [http://localhost:3002](http://localhost:3002).

## Usage
- The backend exposes RESTful endpoints under `/api/users` and `/api/business-ideas`.
- Make sure the Python AI service is running for BMC generation features.

## API Endpoints
### User
- `POST /api/users/signup` — Register a new user
  - Body: `{ name, email, password }`
- `POST /api/users/login` — Login
  - Body: `{ email, password }`

### Business Ideas
- `POST /api/business-ideas/idea` — Create a new business idea and generate BMC
  - Body: `{ userId, idea }`
- `GET /api/business-ideas/:userId` — Get all business ideas for a user
- `GET /api/business-ideas/:userId/:ideaId` — Get a specific business idea for a user
- `PUT /api/business-ideas/idea/:ideaId/version-history` — Update version history (rollback)
  - Body: Full rollback payload

## Project Structure
```
Backend/
  controllers/         # Route logic (user, business idea)
  models/              # Mongoose models (User, BusinessIdea)
  routes/              # Express route definitions
  server.js            # Main server entry point
  package.json         # Project metadata and dependencies
```

## Dependencies
- express
- mongoose
- cors
- axios
- bcryptjs
- nodemon (dev)

## Notes
- Environment variables (such as MongoDB URI) can be managed via a `.env` file (add `.env` to `.gitignore`).
- The backend expects the Python AI service to be running for BMC-related endpoints.

## License
ISC 