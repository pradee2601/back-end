const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const app = express();
const PORT = 3002;

// MongoDB connection
const MONGO_URI = 'mongodb://127.0.0.1:27017/bmc'; // <-- Replace with your MongoDB URI
mongoose.connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.error('MongoDB connection error:', err));

// Middleware
app.use(express.json());
app.use(cors({
  origin: 'http://localhost:3000', // Allow only your frontend
  credentials: true // If you need to send cookies or authentication headers
}));

// User routes
const userRoutes = require('./routes/user');
app.use('/api/users', userRoutes);

// Business Idea routes
const businessIdeaRoutes = require('./routes/businessIdea');
app.use('/api/business-ideas', businessIdeaRoutes);

// Sample route
app.get('/', (req, res) => {
  res.send('Hello from Express!');
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
