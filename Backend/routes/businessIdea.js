const express = require('express');
const router = express.Router();
const businessIdeaController = require('../controllers/businessIdeaController');

// Route to create a new business idea
router.post('/idea', businessIdeaController.createBusinessIdea);

// Route to get all business ideas for a user
router.get('/:userId', businessIdeaController.getBusinessIdeasByUser);

// Route to get a specific business idea for a user
router.get('/:userId/:ideaId', businessIdeaController.getBusinessIdeasByUser);

// Route to update version history for a business idea
router.put('/idea/:ideaId/version-history', businessIdeaController.updateVersionHistory);

module.exports = router; 