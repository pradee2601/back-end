const BusinessIdea = require('../models/BusinessIdea');
const axios = require('axios');

// Create a new business idea
exports.createBusinessIdea = async (req, res) => {
  try {
    const { userId, idea } = req.body;
    // Call the external API with the business idea
    const bmcResponse = await axios.post('http://127.0.0.1:8000/generate-bmc', {
      business_idea: idea
    });
    const bmcData = bmcResponse.data;
    // Save the business idea along with bmc data
    const newIdea = new BusinessIdea({ userId, idea, bmc: bmcData });
    await newIdea.save();
    res.status(201).json({
      success: true,
      id: newIdea._id,
      bmc_draft: bmcData.bmc_draft,
      validation_report: bmcData.validation_report,
      company_examples: bmcData.company_examples,
      version_history: bmcData.version_history,
      processing_times: bmcData.processing_times,
      total_processing_time: bmcData.total_processing_time,
      errors: bmcData.errors
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Get all business ideas for a user
exports.getBusinessIdeasByUser = async (req, res) => {
  try {
    const { userId, ideaId } = req.params;
    let ideas;
    if (ideaId) {
      ideas = await BusinessIdea.findOne({ userId, _id: ideaId });
      if (!ideas) {
        return res.status(404).json({ error: 'Idea not found' });
      }
    } else {
      ideas = await BusinessIdea.find({ userId });
    }
    res.status(200).json(ideas);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Update version history for a business idea
exports.updateVersionHistory = async (req, res) => {
  try {
    const { ideaId } = req.params;
    // Receive the full object from the frontend
    const rollbackPayload = req.body;

    // Send the full object to the rollback endpoint
    const rollbackResponse = await axios.post('http://127.0.0.1:8000/rollback', rollbackPayload);
    const newVersionData = rollbackResponse.data;

    // Find and update the business idea
    const idea = await BusinessIdea.findById(ideaId);
    if (!idea) {
      return res.status(404).json({ error: 'Idea not found' });
    }
    if (!idea.bmc) {
      idea.bmc = {};
    }
    // Save the new version data (assuming version_history is the field to update)
    idea.bmc.version_history = newVersionData.version_history;
    await idea.save();
    res.status(200).json({ success: true, version_history: idea.bmc.version_history });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}; 