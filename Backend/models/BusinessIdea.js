const mongoose = require('mongoose');

const businessIdeaSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  idea: {
    type: String,
    required: true,
  },
  bmc: {
    type: mongoose.Schema.Types.Mixed,
    required: false,
  },
}, { timestamps: true });

module.exports = mongoose.model('BusinessIdea', businessIdeaSchema); 