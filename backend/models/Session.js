const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  role: { type: String, enum: ['user', 'bot'] },
  content: String,
  timestamp: { type: Date, default: Date.now }
});

const sessionSchema = new mongoose.Schema({
  sessionId: { type: String, unique: true, required: true },
  messages: [messageSchema],
  userId: String  // Optional
});

module.exports = mongoose.model('Session', sessionSchema);




