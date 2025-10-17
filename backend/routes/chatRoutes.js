const express = require('express');
const router = express.Router();
const { getOrCreateSession } = require('../helpers/sessionHelper');
const { buildHistory } = require('../helpers/historyHelper');
const { invokeRAG } = require('../helpers/ragHelper');

// Route: Chat Page
router.get('/chat', (req, res) => {
  let sessionId = req.cookies.sessionId || Date.now().toString();
  res.cookie('sessionId', sessionId, { maxAge: 24 * 60 * 60 * 1000, httpOnly: true });
  res.render('chat', { sessionId });
});

// API: Handle Query
router.post('/api/chat', async (req, res) => {
  const { query, sessionId } = req.body;
  if (!query) return res.status(400).json({ error: 'Query required' });

  try {
    const session = await getOrCreateSession(sessionId);
    const recentMessages = session.messages.slice(-10);
    const history = buildHistory(recentMessages);

    const { answer, escalate, suggestion } = await invokeRAG(query, history);

    // Save to Mongo
    session.messages.push({ role: 'user', content: query });
    session.messages.push({ role: 'bot', content: answer });
    await session.save();

    if (escalate) {
      console.log(`Escalation alert for session ${sessionId}: Query="${query}"`);
    }

    res.json({ answer, suggestion, escalate });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;