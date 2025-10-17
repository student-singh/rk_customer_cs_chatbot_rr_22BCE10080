const Session = require('../models/Session');

async function getOrCreateSession(sessionId) {
  let session = await Session.findOne({ sessionId });
  if (!session) {
    session = new Session({ sessionId });
    await session.save();
  }
  return session;
}

module.exports = { getOrCreateSession };