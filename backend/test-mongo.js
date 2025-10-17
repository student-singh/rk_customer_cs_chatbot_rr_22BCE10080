require('dotenv').config();
const mongoose = require('mongoose');
const Session = require('./models/Session'); // <-- This must come before any use of Session

mongoose.connect(process.env.MONGODB_URI)
  .then(() => {
    console.log('Connected');
    Session.findOne({ sessionId: 'test123' }).then(existing => {
      if (!existing) {
        const testSession = new Session({ sessionId: 'test123' });
        testSession.save().then(() => console.log('Saved')).catch(console.error);
      } else {
        console.log('Session already exists');
      }
    });
  });