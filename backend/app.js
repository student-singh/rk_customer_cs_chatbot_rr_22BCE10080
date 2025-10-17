const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(bodyParser.json());
app.use(cookieParser());
app.use(express.static('public'));  // Optional: for extra CSS/JS
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
// Mount routes (add here)
const chatRoutes = require('./routes/chatRoutes');
app.use('/', chatRoutes);
// MongoDB Connect
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB error:', err));

// Root redirect
app.get('/', (req, res) => res.redirect('/chat'));

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));