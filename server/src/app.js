const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const authRoutes = require('./routes/auth.routes');

const app = express();

app.use(express.json({
    limit: "16kb"
}));

app.use(express.static("public"));

app.use(express.urlencoded({
    extended: true,
    limit: "16kb"
}));

app.use(cookieParser());

// global routes :
app.use('/api/v1/users', authRoutes);



module.exports = app;