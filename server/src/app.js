const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');

const app = express();

app.use(express.json({
    limit: "16kb"
}));

app.use(express.static("public"));

app.use(express.urlencoded({
    extended: true,
    limit: "16kb"
}));


module.exports = app;