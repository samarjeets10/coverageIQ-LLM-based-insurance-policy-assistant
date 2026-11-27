const express = require('express');
const { route } = require('../app');
const {
    userRegister,
    userLogin,
    userLogout
} = require('../controllers/user.controller');

const router = express.Router();

router.route('/register', userRegister);
router.route('/login', userLogin);


// secure routes :

router.route = ('./logout', userLogout);

module.exports = router;

