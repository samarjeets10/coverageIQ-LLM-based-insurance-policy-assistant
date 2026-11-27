const userRoute = require('../routes/auth.routes');
const jwt = require('jsonwebtoken');
const asyncHandler = require('../utils/asyncHandler');
const apiResponse = require('../utils/apiResponse');
const apiError = require('../utils/apiError');

generateAccessAndRefreshToken = async (userId) => {
    try {

        const user = await UserModel.

    } catch (error) {
        throw new apiError(500, "Something wentwrong while generating tokens!");
    }
}


const userRegister = asyncHandler(async (req, res) => {

    const {
        fullName,
        email,
        password,
        role
    } = req.body;

    if ([fullName, email, password, role].some(
        (field) => typeof field !== "string" || field.trim() === ""
    )) {
        throw new apiError(400, "All fields are required");
    }

    const existedUser = await UserModel.findOne({
        $or: [{ fullName, email }]
    });

    if (existedUser) {
        throw new apiError(409, "User with this credentials is alreasy exists");
    }


});


const userLogin = asyncHandler(async () => {



});


const userLogout = asyncHandler(async () => {




})


module.exports = {
    userRegister,
    userLogin,
    userLogout
}