require('dotenv').config();
const app = require('./src/app');

const PORT = process.env.SERVER_PORT || 8000;

;(async () => {

    try {
        
        // Database connection here:

        app.on('error', (error) => {
            console.log('Error: ', error);
            throw error;
        })
            
        app.listen(PORT, () => { 
                console.log(`Server is running on port ${PORT}.`);
        });        

    } catch (error) {
        console.log("Server Error :", error.message);
        throw error;
    }

})();

