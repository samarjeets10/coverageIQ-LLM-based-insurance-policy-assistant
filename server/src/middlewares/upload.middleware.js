const multer = require('multer');
const { getMaxPdfSizeBytes } = require('../services/pdfValidation.service');

function uploadSinglePolicyPdf(req, res, next) {
    const upload = multer({
        storage: multer.memoryStorage(),
        limits: {
            fileSize: getMaxPdfSizeBytes(),
            files: 1,
        },
    }).single('file');

    upload(req, res, (err) => {
        if (err) {
            return next(err);
        }
        return next();
    });
}

module.exports = {
    uploadSinglePolicyPdf,
};
