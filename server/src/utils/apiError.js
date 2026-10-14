class apiError extends Error {
    constructor(
        statusCode,
        message="Something went wrong",
        error= [],
        stack []
    ) {
        super(message)

        
    }
}