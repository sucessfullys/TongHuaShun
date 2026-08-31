class ApplicationError(Exception):
    code = 1
    message = "Application Error"

    def __init__(self, message=None, code=None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        super().__init__(self.message)


class RequestError(ApplicationError):
    code = 1
    message = "Request Error"


class RouteNotFound(ApplicationError):
    code = 404

    def __init__(self, route):
        super().__init__(f"Route not found: {route}", self.code)


class ServerError(ApplicationError):
    code = 1
    message = "Internal Error"
