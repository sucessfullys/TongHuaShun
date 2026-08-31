from app.handlers.base_handler import BaseApiHandler


class IndexHandler(BaseApiHandler):
    """App is live."""

    def get(self):
        self.write("I am fine!")

    def head(self):
        self.finish()
