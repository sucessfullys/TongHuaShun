import logging

import tornado.ioloop
import tornado.web
from tornado.options import options

from app.urls import urls


logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


application = tornado.web.Application(
    urls,
    debug=options.debug,
    autoreload=options.debug,
)


def main():
    logger.info("Starting App on Port: %s with Debug Mode: %s", options.port, options.debug)
    application.listen(options.port, address="0.0.0.0")
    tornado.ioloop.IOLoop.current().start()
