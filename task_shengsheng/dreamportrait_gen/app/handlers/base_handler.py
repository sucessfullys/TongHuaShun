import json
import logging
import traceback

import tornado.escape
import tornado.web
from tornado import gen

from app.exceptions.exceptions import ApplicationError, RouteNotFound, ServerError


logger = logging.getLogger("app")


class BaseApiHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        origin = self.request.headers.get("Origin", "*")
        req_headers = self.request.headers.get(
            "Access-Control-Request-Headers", "Content-Type, Authorization"
        )
        self.set_header("Access-Control-Allow-Origin", origin)
        self.set_header("Vary", "Origin")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", req_headers)
        self.set_header("Access-Control-Max-Age", "86400")

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    @gen.coroutine
    def post(self, action):
        try:
            if not hasattr(self, str(action)):
                raise RouteNotFound(action)

            handler = getattr(self, str(action))
            data = tornado.escape.json_decode(self.request.body)
            handler(data)
        except ApplicationError as e:
            logger.warning("%s: %s", e.code, e.message)
            self.respond({}, e.code, e.message)
        except Exception:
            logger.error(traceback.format_exc())
            error = ServerError()
            self.respond({}, error.code, error.message)

    def respond(self, data, code=1, msg="Success"):
        status_code = 200 if code == 0 else 505
        self.set_status(status_code)
        self.write(
            json.JSONEncoder().encode(
                {
                    "code": code,
                    "msg": msg,
                    "data": {"img": data},
                }
            )
        )
        self.finish()
