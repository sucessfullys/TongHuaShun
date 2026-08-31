#!/usr/bin/env python3

from tornado.options import options

from app.config.define_global_variables import define_global_variables


define_global_variables()
options.parse_config_file("./app/config/settings.conf")
options.parse_command_line()

from app import server


if __name__ == "__main__":
    server.main()
