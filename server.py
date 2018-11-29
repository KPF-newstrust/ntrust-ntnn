from aiohttp import web
import logging
import logging.handlers
import os

from ntnn.app import create_app, config


def init_logger():
    log = logging.getLogger(__name__)
    log.setLevel(logging[config.app.log_level])

    log_file = os.path.join(config.app.log_dir, 'app.log')
    handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='d')

    fmt = (
        '%(levelname)s: %(asctime)s pid:%(process)s '
        '%(module)s.%(funcName)s: %(message)s')
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    log.addHandler(handler)
    return log


def main():
    logger = init_logger()

    app = create_app(config.app)

    web.run_app(app, port=config.app.port, access_log=logger)


if __name__ == '__main__':
    main()
