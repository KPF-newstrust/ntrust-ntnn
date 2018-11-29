import asyncio
from aiohttp import web
from functools import partial
import os

from ntnn.app import routes
from ntnn.app.model import Classifier, NER, QER


def init_classifier(model_dir, conf):
    return Classifier(
        os.path.join(model_dir, 'classifier', str(conf.version)))


def init_ner(model_dir, conf):
    return NER(
        os.path.join(model_dir, 'ner', str(conf.version)),
        os.path.join(model_dir, 'w2v', str(conf.w2v_version)))


def init_qer(model_dir, conf):
    return QER(
        os.path.join(model_dir, 'qer', str(conf.version)))


def create_app(conf, loop=None):
    app = web.Application(loop=loop or asyncio.get_event_loop())
    model_dir = conf.model_dir

    classifier = init_classifier(model_dir, conf.classifier)
    categories = routes.categories(classifier)
    app.router.add_post('/categories', categories)

    ner = init_ner(model_dir, conf.NER)
    app.router.add_post(
        '/entities', partial(routes.entities, ner))

    qer = init_qer(model_dir, conf.QER)
    app.router.add_post(
        '/quotes', partial(routes.quotes, qer))
    return app
