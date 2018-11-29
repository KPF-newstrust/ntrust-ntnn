import pytest
from .seeds import *  # NOQA

from ntnn.app import create_app, config


@pytest.fixture
async def cli(test_client, loop):
    app = create_app(config.app, loop=loop)
    client = await test_client(app)
    return client


async def test_should_return_category(cli, samples):
    data = {'contents': samples[0]}

    res = await cli.request('POST', '/categories', data=data)
    assert res.status == 200

    body = await res.json()
    assert 'class' in body

    print(body)
