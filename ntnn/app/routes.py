from aiohttp import web


def categories(model):
    async def handle(req):
        data = await req.post()
        if 'contents' not in data:
            raise web.HTTPBadRequest()

        cls = await model.predict(data['contents'])
        return web.json_response({'class': cls})
    return handle


async def entities(model, req):
    data = await req.post()
    if 'contents' not in data:
        raise web.HTTPBadRequest()

    pers, orgs, locs = await model.predict(data['contents'])
    return web.json_response({
        'persons': list(pers),
        'organizations': list(orgs),
        'locations': list(locs)
    })


async def quotes(model, req):
    data = await req.post()
    if 'contents' not in data:
        raise web.HTTPBadRequest()

    reals, anons = await model.predict(data['contents'])
    return web.json_response({
        'real_speakers': list(reals),
        'anon_speakers': list(anons)
    })
