"""Small round-robin OpenAI-compatible proxy for two local vLLM replicas."""

from __future__ import annotations

import itertools
import os

from aiohttp import ClientSession, ClientTimeout, web


BACKENDS = tuple(
    value.strip()
    for value in os.environ.get(
        "CAUSALITYRAG_PROXY_BACKENDS",
        "http://127.0.0.1:8002,http://127.0.0.1:8003",
    ).split(",")
    if value.strip()
)
HOP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "connection",
}
ROUND_ROBIN = itertools.cycle(BACKENDS)


async def handle(request: web.Request) -> web.StreamResponse:
    backend = next(ROUND_ROBIN)
    url = backend + request.rel_url.raw_path_qs
    body = await request.read()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_HEADERS
    }
    try:
        response = await request.app["session"].request(
            request.method,
            url,
            data=body,
            headers=headers,
            timeout=ClientTimeout(total=None, sock_connect=10, sock_read=None),
        )
    except Exception as exc:
        return web.Response(status=502, text=f"proxy backend error ({backend}): {exc}")
    output = web.StreamResponse(
        status=response.status,
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower() not in HOP_HEADERS
        },
    )
    await output.prepare(request)
    async for chunk in response.content.iter_any():
        await output.write(chunk)
    await output.write_eof()
    return output


async def startup(app: web.Application) -> None:
    app["session"] = ClientSession()


async def cleanup(app: web.Application) -> None:
    await app["session"].close()


def main() -> None:
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    port = int(os.environ.get("CAUSALITYRAG_PROXY_PORT", "8000"))
    web.run_app(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
