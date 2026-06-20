from __future__ import annotations


def vllm_serve_command(
    model_id: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str = "local-routecode",
) -> list[str]:
    return [
        "vllm",
        "serve",
        str(model_id),
        "--host",
        str(host),
        "--port",
        str(port),
        "--dtype",
        "auto",
        "--api-key",
        str(api_key),
    ]
