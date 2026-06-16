"""PoC sub-package.

Intentionally inert: importing ``gonka_poc.poc`` MUST NOT trigger any
side effects (no engine patching, no router registration, no model
imports). Each consumer pulls what it needs from the explicit module
path, e.g.::

    from gonka_poc.poc.routes import router as poc_router
    from gonka_poc.poc.data import encode_vector, decode_vector
    from gonka_poc.poc.poc_model_runner import execute_poc_forward

The previous version eagerly imported engine_patch (which monkey-patched
AsyncLLM.poc_request) and manager (V0-only PoCManager). Both are removed:
PoC dispatch is now via ``collective_rpc("execute_poc_forward", kwargs=...)``
on :class:`gonka_poc.worker.PoCWorkerExtension`.
"""
