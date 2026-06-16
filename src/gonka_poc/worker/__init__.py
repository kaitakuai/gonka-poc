"""Worker-side extensions, mixed into the GPU worker via ``--worker-extension-cls``."""

from gonka_poc.worker.extension import PoCWorkerExtension

__all__ = ["PoCWorkerExtension"]
