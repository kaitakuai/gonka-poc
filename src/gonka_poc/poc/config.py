"""PoC state enum and env-tunable knobs.

Every ``POC_*`` environment variable consumed by the PoC API layer is
parsed here exactly once; ``routes`` / ``generate_queue`` / ``callbacks``
import the resulting constants. A default change in one place cannot
silently diverge between the wait-path and the queue-path.
"""
import os
from enum import Enum


class PoCState(Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPED = "STOPPED"


# --- generate / queue knobs ------------------------------------------------
POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))
POC_GENERATE_RESULT_TTL_SEC = float(os.environ.get("POC_GENERATE_RESULT_TTL_SEC", "300"))
POC_MAX_QUEUED_NONCES = int(os.environ.get("POC_MAX_QUEUED_NONCES", "100000"))
POC_RPC_TIMEOUT_MS = int(os.environ.get("POC_RPC_TIMEOUT_MS", "60000"))
POC_BATCH_SIZE_DEFAULT = int(os.environ.get("POC_BATCH_SIZE_DEFAULT", "32"))

# Poll interval while /generate work (wait-path or queued) spins waiting for
# an active /init/generate mining round to release the GPU.
GENERATION_ACTIVE_POLL_SEC = 0.1

# --- callback knobs --------------------------------------------------------
POC_CALLBACK_INTERVAL_SEC = float(os.environ.get("POC_CALLBACK_INTERVAL_SEC", "5"))
POC_CALLBACK_MAX_ARTIFACTS = int(os.environ.get("POC_CALLBACK_MAX_ARTIFACTS", "1000000"))
POC_CALLBACK_MAX_RETRIES = int(os.environ.get("POC_CALLBACK_MAX_RETRIES", "10"))
POC_CALLBACK_MAX_CONCURRENT = int(os.environ.get("POC_CALLBACK_MAX_CONCURRENT", "10"))
POC_CALLBACK_QUEUE_SIZE = int(os.environ.get("POC_CALLBACK_QUEUE_SIZE", "10000"))
