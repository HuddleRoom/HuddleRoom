"""Live acceptance tests are opt-in.

They run only when both HUDDLEROOM_LIVE_ACCEPTANCE=1 and HUDDLEROOM_LIVE_EVAL=1 are set.
"""

import os

if not (os.environ.get("HUDDLEROOM_LIVE_ACCEPTANCE") == "1" and os.environ.get("HUDDLEROOM_LIVE_EVAL") == "1"):
    collect_ignore_glob = ["*"]
