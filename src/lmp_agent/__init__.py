from pathlib import Path
import logging
import os
import tempfile
import warnings

_mpl_dir = Path(tempfile.gettempdir()) / "lmp_agent_mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

for logger_name in ("linopy", "pypsa", "pypsa.optimization.optimize"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message="The problem includes expressions that don't support CPP backend. Defaulting to the SCIPY backend for canonicalization.",
)

from .agent import PricingWorkflowAgent
from .config import RunConfig

__all__ = ["PricingWorkflowAgent", "RunConfig"]
