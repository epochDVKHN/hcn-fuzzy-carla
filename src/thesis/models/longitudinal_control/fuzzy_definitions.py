import logging
import sys
from pathlib import Path

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)

# Include the project root for relative module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class FLC_Cruise:
    def __init__(self) -> None:
        pass


class FLC_Follow:
    def __init__(self) -> None:
        pass


class FLC_Emergency:
    def __init__(self) -> None:
        pass
