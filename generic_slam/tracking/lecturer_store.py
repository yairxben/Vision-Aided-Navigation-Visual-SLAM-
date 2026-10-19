"""The sole production boundary to the lecturer tracking store."""

# ``tracking_database.py`` is a verbatim copy of the lecturer file kept in
# ``course_exercises/``, so generic_slam no longer depends on course code.
# Keeping this import in one module keeps the lecturer API mechanically
# auditable.
import sys

from . import tracking_database
from .tracking_database import Link, NO_ID, TrackingDB

# Databases pickled before the copy reference the top-level module name
# ``tracking_database``; alias it so those files still load.
sys.modules.setdefault("tracking_database", tracking_database)

__all__ = ["Link", "NO_ID", "TrackingDB"]
