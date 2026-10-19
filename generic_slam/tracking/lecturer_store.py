"""The sole approved production boundary to the lecturer tracking store."""

# The protected store is intentionally shared, not copied or modified. Keeping
# this import in one module makes the exception to the otherwise independent
# generic_slam dependency policy explicit and mechanically auditable.
from tracking_database import Link, NO_ID, TrackingDB

__all__ = ["Link", "NO_ID", "TrackingDB"]
