"""Single source of truth for raw-data validation: re-exports the Lab 5 schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_data import get_bike_schema, validate_schema  # noqa: E402,F401
