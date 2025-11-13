# sqlite_patch.py
import sys
import pysqlite3
import warnings

# Suppress pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

# Force Python to use pysqlite3 as sqlite3
sys.modules["sqlite3"] = pysqlite3
