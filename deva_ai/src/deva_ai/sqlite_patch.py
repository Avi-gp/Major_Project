# sqlite_patch.py
import sys
import pysqlite3

# Force Python to use pysqlite3 as sqlite3
sys.modules["sqlite3"] = pysqlite3
