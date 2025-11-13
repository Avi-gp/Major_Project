# sqlite_patch.py
import sys
import pysqlite3

# Monkey patch pydantic's deprecation warning before importing anything else
try:
    from pydantic import warnings as pydantic_warnings
    original_warn = pydantic_warnings.warn
    
    def patched_warn(message, category=None, stacklevel=1, **kwargs):
        # Filter out skip_file_prefixes related warnings
        if 'skip_file_prefixes' in str(message):
            return
        # Remove skip_file_prefixes from kwargs if present
        kwargs.pop('skip_file_prefixes', None)
        return original_warn(message, category=category, stacklevel=stacklevel, **kwargs)
    
    pydantic_warnings.warn = patched_warn
except:
    pass

# Force Python to use pysqlite3 as sqlite3
sys.modules["sqlite3"] = pysqlite3
