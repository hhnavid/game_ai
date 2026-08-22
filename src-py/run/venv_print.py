import sys
import os

venv_path = sys.prefix
venv_name = os.path.basename(venv_path)
print(venv_name)
