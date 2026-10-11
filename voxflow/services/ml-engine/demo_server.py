import uvicorn
from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).resolve().with_name('main.py')
SPEC = importlib.util.spec_from_file_location('voxflow_main', MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

app = MODULE.app

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
