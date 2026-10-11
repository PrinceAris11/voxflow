import importlib.util
from pathlib import Path

from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'main.py'
SPEC = importlib.util.spec_from_file_location('voxflow_main', MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

app = MODULE.app
client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
