from pathlib import Path

namespace = {"__name__": "__main__"}
base = Path(__file__).resolve().parent

for path in sorted(base.glob("[0-9][0-9]_*.py")):
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, namespace, namespace)
