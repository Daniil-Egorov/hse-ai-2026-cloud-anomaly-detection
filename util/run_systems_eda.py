"""Execute the four EDA notebooks in fresh kernels using this Python environment.

Usage from repository root:
    python -m util.run_systems_eda
    python -m util.run_systems_eda OpenSSH Mac
"""

import argparse
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]


class ProgressClient(NotebookClient):
    """Forward long source-scan progress while retaining it in cell outputs."""

    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream" and "Thunderbird:" in msg["content"].get("text", ""):
            print(msg["content"]["text"].strip(), flush=True)
        return result


def execute(name: str, timeout: int = 7200) -> None:
    path = ROOT / "notebooks" / f"eda_{name.lower()}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = ProgressClient(notebook, km=manager, timeout=timeout, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    started = time.monotonic()
    print(f"Executing {path.name}", flush=True)
    def progress(cell_index, **kwargs):
        print(f"  cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    client.on_cell_start = progress
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    temporary = path.with_suffix(".ipynb.part")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(f"PASS {path.name}: {time.monotonic() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["OpenSSH", "Zookeeper", "Mac", "Thunderbird"])
    args = parser.parse_args()
    allowed = {"openssh", "zookeeper", "mac", "thunderbird"}
    for name in args.datasets:
        if name.lower() not in allowed:
            parser.error(f"Unknown dataset: {name}")
        execute(name)


if __name__ == "__main__":
    main()
