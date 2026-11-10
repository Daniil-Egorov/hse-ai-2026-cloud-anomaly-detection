"""Execute the EDA notebook using the Python environment invoking this module.

Usage from the project root: python -m util.run_eda_romailika
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout per cell in seconds')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks' / 'eda_romailika_loghub.ipynb'
    notebook = nbformat.read(path, as_version=4)
    os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
    with tempfile.TemporaryDirectory(prefix='hse-eda-') as temporary:
        directory = Path(temporary) / 'eda-romailika'
        directory.mkdir()
        (directory / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'EDA RoMaIlIka', 'language': 'python',
        }), encoding='utf-8')
        manager = KernelManager(kernel_name='eda-romailika',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))

        def progress(cell, cell_index, **kwargs):
            print(f'Completed cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

        client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                resources={'metadata': {'path': str(root)}}, on_cell_executed=progress)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
    print(f'Saved executed notebook: {path}')


if __name__ == '__main__':
    main()
