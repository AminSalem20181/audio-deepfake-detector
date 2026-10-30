"""Package source only, excluding datasets, existing outputs and credentials."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
output=root/'sde-seg-colab.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ('sdeseg','scripts','tests','configs'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                z.write(path,Path('sde-seg')/path.relative_to(root))
    for name in ('requirements.txt','requirements-data.txt','requirements-lock-macos.txt','README.md','REPORT.md','colab_run.ipynb','SDE-Seg-final.ipynb','LICENSE'):
        if (root/name).exists():
            z.write(root/name,Path('sde-seg')/name)
print(output)
