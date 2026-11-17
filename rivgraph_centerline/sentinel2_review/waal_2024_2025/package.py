"""Package the verified review products without the large per-date cache."""
import json,shutil,zipfile
from pathlib import Path
from fetch import ROOT
OUT=ROOT/'review'

def main():
    assert(OUT/'verification.json').exists() and(OUT/'Waal_review.qgz').exists()
    with zipfile.ZipFile(OUT/'Waal_review.qgz') as z:
        qgs=next(n for n in z.namelist() if n.endswith('.qgs'));text=z.read(qgs).decode()
        assert str(ROOT) not in text,'Project contains non-portable absolute data paths'
    shutil.copy2(ROOT/'README.md',OUT/'README.md')
    repro=OUT/'reproduce';repro.mkdir(exist_ok=True)
    for name in ['fetch.py','build.py','qgis_project.py','test_build.py','verify.py','package.py','config.json','catalog.json','inventory.json']:
        shutil.copy2(ROOT/name,repro/name)
    (OUT/'HOW_TO_REBUILD.txt').write_text('Code, config, source catalog and date inventory are in reproduce/. Run fetch.py there to retrieve the public imagery, followed by build.py. The original per-date image cache is retained in the workspace, but excluded from this portable review archive. No calibrated Amazon classifier is included or implied.\n')
    archive=ROOT/'Waal_2024_2025_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,Path('Waal_2024_2025_review')/p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('Verified portable package:',archive,'bytes:',archive.stat().st_size)

if __name__=='__main__':main()
