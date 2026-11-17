"""Verified, resumable storage-only migration into a mounted APFS container.
Deletes each source file only after durable copy, SHA256 readback verification,
and a flushed audit record. Does not alter model files or scientific metadata.
"""
import hashlib,json,os,shutil,time
from pathlib import Path
SRC=Path('/Volumes/PhD/river_hierarchy/synthetic_experiments/outputs/depth_methods_factorial_parallel_20261027')
DST=Path('/Volumes/SyntheticResults20261027')
if not DST.is_mount():raise RuntimeError('Destination is not a mounted filesystem')
if SRC.stat().st_dev==DST.stat().st_dev:raise RuntimeError('Destination must be the mounted container')
if os.statvfs(DST).f_frsize>=524288:raise RuntimeError('Destination allocation unit has not improved')
LOG=DST/'storage_migration_sha256.jsonl'
LOCAL=Path(__file__).resolve().parents[1]/'outputs/storage_audit_20261027/migration_progress.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
count=0;logical=0;reclaimed=0;started=time.time();batch=[]
def move_batch(batch):
 global count,logical,reclaimed
 copies=[]
 for src in batch:
  rel=src.relative_to(SRC);dst=DST/rel;dst.parent.mkdir(parents=True,exist_ok=True);before=src.stat()
  expected=sha(src)
  if not dst.exists():
   tmp=dst.with_name(dst.name+'.migrating')
   shutil.copyfile(src,tmp);os.replace(tmp,dst)
  copies.append((src,dst,before,expected))
 os.sync()
 # Verify every byte in this batch before removing any source in it.
 for src,dst,before,expected in copies:
  now=src.stat()
  if (now.st_size,now.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise RuntimeError(f'Source changed: {src}')
  if dst.stat().st_size!=before.st_size or sha(dst)!=expected:raise RuntimeError(f'Copy verification failed: {src}')
 with LOG.open('a') as log:
  for src,dst,before,expected in copies:
   log.write(json.dumps(dict(path=str(src.relative_to(SRC)),bytes=before.st_size,sha256=expected))+'\n')
  log.flush();os.fsync(log.fileno())
 for src,dst,before,expected in copies:
  src.unlink();count+=1;logical+=before.st_size;reclaimed+=before.st_blocks*512
 progress=dict(files_verified_and_moved_this_invocation=count,logical_bytes=logical,source_file_allocation_reclaimed=reclaimed,elapsed_seconds=time.time()-started,phd_free_bytes=shutil.disk_usage(SRC).free)
 LOCAL.write_text(json.dumps(progress,indent=2))
 if count%4096==0:print(json.dumps(progress),flush=True)
# Cache first: it reclaims the most allocation per byte copied.
for root in [SRC/'state_cache',SRC/'groups']:
 for folder,dirs,files in os.walk(root):
  for name in files:
   batch.append(Path(folder)/name)
   if len(batch)>=256:move_batch(batch);batch=[]
for p in SRC.iterdir():
 if p.is_file():batch.append(p)
if batch:move_batch(batch)
# Remove only empty directories; never recursively delete unchecked content.
for folder,dirs,files in os.walk(SRC,topdown=False):
 if Path(folder)!=SRC:Path(folder).rmdir()
if list(SRC.iterdir()):raise RuntimeError('Unmigrated source entries remain')
os.sync()
print('MIGRATION COMPLETE',count,'files verified and moved',flush=True)
