"""Use the validated fixed-K screen with the reconstructed child contexts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import route_full_signed_validation as routing
from reconstruct_child_heads import OUT, EXAMPLES
routing.OUT=OUT

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(routing.run,sys.argv[1:] or EXAMPLES))
