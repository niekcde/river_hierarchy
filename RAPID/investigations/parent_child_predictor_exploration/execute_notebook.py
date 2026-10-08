"""Execute notebook sequentially in IPython without opening kernel socket ports.

This uses the same Python environment and captures real stdout, tables and inline
figures into standard notebook outputs. Any failed cell stops execution.
"""
from pathlib import Path
import sys,traceback
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from matplotlib_inline.backend_inline import configure_inline_support

p=Path(sys.argv[1]);nb=nbformat.read(p,as_version=4);shell=InteractiveShell.instance()
configure_inline_support(shell,'inline');count=0
for i,cell in enumerate(nb.cells):
    if cell.cell_type!='code':continue
    count+=1;cell.execution_count=count;cell.outputs=[]
    print('Executing cell',i+1,'/',len(nb.cells),flush=True)
    with capture_output(stdout=True,stderr=True,display=True) as captured:
        result=shell.run_cell(cell.source,store_history=True)
    if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    error=result.error_before_exec or result.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exception(error)))
        nbformat.write(nb,p);raise RuntimeError(f'Notebook failed at cell {i+1}: {error}') from error
    nbformat.write(nb,p)
nb.metadata['execution_validation']={'engine':'sequential in-process IPython; inline matplotlib','python':sys.version,'cells_executed':count,'failed_cells':0}
nbformat.validate(nb);nbformat.write(nb,p)
print('Executed',count,'code cells:',p)
