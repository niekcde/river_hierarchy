"""Build the scientific concept inventory with the bundled document runtime."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[3]
data=json.loads((ROOT/'synthetic_experiments/notes/geometry_concept_inventory_20261102.json').read_text())
OUT=ROOT/'synthetic_experiments/notes/geometry_concept_inventory_20261102.docx'
doc=Document();sec=doc.sections[0];sec.page_width=Inches(11.69);sec.page_height=Inches(8.27)
sec.top_margin=sec.bottom_margin=Inches(.6);sec.left_margin=sec.right_margin=Inches(.65)
sec.header_distance=sec.footer_distance=Inches(.25)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
 style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
normal=doc.styles['Normal'];normal.font.size=Pt(10.5);normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.08
for name,size in [('Title',23),('Heading 1',17),('Heading 2',13)]:doc.styles[name].font.size=Pt(size)
for style in doc.styles:
 if style.element.pPr is not None:
  for border in list(style.element.pPr.findall(qn('w:pBdr'))):style.element.pPr.remove(border)
doc.styles['Heading 2'].paragraph_format.space_before=Pt(7)
doc.styles['Heading 2'].paragraph_format.space_after=Pt(3)
doc.core_properties.title='Geometric concepts behind the synthetic predictor inventory'
doc.core_properties.subject='An exhaustive middle-level taxonomy of 205 candidate representations'
doc.core_properties.author='River hierarchy project'
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run('Geometry concept inventory  |  ');r.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)

def p(text,style=None):return doc.add_paragraph(text,style)
def table(headers,widths):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for c,w,h in zip(t.rows[0].cells,widths,headers):c.width=Inches(w);c.text=h
 for c,w in zip(t.columns,widths):c.width=Inches(w)
 pr=t._tbl.tblPr;b=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
 pr.append(b);rep=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rep)
 return t

def style_table(t,font=10):
 for ri,row in enumerate(t.rows):
  cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
  for ci,c in enumerate(row.cells):
   c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=c._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
   for edge,size in [('top',90),('bottom',90),('left',100),('right',100)]:
    v=OxmlElement('w:'+edge);v.set(qn('w:w'),str(size));v.set(qn('w:type'),'dxa');m.append(v)
   pr.append(m);shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DCE6F1' if ri==0 else 'F5F7FA' if ri%2==0 else 'FFFFFF');pr.append(shade)
   for para in c.paragraphs:
    para.paragraph_format.space_after=Pt(2);para.paragraph_format.space_before=Pt(1);para.paragraph_format.line_spacing=1.05
    if ci==0 or (len(row.cells)==5 and ci==2):para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in para.runs:r.font.size=Pt(font);r.bold=ri==0;r.font.color.rgb=RGBColor(0,0,0)

def equation(cell,text):
 para=cell.paragraphs[0]
 r=para.add_run(text);r.font.name='Cambria Math';r.font.size=Pt(10)

doc.add_paragraph('Geometric concepts behind the synthetic predictor inventory','Title')
p('Full middle layer classification  |  2 November 2026','Subtitle')
p('The 205 candidate columns can be organised into 21 geometric concepts. Each concept describes a geometric object or mechanism; its candidates are statistical summaries, parent–child comparisons, transformations or duplicate implementations of that concept. This table supplies the missing link between the four geometric ingredients and the expanded predictor inventory.')

p('The 21 groups are an interpretive taxonomy, not 21 independent physical quantities or a new predictor selection. Every column is assigned exactly once, but the concepts can overlap mathematically. For example, the branch-contrast variance sum equals allocated route-length variance. Different weighting schemes remain explicit.')
doc.add_heading('How to read the counts',1)
p('The count is the number of columns actually included in the frozen 205-candidate screen, including partially defined columns and duplicates. It is not the number of possible formulas or independent predictors. Standard forms are parent xₚ, child x꜀, difference x꜀ − xₚ and ratio x꜀/xₚ. Not every descriptor has all four forms.')
p('Example: allocated route length has six summaries—mean, SD, CV, minimum, maximum and skewness. The first five have four comparison forms and skewness has three: 23 columns. Five additional columns (two log1p ratios, a kilometre conversion, a duplicate spread difference and a normalised spread difference) bring this concept to 28 columns.')
doc.add_heading('Notation and scope',1)
nt=table(['Symbol','Meaning'],[1.45,8.94])
for a,b in [('L, B, S, C','Reach length, reach width, reach slope and directed network connectivity. U denotes the selected source-unit membership.'),('e, v, j, r','Reach, vertex, outgoing branch and complete source-to-sink route. ℬ is the set of bifurcation vertices.'),('aᵥⱼ, pᵥ, fₑ','Local outgoing width fraction, vertex visit probability (inlet p = 1) and reach allocation fraction. At a split, fᵥⱼ = pᵥaᵥⱼ; confluences add incoming fractions.'),('πᵣ, N, qᵣ','πᵣ is the product of local split fractions along route r; N is the number of routes; qᵣ is a separate harmonic-path-width weight.'),('Domain','Most quantities use the closed outer B domain. Affected extent uses the selected source unit U. Manning ΔG uses whole-network sums.')]:
 cells=nt.add_row().cells;cells[0].text=a;cells[1].text=b
style_table(nt)
doc.add_page_break();doc.add_heading('Concept inventory',1)
p('Core equations define the concept; the appendix identifies the summaries and comparison forms counted under it. C is connectivity, not a scalar measurement. A mean or SD is a summary within a concept, not a new ingredient.')
t=table(['ID','Geometric concept','Count','Ingredients','Core equation'],[.4,2.55,.55,1.25,5.64])
for c in data['concepts']:
 cells=t.add_row().cells
 for i,text in [(0,str(c['id'])),(1,c['name']),(2,str(c['count'])),(3,c['ingredients'])]:cells[i].text=text
 equation(cells[4],c['equation'])
cells=t.add_row().cells;cells[1].text='Total candidate columns';cells[2].text='205';cells[4].text='21 concepts; 66 catalogue metric labels'
style_table(t,10)
doc.add_paragraph().paragraph_format.space_after=Pt(2)
p('Grouping choice: topology counts and their compound score are grouped as architecture; local dominance, effective branch number and normalised entropy are grouped as summaries of the same bifurcation allocation distribution. These are organisational choices, not assertions that the resulting statistics are interchangeable.')
doc.add_page_break();doc.add_heading('Derivations and weighting conventions',1)
p('Each entry below lists the catalogue metric labels and their column counts. Parent, child, difference and ratio are included where present. Transformed or duplicated columns remain in the count so that it reconciles exactly to the original screen.')
for c in data['concepts']:
 doc.add_heading(f"{c['id']}  {c['name']}  {c['count']} {'column' if c['count']==1 else 'columns'}",2)
 note=p(c['note']);note.paragraph_format.space_after=Pt(4)
 pp=p('; '.join(m['metric']+' ('+str(m['count'])+')' for m in c['metrics'])+'.')
 pp.paragraph_format.space_after=Pt(3)
 for r in pp.runs:r.font.size=Pt(9);r.font.color.rgb=RGBColor.from_string('444444')
doc.add_heading('Interpretation limits and source record',1)
p('This inventory does not rerun the experiments, alter equations, change the 19-candidate shortlist or remove any candidate. It separates the concept being measured from its statistical summary and parent–child representation. Undefined ratios remain undefined; a valid zero spread or zero entropy is not replaced by a missing value.')
p('Scope and weighting are essential. Harmonic path-width weights qᵣ differ from routed allocation πᵣ; topology density also contains a length normalisation; slope summaries may be coupled to length by the conserved-head-loss construction. Head loss ΔH should not be confused with a child-minus-parent difference of that head loss.')
p('The active catalogue is broad_descriptors_20261030/geometry_with_manning/candidate_catalog.csv on the SyntheticResults20261027 results volume. The mapping was checked against all 205 column names and all 66 metric labels, with no missing or multiply assigned columns. It is stored beside this document as geometry_concept_inventory_20261102.json, including the exact column membership and source SHA-256 digest.')
p('Definition sources: RAPID/investigations/parent_child_predictor_exploration/features.py; RAPID/investigations/hydrograph_predictors_20261014/build.py; synthetic_experiments/investigations/geometry_predictor_screen.py; broad_descriptor_audit.py; current_predictor_bridge.py; and hierarchy_level_definition/metrics/unit_metrics.py. Reference: SYN-GEOMETRIC-CONCEPT-INVENTORY-2026-11-02.')
# Avoid isolated headings. Keep paragraphs under a heading together only naturally;
# table rows are indivisible and table headers repeat across pages.
doc.save(OUT);print(OUT)
