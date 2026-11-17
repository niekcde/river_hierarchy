"""Create a portable, styled QGIS review project using bundled PyQGIS."""
import os,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from qgis.core import (QgsApplication,QgsProject,QgsRasterLayer,QgsVectorLayer,QgsCoordinateReferenceSystem,QgsRectangle,QgsReferencedRectangle,QgsPalettedRasterRenderer,QgsColorRampShader,QgsRasterShader,QgsSingleBandPseudoColorRenderer,QgsFillSymbol,QgsSingleSymbolRenderer,QgsCategorizedSymbolRenderer,QgsRendererCategory,QgsEditorWidgetSetup)
from qgis.PyQt.QtGui import QColor
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'review'
app=QgsApplication([],False);app.initQgis()
p=QgsProject.instance();crs=QgsCoordinateReferenceSystem('EPSG:32631');assert crs.isValid();p.setCrs(crs)
p.setTitle('Waal · 2024–2025 · Single-channel mask review · PENDING')
p.setFileName(str(OUT/'Waal_review.qgz'));p.writeEntry('Paths','/Absolute',False)

def add_raster(name,title,kind=None,visible=False):
    layer=QgsRasterLayer(str(OUT/(name+'.tif')),title);assert layer.isValid(),name
    if kind=='mask':
        classes=[QgsPalettedRasterRenderer.Class(0,QColor(0,0,0,0),'Land'),QgsPalettedRasterRenderer.Class(1,QColor(0,185,255,120),'Candidate water')]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(),1,classes))
    elif kind=='excluded':
        classes=[QgsPalettedRasterRenderer.Class(0,QColor(0,0,0,0),'Other'),QgsPalettedRasterRenderer.Class(1,QColor(230,50,150,170),'Excluded component (retained)')]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(),1,classes))
    elif kind=='occurrence':
        fn=QgsColorRampShader(0,1);fn.setColorRampType(QgsColorRampShader.Interpolated)
        fn.setColorRampItemList([QgsColorRampShader.ColorRampItem(0,QColor('#f7fbff'),'0'),QgsColorRampShader.ColorRampItem(.5,QColor('#6baed6'),'0.5'),QgsColorRampShader.ColorRampItem(1,QColor('#08306b'),'1')])
        shader=QgsRasterShader();shader.setRasterShaderFunction(fn);layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(),1,shader))
    p.addMapLayer(layer);p.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(visible)
    layer.saveNamedStyle(str(OUT/(name+'.qml')))
    return layer

add_raster('rgb_median_display','01 · Two-year median RGB · 2024–2025',visible=True)
for f in sorted(OUT.glob('dated_RGB_*.tif')):add_raster(f.stem,'Dated RGB · '+f.stem.replace('dated_RGB_',''))
add_raster('valid_observation_count','Usable observations · 2024–2025')
add_raster('mean_mndwi_NOT_probability','Mean MNDWI · index, NOT probability')
for year in ['2024','2025']:add_raster('spectral_water_occurrence_'+year,'Spectral occurrence · '+year,'occurrence')
add_raster('scl_water_occurrence','SCL water occurrence · independent classification baseline','occurrence')
add_raster('spectral_water_occurrence','Spectral-rule water occurrence · 2024–2025','occurrence')
add_raster('excluded_components_retained','Excluded components · retained for inspection','excluded')
add_raster('water_mask_occ75_largest_component','Optional · Largest component of 75% occurrence mask','mask')
add_raster('water_mask_occ75_all','02 · Candidate mask · occurrence ≥75% · all components','mask',True)

for name,title,color in [('uncertain_areas','Uncertain areas · excluded from future training','#f28e2b'),('manual_edits','Manual edits · add / remove','#e15759'),('review_tiles','03 · Review window · PENDING','#ffb000')]:
    l=QgsVectorLayer(str(OUT/'review_annotations.gpkg')+'|layername='+name,title,'ogr');assert l.isValid(),name
    symbol=QgsFillSymbol.createSimple({'style':'no','outline_color':color,'outline_width':'0.6'})
    l.setRenderer(QgsSingleSymbolRenderer(symbol))
    if name=='manual_edits':
        categories=[]
        for action,col in [('add','#00bfff'),('remove','#ff5c35')]:
            sym=QgsFillSymbol.createSimple({'color':col,'outline_color':col,'outline_width':'.4'});sym.setOpacity(.4)
            categories.append(QgsRendererCategory(action,sym,action))
        l.setRenderer(QgsCategorizedSymbolRenderer('action',categories))
        for field,values in [('action',['add','remove']),('reason',['bank','bar_island','channel_connection','river_selection']),('confidence',['confident','uncertain'])]:
            l.setEditorWidgetSetup(l.fields().indexFromName(field),QgsEditorWidgetSetup('ValueMap',{'map':[{v:v} for v in values]}))
    if name=='review_tiles':
        l.setEditorWidgetSetup(l.fields().indexFromName('status'),QgsEditorWidgetSetup('ValueMap',{'map':[{v:v} for v in ['pending','accepted','needs_edits']]}))
        extent=l.extent();extent.scale(1.2)
    p.addMapLayer(l)
    l.saveNamedStyle(str(OUT/(name+'.qml')))
p.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent,crs))
assert p.write();n=len(p.mapLayers());p.clear();assert p.read(str(OUT/'Waal_review.qgz'))
assert len(p.mapLayers())==n and all(l.isValid() for l in p.mapLayers().values())
print('QGIS project verified:',n,'valid layers',OUT/'Waal_review.qgz')
p.clear();app.exitQgis()
