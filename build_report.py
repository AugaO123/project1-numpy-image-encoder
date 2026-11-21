from pathlib import Path
import math, re, shutil, zipfile
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Polygon,Circle
from reportlab.graphics import renderPDF
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'report_output'; FIG=OUT/'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
BLUE='#173C61'; TEAL='#007F86'; RED='#BB313F'; INK='#223344'; GREY='#5D6B79'; LIGHT='#EDF3F8'
def txt(d,x,y,s,size=9,color=INK,font='DV',anchor='middle'):
 d.add(String(x,y,s,fontName=font,fontSize=size,fillColor=colors.HexColor(color),textAnchor=anchor))
def box(d,x,y,w,h,label,sub='',red=False):
 col=RED if red else BLUE
 d.add(Rect(x,y,w,h,rx=5,ry=5,strokeColor=colors.HexColor(col),fillColor=colors.HexColor('#FFF1F1' if red else LIGHT),strokeWidth=1.1))
 txt(d,x+w/2,y+h/2+(4 if sub else -3),label,9,col,'DVB')
 if sub: txt(d,x+w/2,y+h/2-11,sub,7.8,GREY)
def arrow(d,x1,y1,x2,y2,color=GREY):
 col=colors.HexColor(color);d.add(Line(x1,y1,x2,y2,strokeColor=col,strokeWidth=1))
 a=math.atan2(y2-y1,x2-x1); r=4
 d.add(Polygon([x2,y2,x2-r*math.cos(a-.55),y2-r*math.sin(a-.55),x2-r*math.cos(a+.55),y2-r*math.sin(a+.55)],fillColor=col,strokeColor=col))
FIGURES={}
def save_fig(name,d):
 FIGURES[name]=d
 renderPDF.drawToFile(d,str(FIG/(name+'.pdf')))
# 1. Image tower, including the pre-Norm and final head.
d=Drawing(480,240)
for j in range(7):
 for i in range(7):
  d.add(Rect(12+i*11,143+(6-j)*11,10,10,fillColor=colors.HexColor(['#A8CDD9','#D4E5F0','#8BB2C8'][(i+j)%3]),strokeColor=colors.white,strokeWidth=.4))
txt(d,50,128,'224 × 224 RGB',8);txt(d,50,115,'7 × 7 patch grid',8)
box(d,119,163,143,48,'01  Patchify','[B,49,3072]',True);arrow(d,91,181,116,181)
box(d,300,163,166,48,'05  Embed + positions','49 patches + 1 CLS = 50',True);arrow(d,265,187,296,187)
box(d,300,89,166,44,'Pre-Norm','normalize token features');arrow(d,383,161,383,136)
box(d,119,89,143,44,'02–04  Transformer ×12','[B,50,768]',True);arrow(d,297,111,265,111)
box(d,12,17,192,43,'06  CLS + post-Norm','[B,768]',True);arrow(d,190,86,190,67);arrow(d,190,67,108,67);arrow(d,108,67,108,62)
box(d,264,17,202,43,'06  Project + unit norm','[B,512] • one vector / image',True);arrow(d,207,38,260,38)
save_fig('01_pipeline',d)
# 2. Axis order and multi-head reshape.
d=Drawing(480,193)
txt(d,8,179,'PATCH ORDER: row-major grid; C,H,W inside each patch',9,BLUE,'DVB','start')
box(d,8,117,133,40,'Input','[B,3,224,224]');box(d,172,117,146,40,'Separate grid axes','[B,3,7,32,7,32]',True);box(d,349,117,123,40,'Transpose + flatten','[B,49,3072]',True)
arrow(d,144,137,168,137);arrow(d,321,137,345,137)
txt(d,240,99,'transpose(0,2,4,1,3,5): grid row, grid column, C, H, W',8,GREY)
d.add(Line(8,85,472,85,strokeColor=colors.HexColor('#CCD8E2')))
txt(d,8,69,'HEAD ORGANIZATION: keep batch and token order',9,BLUE,'DVB','start')
box(d,8,10,133,40,'Q / K / V','[B,N,768]');box(d,172,10,146,40,'Split feature dimension','[B,N,12,64]',True);box(d,349,10,123,40,'Move head axis','[B,12,N,64]',True)
arrow(d,144,30,168,30);arrow(d,321,30,345,30)
save_fig('02_axes',d)
# 3. Two residual paths; full data direction with branches.
d=Drawing(480,262)
box(d,115,219,215,34,'Input x','[B,50,768]')
box(d,115,169,215,33,'Norm 1 → attention','QKᵀ / sqrt(64) → softmax → V',True);arrow(d,222,218,222,204)
d.add(Circle(222,147,9,fillColor=colors.white,strokeColor=colors.HexColor(RED)));txt(d,222,143,'+',12,RED,'DVB');arrow(d,222,168,222,158)
d.add(Line(115,235,65,235,strokeColor=colors.HexColor(TEAL)));d.add(Line(65,235,65,147,strokeColor=colors.HexColor(TEAL)));arrow(d,65,147,211,147,TEAL);txt(d,23,191,'skip x',8,TEAL,anchor='start')
box(d,115,89,215,34,'Norm 2 → MLP','768 → 3072 → GELU → 768',True);arrow(d,222,137,222,125)
d.add(Circle(222,66,9,fillColor=colors.white,strokeColor=colors.HexColor(RED)));txt(d,222,62,'+',12,RED,'DVB');arrow(d,222,88,222,77)
d.add(Line(239,147,381,147,strokeColor=colors.HexColor(TEAL)));d.add(Line(381,147,381,66,strokeColor=colors.HexColor(TEAL)));arrow(d,381,66,233,66,TEAL);txt(d,386,110,'skip r',8,TEAL,anchor='start')
box(d,115,5,215,35,'Output r + mlp_update','[B,50,768]');arrow(d,222,56,222,42)
save_fig('03_residual',d)
# Correct unicode superscript which DV supports but diagram style use T.
# 4. Error chart uses measured numbers, not visualized tolerances.
d=Drawing(480,126)
names=['tokens','block0','last_tokens','output','batch'];vals=[4.387e-5,7.147e-6,4.196e-5,5.662e-7,4.061e-7]
for k in [-7,-6,-5,-4]:
 x=104+(k+7)*100;d.add(Line(x,23,x,116,strokeColor=colors.HexColor('#D5E0E9')));txt(d,x,7,'1e'+str(k),8,GREY)
for i,(name,val) in enumerate(zip(names,vals)):
 y=106-i*19;txt(d,94,y-3,name,8,GREY,anchor='end');x=104+(math.log10(val)+7)*100
 d.add(Line(104,y,x,y,strokeColor=colors.HexColor('#8FAFC0'),strokeWidth=3));d.add(Circle(x,y,3.4,fillColor=colors.HexColor(TEAL),strokeColor=None));txt(d,x+8,y-3,f'{val:.3e}',7,TEAL,anchor='start')
save_fig('04_error_chart',d)
PAGES=[
('Objectives and Overview',[
 ('p','The goal is to reproduce the supplied OpenCLIP ViT-B/32 image encoder using NumPy. Given fixed learned visual parameters, the encoder maps preprocessed RGB images to normalized 512-dimensional feature vectors. The task is forward inference: no training, loss function, gradient calculation, or parameter update is required.'),
 ('h','1.1 Input, architecture and output contract'),
 ('p','The input is float32 [B,3,224,224], where B is the number of images. A 32×32 patch size produces a 7×7 grid, hence 49 patch rows. The learned CLS token increases the sequence length to 50. Twelve Transformer blocks preserve [B,50,768]; the final head selects CLS and returns [B,512]. Each output row has unit Euclidean length.'),
 ('fig',('01_pipeline',240,'Figure 1. Image-encoder data flow. Red boxes identify the six implemented TODO groups; neutral boxes show supplied operations.')),
 ('h','1.2 Scope and implementation boundaries'),
 ('p','<font color="#BB313F"><b>Implemented:</b></font> patchification, head splitting, attention calculation and merging, both residual additions, the MLP, embedding, and CLS output projection. The supplied preprocessing, LayerNorm, exact GELU, attention-weight conversion and unit-vector helpers remain in use. Weight names, tensor values and public class interfaces are preserved.'),
 ('p','The course configuration fixes width=768, heads=12, head_dim=64, MLP width=3072 and output_dim=512. Changing configuration values alone cannot produce a different valid checkpoint. The reference archive is used only by check.py; the encoder never reads cached reference rows.')]),
('Implementations: patches and array organization',[
 ('p','The first challenge is preserving element order while changing shape. Reshape groups existing elements; transpose reorders axes. Matching only the final shape is insufficient because an incorrect patch or head order changes every later calculation.'),
 ('fig',('02_axes',193,'Figure 2. Two assessed axis transformations. Batch B is preserved; N denotes token count and is 50 in the image tower.')),
 ('h','2.1 Patchify and embedding (TODO-01 / TODO-05)'),
 ('p','For each image, separate height and width into grid and within-patch axes: [B,3,7,32,7,32]. Reorder to [B,7,7,3,32,32] before flattening. Patches are visited left to right, then top to bottom; inside each patch the flattening order is C,H,W. Each row has 3×32×32=3072 elements.'),
 ('code','patches = images.reshape(B,3,7,32,7,32)\npatches = patches.transpose(0,2,4,1,3,5).reshape(B,49,3072)\npatch_tokens = patches @ self.patch_weight.T\ntokens = np.concatenate([cls, patch_tokens], axis=1)\nreturn tokens + self.position'),
 ('p','The patch projection has no bias and maps 3072 features to 768. CLS is repeated once per image and prepended along axis 1. The position array [50,768] broadcasts across B, assigning a learned position to CLS and each patch. The supplied pre-Norm then prepares the sequence for the first block.'),
 ('h','2.2 Split heads and transpose K (TODO-02)'),
 ('p','Each Q, K and V tensor [B,N,768] is reshaped to [B,N,12,64], then transposed to [B,12,N,64]. The last axis carries head features. For query-key comparisons, only K’s final two axes are swapped: k.swapaxes(-1,-2). Using k.T on a four-dimensional array would reverse all axes and mix the batch/head organization.'),
 ('p','<font color="#BB313F"><b>Shape invariant:</b></font> Q @ Kᵀ yields [B,12,N,N]. Each row compares one query token with all N key tokens within the same image and attention head.')]),
('Implementations: attention, residuals and output',[
 ('fig',('03_residual',245,'Figure 3. Pre-Norm Transformer block. Teal paths retain the input and intermediate residual; red operations are implemented in the student encoder.')),
 ('h','3.1 Scaled dot-product attention (TODO-03)'),
 ('p','Compute scores=(Q @ Kᵀ)/sqrt(64), then apply the supplied softmax helper over the final key axis. Multiplying these coefficients by V yields context [B,12,N,64]. Transpose to [B,N,12,64] and reshape to [B,N,768] before applying the attention output projection. The image path uses causal=False; the supplied optional causal switch is preserved for later text-tower compatibility.'),
 ('h','3.2 Residual and MLP calculation (TODO-04)'),
 ('code','residual = x + self.attention(self.norm1(x), causal=causal)\nu = self.norm2(residual)\nhidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)\nmlp_update = hidden @ self.fc2_weight.T + self.fc2_bias\nreturn residual + mlp_update'),
 ('p','Both residual additions retain [B,N,768]. The MLP expands the feature axis from 768 to 3072, applies exact GELU, and projects back to 768. Each of the 12 blocks uses its own supplied parameters; the blocks do not share learned weights.'),
 ('h','3.3 CLS and normalized output (TODO-06)'),
 ('p','Select tokens[:,0,:], apply post_norm, then multiply by visual.proj [768,512]. This projection is already stored as [in,out], so it must not be transposed. Finally unit_vector scales each image row independently. Attention/MLP weights instead use [out,in] and require .T. These distinct layouts are a critical implementation convention.')]),
('Results and Discussion',[
 ('p','The supplied checker was rerun for the revised report with python check.py --stage all. Asset integrity and every numerical stage passed. The table reports measured absolute errors, not estimated or fabricated values.'),
 ('table',None),
 ('fig',('04_error_chart',112,'Figure 4. Maximum absolute error on a logarithmic axis. Exact-zero preprocessing and patch errors are omitted from this log plot.')),
 ('h','4.1 Validation criteria and debugging'),
 ('p','The checker requires matching shapes, finite float32 arrays, and elementwise error ≤ atol + rtol×|reference|, with rtol=2e-4. The absolute tolerances are 2e-6 for preprocessing, 2e-4 for internal stages, and 2e-5 for output/batch. It also checks unit-length output rows. Batch order [1,0,1] tests both reordering and duplicate images.'),
 ('p','<font color="#BB313F"><b>Observed bug and fix:</b></font> the first implementation divided attention scores by a NumPy float64 square root. In the tested environment this promoted subsequent tensors and caused the checker to reject their dtype. Casting the scale to np.float32(np.sqrt(self.head_dim)) fixed the dtype failure; all stages then passed. Other risks include a wrong patch transpose, omitted residual addition, and transposing visual.proj.'),
 ('h','4.2 Reproduction, limits and assistance'),
 ('p','Install requirements.txt, then copy the completed image_encoder.py into the original course project folder. Keep its weights/, assets/ and manifest files intact. Run python check.py --stage all; run python run.py --output output_tokens.npz to save image IDs and features. Large course weights and reference data are intentionally excluded from the submission. These tests establish agreement on the supplied images and reordered batch; they do not measure recognition accuracy or certify all possible inputs.'),
 ('p','<b>Assistance disclosure:</b> ChatGPT assisted with the NumPy implementation, debugging, validation commands, diagrams and report drafting. Supplied course helpers, weights and reference data were retained. Repository: https://github.com/Auga0123/project1-numpy-image-encoder')])]
ROWS=[['Stage','Shape','Max abs.','Mean abs.','Status'],['preprocess','(10,3,224,224)','0','0','PASS'],['patches','(2,49,3072)','0','0','PASS'],['tokens','(2,50,768)','4.387e-5','1.335e-6','PASS'],['block0','(2,50,768)','7.147e-6','3.348e-7','PASS'],['last_tokens','(2,50,768)','4.196e-5','2.633e-6','PASS'],['output','(10,512)','5.662e-7','8.688e-8','PASS'],['batch','(3,512)','4.061e-7','7.848e-8','PASS']]
# Render a ready-to-submit PDF; page content shares its source with LaTeX.
W,H=A4;c=canvas.Canvas(str(OUT/'Project1_Report_JinJiayuan_2025221025.pdf'),pagesize=A4)
c.setTitle('Project 1 - NumPy Image Encoder - 金佳圆 2025221025');c.setAuthor('金佳圆')
c.setFillColor(colors.HexColor(BLUE));c.setFont('STSong-Light',27);c.drawCentredString(W/2,H-145,'深 圳 大 学 实 验 报 告')
cover=[('课程名称','Python程序设计'),('项目名称','NumPy 图像编码器'),('学 院','人工智能学院'),('指导教师','樊超'),('报 告 人','金佳圆'),('学 号','2025221025'),('实验时间','2026.11.19'),('提交时间','2026.11.22')]
for i,(a,b) in enumerate(cover):
 y=H-230-i*44;c.setFont('STSong-Light',15);c.setFillColor(colors.black);c.drawString(90,y,a);c.drawCentredString(357,y,b);c.setStrokeColor(colors.HexColor('#AAB5BF'));c.line(215,y-7,500,y-7)
c.setFont('STSong-Light',15);c.drawCentredString(W/2,85,'教务处制');c.showPage()
style=ParagraphStyle('body',fontName='DV',fontSize=9.3,leading=13.8,textColor=colors.HexColor(INK),spaceAfter=8)
capstyle=ParagraphStyle('caption',fontName='DV',fontSize=7.7,leading=10,textColor=colors.HexColor(GREY))
for n,(title,items) in enumerate(PAGES,1):
 x=54;y=H-48;w=W-108
 c.setStrokeColor(colors.HexColor('#C7D5E1'));c.line(x,y,W-x,y)
 c.setFont('DV',8);c.setFillColor(colors.HexColor(GREY));c.drawString(x,y+9,'PYTHON PROGRAMMING  /  PROJECT 1');c.drawRightString(W-x,y+9,'金佳圆' if False else '2025221025')
 y-=32;c.setFillColor(colors.HexColor(BLUE));c.setFont('DVB',16);c.drawString(x,y,f'{n:02d}  {title}');y-=23
 for typ,val in items:
  if typ=='h':
   y-=6;c.setFont('DVB',10.6);c.setFillColor(colors.HexColor(BLUE));c.drawString(x,y,val);y-=18
  elif typ=='p':
   p=Paragraph(val,style);_,hh=p.wrap(w,1000);p.drawOn(c,x,y-hh);y-=hh+9
  elif typ=='fig':
   name,fh,caption=val
   import fitz
   # Draw vector source directly at the desired height.
   drawings={'01_pipeline':None}
   src=__import__('reportlab.graphics.renderPDF',fromlist=['x'])
   # Store drawings separately above through dictionary, loaded below.
   dr=FIGURES[name];scale=min(w/480,fh/dr.height)
   c.saveState();c.translate(x+(w-480*scale)/2,y-dr.height*scale);c.scale(scale,scale);renderPDF.draw(dr,c,0,0);c.restoreState();y-=dr.height*scale+6
   p=Paragraph(caption,capstyle);_,hh=p.wrap(w,1000);p.drawOn(c,x,y-hh);y-=hh+12
  elif typ=='code':
   lines=val.splitlines();hh=len(lines)*12+16;c.setFillColor(colors.HexColor('#F0F4F7'));c.roundRect(x,y-hh,w,hh,4,fill=1,stroke=0);c.setFillColor(colors.HexColor(RED));c.setFont('Courier',8.0)
   for i,l in enumerate(lines):c.drawString(x+9,y-14-i*12,l)
   y-=hh+10
  elif typ=='table':
   t=Table(ROWS,colWidths=[83,133,91,91,75],rowHeights=20)
   t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'DVB'),('FONTNAME',(0,1),(-1,-1),'DV'),('FONTSIZE',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)),('TEXTCOLOR',(0,0),(-1,0),colors.white),('TEXTCOLOR',(-1,1),(-1,-1),colors.HexColor(TEAL)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F4F7')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#AABBCC'))]))
   tw,th=t.wrap(w,1000);t.drawOn(c,x,y-th);y-=th+14
 print('PDF body page',n,'bottom',y)
 if y<43:raise RuntimeError('Page overflows')
 c.setFont('DV',8);c.setFillColor(colors.HexColor(GREY));c.drawCentredString(W/2,28,f'{n} / 4');c.showPage()
c.save()
# XeLaTeX source with the same prose, code, data and vector artwork.
def esc(t):
 for a,b in [('&','\\&'),('%','\\%'),('_','\\_'),('#','\\#')]:t=t.replace(a,b)
 t=t.replace('×',r'$\times$').replace('≤',r'$\leq$').replace('→',r'$\to$').replace('ᵀ',r'$^{T}$')
 t=re.sub(r'<font color="#BB313F"><b>(.*?)</b></font>',lambda m:r'\textcolor{red}{\textbf{'+m.group(1)+'}}',t)
 t=re.sub(r'<b>(.*?)</b>',lambda m:r'\textbf{'+m.group(1)+'}',t)
 return t
tex=r'''% !TeX program = xelatex
\documentclass[a4paper,11pt]{article}
\usepackage{fontspec,xeCJK,geometry,graphicx,xcolor,booktabs,fancyhdr,fvextra,hyperref}
\setCJKmainfont{FandolSong-Regular}
\geometry{left=20mm,right=20mm,top=18mm,bottom=18mm}
\definecolor{navy}{HTML}{173C61}
\hypersetup{colorlinks=true,urlcolor=navy}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Python Programming / Project 1}\fancyhead[R]{\small 2025221025}\fancyfoot[C]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
\begin{titlepage}\centering
\vspace*{25mm}{\Huge\bfseries 深\ 圳\ 大\ 学\ 实\ 验\ 报\ 告}\\[22mm]
'''
for a,b in cover:tex+=r'{\Large '+a+r'\quad\underline{\makebox[85mm][c]{'+b+r'}}}\\[9mm]'+'\n'
tex+=r'\vfill{\Large 教务处制}\end{titlepage}'+'\n'
for n,(title,items) in enumerate(PAGES,1):
 if n>1:tex+='\n\\newpage\n'
 tex+=r'\section*{'+str(n)+' '+title+'}\n'
 for typ,val in items:
  if typ=='p':tex+=esc(val)+'\n\n'
  elif typ=='h':tex+=r'\subsection*{'+esc(val)+'}\n'
  elif typ=='code':tex+=r'\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,frame=single]'+'\n'+val+'\n'+r'\end{Verbatim}'+'\n'
  elif typ=='fig':
   name,fh,caption=val;tex+=r'\begin{center}\includegraphics[width=\linewidth,height='+str(round(fh*.3528,1))+r'mm,keepaspectratio]{figures/'+name+r'.pdf}\\{\footnotesize '+esc(caption)+r'}\end{center}'+'\n'
  elif typ=='table':
   tex+=r'\begin{center}\small\begin{tabular}{lllll}\toprule'+'\n'
   for i,row in enumerate(ROWS):tex+=' & '.join(esc(z) for z in row)+r'\\'+'\n'+(r'\midrule'+'\n' if i==0 else '')
   tex+=r'\bottomrule\end{tabular}\end{center}'+'\n'
tex+='\n\\end{document}\n';(OUT/'report.tex').write_text(tex)
