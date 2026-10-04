(function(){
const cv=document.getElementById('stage'), ctx=cv.getContext('2d');
const scrub=document.getElementById('scrub'), playBtn=document.getElementById('play');
const icon=document.getElementById('icon'), clock=document.getElementById('clock');
const chips=[...document.querySelectorAll('.chip')];
const DUR=48000;
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
let VW=1000,VH=520,narrow=false,P=pal(),t=reduce?.94:0,playing=!reduce,last=0;

function pal(){const c=getComputedStyle(document.documentElement),g=n=>c.getPropertyValue('--'+n).trim();
 return {sf:g('surface'),sk:g('sunk'),ink:g('ink'),mut:g('muted'),ln:g('line'),fn:g('faint'),
         dir:g('direct'),tf:g('tf'),se:g('se'),bz:g('boltz'),wrn:g('warn')};}

/* ---------- data, verbatim from mlp-embeds/figs/compare/summary_all75_test_metrics.md ---------- */
const SPLITS=[
 {k:'A',  short:'random',  plain:'rows shuffled',      tiny:'shuffled'},
 {k:'B',  short:'peptide', plain:'unseen peptides',    tiny:'new peptide'},
 {k:'C',  short:'allele',  plain:'unseen alleles',     tiny:'new allele'},
 {k:'C2e',short:'cluster', plain:'unseen grooves',     tiny:'new groove'}];
const M={
 direct:{A:[0.757,0.736,0.520,4.366,11.391],B:[0.715,0.718,0.433,4.982,13.051],C:[0.584,0.501,0.387,4.359,8.355],C2e:[-0.034,0.024,0.017,3.215,6.680]},
 transformer:{A:[0.655,0.653,0.378,4.851,11.932],B:[0.643,0.667,0.381,5.214,12.803],C:[0.379,0.365,0.143,4.841,9.126],C2e:[0.158,0.216,0.027,3.192,6.471]},
 semlp:{A:[0.773,0.745,0.564,4.252,11.019],B:[0.723,0.724,0.457,4.783,12.200],C:[0.663,0.607,0.434,4.269,8.296],C2e:[0.116,0.144,0.026,3.198,6.312]},
 sexgb:{A:[0.773,0.739,0.552,4.198,11.014],B:[0.720,0.720,0.468,4.633,11.768],C:[0.666,0.654,0.353,4.203,7.645],C2e:[0.014,0.103,0.059,3.279,6.380]},
 boltz2:{A:[0.792,0.766,0.576,4.196,11.297],B:[0.764,0.761,0.566,4.525,11.337],C:[0.792,0.773,0.578,3.252,6.479],C2e:[0.511,0.545,0.279,2.912,5.777]}};
const MODELS=[{id:'direct',l:'direct (MLP)',ck:'dir'},{id:'transformer',l:'transformer',ck:'tf'},
 {id:'semlp',l:'SE+MLP',ck:'se'},{id:'sexgb',l:'SE+XGB',ck:'se',dash:[5,3]},{id:'boltz2',l:'boltz2 (frozen)',ck:'bz'}];
/* the 34 NetMHCpan pseudosequence positions, 1-indexed on the 182-aa chain */
const PSEUDO=[7,9,24,45,59,62,63,66,67,69,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171];
const NHLA=182, NPEP=9, NALL=NHLA+NPEP;
const SURV=PSEUDO.map(p=>p-1).concat(Array.from({length:NPEP},(_,i)=>NHLA+i)); // 43 of the 191

/* ---------- helpers ---------- */
const cl=(v,a,b)=>Math.max(a,Math.min(b,v));
const sub=(x,a,b)=>cl((x-a)/(b-a),0,1);
const ease=x=>x<.5?2*x*x:1-Math.pow(-2*x+2,2)/2;
const lerp=(a,b,u)=>a+(b-a)*u;
function hex(c,al){const h=c.replace('#','');
  const n=parseInt(h.length===3?h.split('').map(s=>s+s).join(''):h,16);
  return `rgba(${n>>16&255},${n>>8&255},${n&255},${al})`;}
function font(k,s,w){const f=k==='m'?'"IBM Plex Mono",monospace':k==='d'?'"Chivo",system-ui,sans-serif':'"Source Sans 3",system-ui,sans-serif';
  return `${w||400} ${s}px ${f}`;}
function txt(s,x,y,o={}){ctx.save();ctx.globalAlpha=o.al??1;ctx.fillStyle=o.c||P.ink;
  ctx.font=font(o.f||'m',o.s||12,o.w);ctx.textAlign=o.ta||'left';ctx.fillText(s,x,y);ctx.restore();}
function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);
  ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
function proj(p,ang,cx,cy,zoom){const c=Math.cos(ang),s=Math.sin(ang);
  const x=p.x*c+p.z*s, z=-p.x*s+p.z*c, k=520/(520+z);
  return {x:cx+x*k*zoom,y:cy+p.y*k*zoom,k,z};}
const rnd=(s=>()=>((s=s*16807%2147483647)/2147483647))(42);
const JIT=Array.from({length:260},()=>rnd());

/* ---------- secondary structure + cartoon ribbons ----------
   Ca-only assignment (P-SEA style): score each window against ideal helix and
   ideal strand geometry, take the better fit, then drop runs too short to be
   real. On this chain it recovers the two groove helices and the sheet floor. */
function assignSS(ca){
  const n=ca.length, d=(i,j)=>Math.hypot(ca[i][0]-ca[j][0],ca[i][1]-ca[j][1],ca[i][2]-ca[j][2]);
  const H={2:5.5,3:5.3,4:6.2}, E={2:6.7,3:9.9,4:12.4}, TOL={2:1.0,3:1.1,4:1.3};
  const fit=(i,T)=>{let s=0;for(const k of [2,3,4]){if(i+k>=n)return 1e9;
    s+=Math.pow((d(i,i+k)-T[k])/TOL[k],2);} return s/3;};
  const ss=new Array(n).fill('C');
  for(let i=0;i+4<n;i++){
    const h=fit(i,H), e=fit(i,E);
    if(Math.min(h,e)>1)continue;
    const t=h<e?'H':'E';
    for(let k=i;k<=i+4;k++) if(ss[k]==='C') ss[k]=t;
  }
  const MIN={H:6,E:3};
  for(let pass=0;pass<2;pass++){let s=0;
    for(let i=1;i<=n;i++) if(i===n||ss[i]!==ss[s]){
      const t=ss[s]; if(t!=='C'&&i-s<MIN[t]) for(let k=s;k<i;k++) ss[k]='C';
      s=i;}}
  return ss.join('');
}
const SS=assignSS(BOLTZ.tr);
/* covalent bonds at 1.95 A: 74 bonds over 74 heavy atoms, 8 of them the
   inter-residue peptide bonds, one ring (His imidazole at P7). */
const BONDS=(()=>{const o=[];
  for(let i=0;i<ATOMS.length;i++)for(let j=i+1;j<ATOMS.length;j++){
    const a=ATOMS[i].p,c=ATOMS[j].p;
    if(Math.hypot(a.x-c.x,a.y-c.y,a.z-c.z)<1.95*SC)o.push([i,j]);}
  return o;})();
const RUNS=(()=>{const r=[];let s=0;
  for(let i=1;i<=SS.length;i++) if(i===SS.length||SS[i]!==SS[s]){
    r.push({t:SS[s],a:Math.max(0,s-1),b:Math.min(SS.length-1,i)}); s=i;}
  return r;})();

function chaikin(pts,iters){
  let p=pts;
  for(let n=0;n<iters;n++){
    const q=[p[0]];
    for(let i=0;i<p.length-1;i++){const a=p[i],b=p[i+1];
      q.push({x:a.x*.75+b.x*.25,y:a.y*.75+b.y*.25,z:a.z*.75+b.z*.25});
      q.push({x:a.x*.25+b.x*.75,y:a.y*.25+b.y*.75,z:a.z*.25+b.z*.75});}
    q.push(p[p.length-1]); p=q;
  }
  return p;
}
/* mix a hex colour toward white (t>0) or black (t<0) */
function shade(c,t){
  const h=c.replace('#',''), n=parseInt(h.length===3?h.split('').map(s=>s+s).join(''):h,16);
  let r=n>>16&255,g=n>>8&255,b=n&255;
  const m=t>0?255:0, a=Math.abs(t);
  r=Math.round(r+(m-r)*a); g=Math.round(g+(m-g)*a); b=Math.round(b+(m-b)*a);
  return `rgb(${r},${g},${b})`;
}

/* Painter's-algorithm cartoon: every ribbon segment and every peptide atom
   becomes one primitive, all sorted far-to-near, so the sheet occludes
   correctly as the complex turns. */
function drawCartoon(ang,cx,cy,zoom,al){
  const HW=2.1*SC*zoom, EW=1.6*SC*zoom, CW=0.62*SC*zoom, HEAD=.72;
  const BR=0.36*SC*zoom, SW=0.46*SC*zoom;
  const prims=[];

  RUNS.forEach(r=>{
    const raw=TRACE.slice(r.a,r.b+1); if(raw.length<2)return;
    const sm=chaikin(raw, r.t==='H'?2:3);
    const sp=sm.map(q=>proj(q,ang,cx,cy,zoom));
    const n=sp.length;
    const wAt=u=>r.t==='H'?HW : r.t==='E'?(u<HEAD?EW:EW*2.0*(1-(u-HEAD)/(1-HEAD))) : CW;
    // offset each point perpendicular to the screen-space tangent
    const Lp=[],Rp=[];
    for(let i=0;i<n;i++){
      const a=sp[Math.max(0,i-1)], c=sp[Math.min(n-1,i+1)];
      const dx=c.x-a.x, dy=c.y-a.y, m=Math.hypot(dx,dy)||1;
      const w=wAt(i/(n-1))*sp[i].k;
      Lp.push([sp[i].x-dy/m*w, sp[i].y+dx/m*w]);
      Rp.push([sp[i].x+dy/m*w, sp[i].y-dx/m*w]);
    }
    for(let i=0;i<n-1;i++)
      prims.push({z:(sp[i].z+sp[i+1].z)/2, t:r.t,
                  q:[Lp[i],Lp[i+1],Rp[i+1],Rp[i]], k:(sp[i].k+sp[i+1].k)/2});
  });

  const ap=ATOMS.map(a=>proj(a.p,ang,cx,cy,zoom));
  ap.forEach(s=>prims.push({z:s.z,t:'P',x:s.x,y:s.y,r:BR*s.k,k:s.k}));
  BONDS.forEach(([i,j])=>{const a=ap[i],c=ap[j];
    prims.push({z:(a.z+c.z)/2,t:'B',x:a.x,y:a.y,x2:c.x,y2:c.y,w:SW*(a.k+c.k)/2,k:(a.k+c.k)/2});});

  prims.sort((a,b)=>b.z-a.z);

  prims.forEach(o=>{
    ctx.save(); ctx.globalAlpha=al;
    const lit=cl((o.k-.86)*2.1,-.3,.34);           // nearer segments catch the light
    if(o.t==='B'){
      ctx.lineCap='round';
      ctx.strokeStyle=shade(P.dir,-.45); ctx.lineWidth=Math.max(1.4,o.w+1.5);
      ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(o.x2,o.y2);ctx.stroke();
      ctx.strokeStyle=shade(P.dir,lit*.9); ctx.lineWidth=Math.max(.9,o.w);
      ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(o.x2,o.y2);ctx.stroke();
    }else if(o.t==='P'){
      ctx.fillStyle=shade(P.dir,lit*.9);
      ctx.strokeStyle=shade(P.dir,-.45); ctx.lineWidth=.9;
      ctx.beginPath();ctx.arc(o.x,o.y,Math.max(.9,o.r),0,7);ctx.fill();ctx.stroke();
    }else{
      const base=o.t==='C'?shade(P.bz,-.12):P.bz;
      ctx.fillStyle=shade(base,lit);
      ctx.beginPath();ctx.moveTo(o.q[0][0],o.q[0][1]);
      for(let i=1;i<4;i++)ctx.lineTo(o.q[i][0],o.q[i][1]);
      ctx.closePath();ctx.fill();
      // the two long edges give the ribbon its slab thickness
      ctx.strokeStyle=shade(P.bz,-.46); ctx.lineWidth=o.t==='C'?.8:1.05;
      ctx.lineCap='round';
      ctx.beginPath();ctx.moveTo(o.q[0][0],o.q[0][1]);ctx.lineTo(o.q[1][0],o.q[1][1]);ctx.stroke();
      ctx.beginPath();ctx.moveTo(o.q[2][0],o.q[2][1]);ctx.lineTo(o.q[3][0],o.q[3][1]);ctx.stroke();
    }
    ctx.restore();
  });
}

/* ---------- layout ---------- */
function L(){
  return narrow
   ? {rail:{x:16,y:34,w:368,h:22,horiz:true}, pan:{x:16,y:80,w:368,h:438}}
   : {rail:{x:30,y:54,w:122,h:426,horiz:false}, pan:{x:188,y:52,w:782,h:428}};
}
const RAIL=[['LEVEL 1','BASELINE'],['LEVEL 1b','SE GATE'],['LEVEL 2','ATTENTION'],['LEVEL 3','BOLTZ-2'],['','RESULTS']];
function drawRail(idx,r){
  const N=RAIL.length;
  if(r.horiz){
    const w=r.w/N;
    RAIL.forEach((s,i)=>{const x=r.x+i*w, on=i===idx;
      ctx.save();ctx.globalAlpha=on?1:.3;ctx.strokeStyle=on?P.ink:P.ln;ctx.lineWidth=on?2.5:1.5;
      ctx.beginPath();ctx.moveTo(x+2,r.y+16);ctx.lineTo(x+w-4,r.y+16);ctx.stroke();ctx.restore();
      txt(s[1].slice(0,5),x+2,r.y+9,{s:8,c:on?P.ink:P.mut,al:on?1:.4,w:on?600:400});});
  }else{
    RAIL.forEach((s,i)=>{const y=r.y+i*(r.h/N), on=i===idx;
      ctx.save();ctx.globalAlpha=on?1:.26;ctx.fillStyle=on?P.ink:P.ln;
      rr(r.x,y,on?3:2,34,1.5);ctx.fill();ctx.restore();
      txt(s[0],r.x+14,y+12,{s:9,c:P.mut,al:on?.9:.3});
      txt(s[1],r.x+14,y+27,{s:on?12.5:11.5,c:on?P.ink:P.mut,al:on?1:.35,w:on?600:400});});
  }
}

/* ---------- cell grids ---------- */
function gridRect(i,n,reg,rows,gap){
  const per=Math.ceil(n/rows), w=(reg.w-(per-1)*gap)/per, h=reg.h/rows-gap;
  const r=Math.floor(i/per), c=i-r*per;
  return {x:reg.x+c*(w+gap), y:reg.y+r*(h+gap), w, h};
}

/* ---------- scene 1: the real baseline ---------- */
function s1(p,pan){
  txt('LEVEL 1 — DIRECT (MLP) · THE BASELINE',pan.x,pan.y+14,{s:narrow?11:13,w:600,c:P.dir,f:'d'});
  txt('hla_pseudoseq 34 aa + peptide 9 aa + allele index',pan.x,pan.y+(narrow?30:34),{s:narrow?9.5:11,c:P.mut});

  const rows=narrow?4:2, gap=narrow?1:1.5;
  const AW=narrow?44:70;                     // allele block sits to the right
  const regA={x:pan.x,y:pan.y+(narrow?92:120),w:pan.w,h:narrow?78:58};
  const regB={x:pan.x,y:pan.y+(narrow?100:128),w:pan.w-AW-(narrow?8:14),h:narrow?32:42};
  const drop=ease(sub(p,.03,.22)), move=ease(sub(p,.14,.40)), grow=sub(p,.32,.50);
  const survSet=new Set(SURV);

  for(let i=0;i<NALL;i++){
    const A=gridRect(i,NALL,regA,rows,gap);
    if(!survSet.has(i)){
      const a=1-drop; if(a<=.01)continue;
      ctx.save();ctx.globalAlpha=a*.4;ctx.fillStyle=i>=NHLA?P.dir:P.ln;
      ctx.fillRect(A.x,A.y+(1-a)*18,A.w,A.h);ctx.restore();continue;
    }
    const k=SURV.indexOf(i), B=gridRect(k,SURV.length,regB,1,narrow?2:3);
    const x=lerp(A.x,B.x,move), y=lerp(A.y,B.y,move), w=lerp(A.w,B.w,move);
    const hh=lerp(A.h,B.h*(.3+JIT[k]*.7),Math.max(move,grow));
    ctx.save();ctx.globalAlpha=.55+JIT[k]*.45;ctx.fillStyle=i>=NHLA?P.dir:P.mut;
    ctx.fillRect(x,y+(B.h-hh)*move,w,hh);ctx.restore();
  }
  // the third input: a learned per-allele vector
  const ae=sub(p,.38,.55);
  if(ae>0){
    const ax=pan.x+pan.w-AW;
    for(let k=0;k<6;k++){const w=AW/6;
      ctx.save();ctx.globalAlpha=ae*(.35+JIT[k+90]*.6);ctx.fillStyle=P.bz;
      const hh=regB.h*(.3+JIT[k+90]*.65);
      ctx.fillRect(ax+k*w,regB.y+regB.h-hh,w-1,hh);ctx.restore();}
    txt('allele',ax,regB.y+regB.h+(narrow?12:14),{s:narrow?8.5:9.5,c:P.bz,al:ae});
  }
  const ly=regB.y+regB.h+(narrow?12:14);
  txt('34 pocket residues',pan.x,ly,{s:narrow?9:10.5,c:P.mut,al:move});
  txt('peptide 9',pan.x+regB.w,ly,{s:narrow?9:10.5,c:P.dir,ta:'right',al:move});
  txt('shared AA embedding → flatten → concat allele → MLP head',pan.x,ly+(narrow?20:24),
    {s:narrow?10:11.5,c:P.ink,al:grow});
  resultStrip('direct',p,pan,ly+(narrow?44:52)+38,.56);
}

/* ---------- scene 2: squeeze-and-excitation ---------- */
function s2(p,pan){
  txt('LEVEL 1b — SQUEEZE-AND-EXCITATION GATE',pan.x,pan.y+14,{s:narrow?11:13,w:600,c:P.se,f:'d'});
  txt('the identical direct-network features · 704 channels, recalibrated',pan.x,pan.y+(narrow?30:34),{s:narrow?9.5:11,c:P.mut});

  const N=narrow?48:72, gap=narrow?2:3;
  const reg={x:pan.x,y:pan.y+(narrow?150:166),w:pan.w,h:narrow?40:52};
  const show=sub(p,.02,.14), squeeze=sub(p,.10,.32), gate=sub(p,.28,.50);
  for(let i=0;i<N;i++){
    const R=gridRect(i,N,reg,1,gap);
    const base=.25+JIT[i+120]*.7;
    const g=.2+Math.abs(Math.sin(i*.55+1.1))*1.05;      // learned per-channel multiplier
    const hh=R.h*cl(lerp(base,base*g,gate),.06,1);
    ctx.save();ctx.globalAlpha=show*(.35+JIT[i+150]*.6);
    ctx.fillStyle=gate>.1&&g>1?P.se:(gate>.1&&g<.6?P.ln:P.mut);
    ctx.fillRect(R.x,R.y+R.h-hh,R.w,hh);ctx.restore();
  }
  // the squeeze: mean/var per field
  if(squeeze>0){
    const fy=reg.y-(narrow?34:42), fw=pan.w/3;
    ['peptide','HLA','allele'].forEach((f,i)=>{
      const a=sub(squeeze,i*.12,.5+i*.12);
      ctx.save();ctx.globalAlpha=a*.5;ctx.strokeStyle=P.se;ctx.lineWidth=1;
      rr(pan.x+i*fw,fy,fw-(narrow?6:12),narrow?22:26,4);ctx.stroke();ctx.restore();
      txt(f+'  μ σ²',pan.x+i*fw+(narrow?6:12),fy+(narrow?15:17),{s:narrow?8.5:10,c:P.se,al:a});
      ctx.save();ctx.globalAlpha=a*.3;ctx.strokeStyle=P.se;ctx.lineWidth=1;ctx.setLineDash([2,3]);
      ctx.beginPath();ctx.moveTo(pan.x+i*fw+fw/2-6,fy+(narrow?24:28));
      ctx.lineTo(pan.x+i*fw+fw/2-6,reg.y-4);ctx.stroke();ctx.restore();});
    txt('squeeze: 6 summary statistics steer the gate',pan.x,fy-(narrow?10:14),{s:narrow?9:10.5,c:P.mut,al:squeeze});
  }
  txt('two heads on the recalibrated vector — MLP, and XGBoost',pan.x,reg.y+reg.h+(narrow?22:26),
    {s:narrow?10:11.5,c:P.ink,al:gate});
  resultStrip('semlp',p,pan,reg.y+reg.h+(narrow?22:26)+38,.56,'SE+MLP');
}

/* ---------- scene 3: attention ---------- */
function s3(p,pan){
  txt('LEVEL 2 — CONTENT-DEPENDENT SELF-ATTENTION',pan.x,pan.y+14,{s:narrow?11:13,w:600,c:P.tf,f:'d'});
  txt('the same indices · [CLS] + peptide + pseudoseq · d_model 64, 4 heads',pan.x,pan.y+(narrow?30:34),{s:narrow?9.5:11,c:P.mut});

  const N=44, gap=narrow?2:3;
  const reg={x:pan.x,y:pan.y+(narrow?152:168),w:pan.w,h:narrow?34:46};
  const ins=ease(sub(p,.03,.15)), arcs=sub(p,.10,.72), mix=sub(p,.22,.46);
  for(let i=0;i<N;i++){
    const R=gridRect(i,N,reg,1,gap);
    const isCls=i===0, isPep=i>=N-NPEP;
    const base=.3+JIT[i]*.7, ctxd=.3+Math.abs(Math.sin(i*1.9))*.7;
    const hh=R.h*lerp(base,ctxd,mix);
    ctx.save();ctx.globalAlpha=(isCls?ins:1)*(.4+JIT[i+40]*.55);
    ctx.fillStyle=isCls?P.bz:(isPep?P.dir:P.tf);
    ctx.fillRect(R.x,R.y+R.h-hh,R.w,hh);ctx.restore();
  }
  txt('[CLS]',gridRect(0,N,reg,1,gap).x,reg.y+reg.h+(narrow?13:15),{s:narrow?8.5:9.5,c:P.bz,al:ins});
  ctx.save();ctx.lineWidth=narrow?.8:1;
  for(let q=0;q<N;q+=2)for(let k=q+3;k<N;k+=5){
    const u=(q/N+k/N)/2, w=Math.exp(-Math.pow((u-(arcs*1.45-.22))/.17,2));
    if(w<.04)continue;
    const A=gridRect(q,N,reg,1,gap), B=gridRect(k,N,reg,1,gap);
    const ax=A.x+A.w/2, bx=B.x+B.w/2, top=reg.y-Math.min(narrow?48:76,(bx-ax)*.42);
    ctx.strokeStyle=hex(P.tf,w*.5);
    ctx.beginPath();ctx.moveTo(ax,reg.y-2);ctx.quadraticCurveTo((ax+bx)/2,top,bx,reg.y-2);ctx.stroke();
  }
  ctx.restore();
  txt('all-pairs attention over 44 tokens, trained from scratch on 9,031 rows',
    pan.x,reg.y+reg.h+(narrow?30:34),{s:narrow?10:11.5,c:P.ink,al:mix});
  resultStrip('transformer',p,pan,reg.y+reg.h+(narrow?30:34)+38,.56);
}

/* ---------- scene 4: boltz-2 ---------- */
function s4(p,pan,ang){
  txt('LEVEL 3 — 3D CO-FOLDING REPRESENTATION',pan.x,pan.y+14,{s:narrow?11:13,w:600,c:P.bz,f:'d'});
  txt('precomputed Boltz-2 complex embedding · frozen · z-scored from train',pan.x,pan.y+(narrow?30:34),{s:narrow?9.5:11,c:P.mut});
  txt('\u03b11/\u03b12 platform \u00b7 2 helices, 9 strands \u00b7 ALLENIHRV in the groove',
    pan.x,pan.y+pan.h-(narrow?76:72),{s:narrow?9:10.5,c:P.mut,al:sub(p,.14,.26)*(1-sub(p,.42,.56))});

  const show=sub(p,.02,.12), fold=sub(p,.40,.58), vec=sub(p,.60,.78);
  const cx=pan.x+pan.w/2, cy=pan.y+(narrow?166:146), zoom=(narrow?.42:.66)*(1-fold*.45);
  if(show>0&&fold<1){
    const al=show*(1-fold*.9);
    drawCartoon(ang,cx,cy,zoom,al);
  }
  const reg={x:pan.x,y:pan.y+(narrow?236:220),w:pan.w,h:narrow?44:58};
  if(vec>0){
    const N=narrow?80:160;
    for(let i=0;i<N;i++){
      const a=sub(vec,(i/N)*.55,.3+(i/N)*.55);
      if(a<=0)continue;
      const w=reg.w/N, h=reg.h*(.16+Math.abs(Math.sin(i*.41)+Math.sin(i*.13))*.42);
      ctx.save();ctx.globalAlpha=a*(.3+Math.abs(Math.sin(i*.27))*.65);ctx.fillStyle=P.bz;
      ctx.fillRect(reg.x+i*w,reg.y+reg.h-h,w-.6,h);ctx.restore();
    }
    txt('1,547-d → the same 2×256 ReLU MLP head as the baseline',pan.x,reg.y+reg.h+(narrow?17:20),
      {s:narrow?10:11.5,c:P.ink,al:sub(vec,.45,.75)});
    txt('backbone frozen',pan.x+pan.w,reg.y+reg.h+(narrow?17:20),{s:narrow?9.5:10.5,c:P.bz,ta:'right',al:sub(vec,.45,.75)});
  }
  resultStrip('boltz2',p,pan,reg.y+reg.h+(narrow?17:20)+38,.72);
}

/* ---------- shared results strip ---------- */
function resultStrip(id,p,pan,top,at,name){
  const a=sub(p,at,at+.12); if(a<=0)return;
  const w=pan.w/4, S=narrow?.82:1;

  ctx.save();ctx.globalAlpha=a*.6;ctx.strokeStyle=P.ln;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pan.x,top);ctx.lineTo(pan.x+pan.w,top);ctx.stroke();ctx.restore();
  txt('PEARSON r  ·  HELD-OUT TEST'+(name?'  ·  '+name:''),pan.x,top-7,
    {s:narrow?8.5:10,c:P.mut,w:500,al:a});

  SPLITS.forEach((s,i)=>{
    const aa=sub(p,at+i*.022,at+.1+i*.022), x=pan.x+i*w, v=M[id][s.k][0];
    // key badge
    const bw=(s.k.length>1?26:17)*S, bh=15*S;
    ctx.save();ctx.globalAlpha=aa;ctx.fillStyle=P.mut;
    rr(x,top+11,bw,bh,3);ctx.fill();ctx.restore();
    txt(s.k,x+bw/2,top+11+bh-4*S,{s:(narrow?8.5:10)*1,c:P.sf,ta:'center',w:600,al:aa});
    txt(s.short,x+bw+6*S,top+11+bh-4*S,{s:narrow?9:11,c:P.ink,w:600,al:aa});
    txt(s.plain,x,top+(narrow?40:42),{s:narrow?8.5:10,c:P.mut,al:aa});
    txt(v.toFixed(3),x,top+(narrow?64:70),{s:narrow?17:24,w:600,c:v>.4?P.ink:P.wrn,al:aa});
  });

  // difficulty ramp under the four values
  const ry=top+(narrow?76:84), g=sub(p,at+.06,at+.18);
  if(g>0){
    ctx.save();ctx.globalAlpha=g*.45;ctx.strokeStyle=P.ln;ctx.lineWidth=1.5;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(pan.x,ry);ctx.lineTo(pan.x+pan.w-(narrow?44:58),ry);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pan.x+pan.w-(narrow?50:64),ry-3.5);
    ctx.lineTo(pan.x+pan.w-(narrow?44:58),ry);ctx.lineTo(pan.x+pan.w-(narrow?50:64),ry+3.5);
    ctx.stroke();ctx.restore();
    txt('more leakage',pan.x,ry-6,{s:narrow?8:9.5,c:P.mut,al:g*.9});
    txt('harder',pan.x+pan.w,ry+3,{s:narrow?8:9.5,c:P.mut,ta:'right',al:g*.9,w:500});
  }
}

/* ---------- scene 5: results ---------- */
function s5(p,pan){
  txt('WHERE EACH REPRESENTATION BREAKS',pan.x,pan.y+14,{s:narrow?11:13,w:600,c:P.ink,f:'d'});
  txt('Pearson r · 9,031 embedded rows, 54 alleles · splits by decreasing leakage',pan.x,pan.y+(narrow?30:34),{s:narrow?9.5:11,c:P.mut});

  const X0=pan.x+(narrow?34:44), X1=pan.x+pan.w-(narrow?6:10);
  const Y0=pan.y+(narrow?62:70), Y1=pan.y+pan.h-(narrow?124:98);
  const RMIN=-.12,RMAX=.85;
  const xOf=i=>X0+(X1-X0)*(i/3), yOf=r=>Y1-(r-RMIN)/(RMAX-RMIN)*(Y1-Y0);

  const grid=sub(p,0,.12);
  ctx.save();ctx.globalAlpha=grid;
  for(let r=0;r<=.8001;r+=.2){const v=Math.round(r*10)/10,y=yOf(v);
    ctx.strokeStyle=P.ln;ctx.lineWidth=1;ctx.setLineDash(v===0?[]:[2,4]);
    ctx.beginPath();ctx.moveTo(X0,y);ctx.lineTo(X1,y);ctx.stroke();
    txt(v.toFixed(1),X0-7,y+4,{s:narrow?9:10.5,c:P.mut,ta:'right'});}
  ctx.restore();
  SPLITS.forEach((s,i)=>{const x=xOf(i), ta=i===0?'left':i===3?'right':'center';
    const lab=s.k+' · '+s.short;
    txt(lab,x,Y1+(narrow?19:23),{s:narrow?9.5:12,c:P.ink,ta,w:600,al:grid});
    txt(s.plain,x,Y1+(narrow?31:38),{s:narrow?8.5:10.5,c:i===3?P.wrn:P.mut,ta,al:grid});});
  const ay=Y1+(narrow?44:54);
  ctx.save();ctx.globalAlpha=grid*.45;ctx.strokeStyle=P.ln;ctx.lineWidth=1.5;ctx.lineCap='round';
  ctx.beginPath();ctx.moveTo(X0,ay);ctx.lineTo(X1-(narrow?40:54),ay);ctx.stroke();
  ctx.beginPath();ctx.moveTo(X1-(narrow?46:60),ay-3.5);ctx.lineTo(X1-(narrow?40:54),ay);
  ctx.lineTo(X1-(narrow?46:60),ay+3.5);ctx.stroke();ctx.restore();
  txt('more leakage',X0,ay-5,{s:narrow?8:9.5,c:P.mut,al:grid*.9});
  txt('harder',X1,ay+3,{s:narrow?8:9.5,c:P.mut,ta:'right',al:grid*.9,w:500});

  ['transformer','direct','sexgb','semlp','boltz2'].forEach((id,oi)=>{
    const m=MODELS.find(q=>q.id===id), lead=id==='boltz2';
    const start=.10+oi*.055, a=sub(p,start,start+.17);
    if(a<=0)return;
    const pts=SPLITS.map((s,i)=>[xOf(i),yOf(M[id][s.k][0])]), grow=ease(a)*3;
    ctx.save();ctx.strokeStyle=P[m.ck];ctx.globalAlpha=lead?1:.45;
    ctx.lineWidth=lead?(narrow?2.6:3.4):(narrow?1.3:1.8);ctx.lineJoin='round';ctx.lineCap='round';
    if(m.dash)ctx.setLineDash(m.dash);
    ctx.beginPath();
    for(let i=0;i<3;i++){const seg=cl(grow-i,0,1); if(seg<=0)break;
      const A=pts[i],B=pts[i+1]; if(i===0)ctx.moveTo(A[0],A[1]);
      ctx.lineTo(lerp(A[0],B[0],seg),lerp(A[1],B[1],seg));}
    ctx.stroke();ctx.restore();
    pts.forEach((q,i)=>{if(grow<i)return;
      ctx.save();ctx.globalAlpha=lead?1:.5;ctx.fillStyle=P.sf;ctx.strokeStyle=P[m.ck];
      ctx.lineWidth=lead?2.6:1.6;
      ctx.beginPath();ctx.arc(q[0],q[1],lead?(narrow?4:4.8):(narrow?2.2:2.9),0,7);
      ctx.fill();ctx.stroke();ctx.restore();});
    if(lead&&grow>=3)pts.forEach((q,i)=>
      txt(M[id][SPLITS[i].k][0].toFixed(3),q[0],q[1]-(narrow?10:13),
        {s:narrow?10:11.5,w:600,c:P.bz,ta:i===3?'right':'center',al:sub(p,.50,.60)}));
    if(grow>=3)txt(m.l,pts[3][0]+(narrow?0:6),pts[3][1]+(lead?(narrow?20:24):4),
      {s:narrow?8.5:10.5,c:P[m.ck],ta:'right',al:lead?1:.6,w:lead?600:400});
  });

  const k=sub(p,.56,.68);
  if(k>0){
    const by=Y1+(narrow?62:74);
    ctx.save();ctx.globalAlpha=k*.5;ctx.strokeStyle=P.bz;ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(pan.x,by);ctx.lineTo(pan.x+pan.w*ease(k),by);ctx.stroke();ctx.restore();
    txt('On the cluster split every sequence model is at chance.',pan.x,by+(narrow?19:24),
      {s:narrow?11:14,c:P.ink,f:'s',w:600,al:k});
    txt('Frozen Boltz-2 reaches r = 0.511, against −0.034 for the baseline MLP.',pan.x,by+(narrow?35:44),
      {s:narrow?9.5:12.5,c:P.mut,f:'s',al:sub(p,.64,.74)});
    txt('C2e holds only 2 test clusters — read that column as a spread, not a point value.',
      pan.x,by+(narrow?50:62),{s:narrow?9:11,c:P.wrn,f:'s',al:sub(p,.72,.82)});
  }
}

/* ---------- compose ---------- */
const CUT=[.20,.35,.50,.76];
const SCENES=[s1,s2,s3,s4,s5];
function frame(){
  ctx.clearRect(0,0,VW,VH);ctx.fillStyle=P.sf;ctx.fillRect(0,0,VW,VH);
  const g=L(), ang=t*Math.PI*2.6;
  let idx=CUT.findIndex(c=>t<c); if(idx<0)idx=4;
  drawRail(idx,g.rail);
  const bounds=[0,...CUT,1], seg=[bounds[idx],bounds[idx+1]];
  SCENES[idx](sub(t,seg[0],seg[1]),g.pan,ang);
  const fade=Math.min(sub(t,seg[0],seg[0]+.022),1-sub(t,seg[1]-.022,seg[1]));
  if(fade<1){ctx.save();ctx.globalAlpha=1-fade;ctx.fillStyle=P.sf;
    ctx.fillRect(g.pan.x-6,g.pan.y-6,g.pan.w+12,g.pan.h+14);ctx.restore();}
  chips.forEach((c,i)=>c.setAttribute('aria-current',String(i===idx)));
}

function resize(){
  const w=cv.parentElement.clientWidth;
  narrow=w<680; VW=narrow?400:1000; VH=narrow?540:520;
  const dpr=Math.min(devicePixelRatio||1,2);
  cv.style.height=(w*VH/VW)+'px';
  cv.width=Math.round(w*dpr); cv.height=Math.round(w*VH/VW*dpr);
  const s=(w/VW)*dpr; ctx.setTransform(s,0,0,s,0,0);
  P=pal(); frame();
}
function loop(ts){
  if(playing){if(last)t=(t+(ts-last)/DUR)%1;last=ts;}else last=0;
  frame(); scrub.value=Math.round(t*1000);
  clock.textContent='00:'+String(Math.round(t*DUR/1000)).padStart(2,'0');
  requestAnimationFrame(loop);
}
function setPlay(v){playing=v;last=0;
  playBtn.setAttribute('aria-label',v?'Pause animation':'Play animation');
  icon.innerHTML=v?'<rect x="1.5" y="1" width="3" height="10" rx="1"/><rect x="7.5" y="1" width="3" height="10" rx="1"/>'
                 :'<path d="M2 1l9 5-9 5z"/>';}
playBtn.addEventListener('click',()=>setPlay(!playing));
scrub.addEventListener('input',()=>{t=scrub.value/1000;setPlay(false);});
chips.forEach(c=>c.addEventListener('click',()=>{t=parseFloat(c.dataset.at);setPlay(true);}));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{P=pal();});
new ResizeObserver(resize).observe(cv.parentElement);

/* ---------- table ---------- */
const COLS=['Pearson r','Spearman ρ','within-allele ρ','MAE (h)','RMSE (h)'], LOWER=[0,0,0,1,1];
let tab='C2e';
function buildTable(){
  document.getElementById('tabs').innerHTML=SPLITS.map(s=>
    `<button class="tab" data-k="${s.k}" aria-current="${s.k===tab}">${s.k} · ${s.short} — ${s.plain}</button>`).join('');
  document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{tab=b.dataset.k;buildTable();}));
  const best=COLS.map((_,c)=>{const v=MODELS.map(m=>M[m.id][tab][c]);
    return LOWER[c]?Math.min(...v):Math.max(...v);});
  document.getElementById('tbl').innerHTML=
    `<thead><tr><th>model</th>${COLS.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`+
    MODELS.map(m=>`<tr><td><span class="mk" style="background:var(--${m.ck==='dir'?'direct':m.ck==='bz'?'boltz':m.ck})"></span>${m.l}</td>`+
      M[m.id][tab].map((v,c)=>`<td class="${v===best[c]?'best':''}">${v.toFixed(3)}</td>`).join('')+'</tr>').join('')+'</tbody>';
}
buildTable(); setPlay(playing); resize(); requestAnimationFrame(loop);
})();
