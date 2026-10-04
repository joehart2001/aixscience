(function(){
const cv=document.getElementById('stage'), ctx=cv.getContext('2d');
const scrub=document.getElementById('scrub'), playBtn=document.getElementById('play');
const icon=document.getElementById('icon'), clock=document.getElementById('clock');
const chips=[...document.querySelectorAll('.chip')];
const DUR=68000;
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
 {k:'C2', short:'cluster', plain:'unseen grooves',     tiny:'new groove'}];
const M={
 direct:{A:[0.728,0.701,0.49,3.7,9.993],B:[0.696,0.689,0.461,4.303,12.49],C:[0.57,0.543,0.256,3.77,6.688],C2:[-0.065,-0.054,0.015,3.468,6.835]},
 transformer:{A:[0.634,0.636,0.34,4.104,10.582],B:[0.596,0.621,0.305,4.474,10.672],C:[0.375,0.304,0.123,4.167,8.054],C2:[-0.098,-0.127,0.041,3.397,6.911]},
 semlp:{A:[0.749,0.718,0.521,3.489,9.342],B:[0.712,0.714,0.461,4.563,17.082],C:[0.592,0.562,0.206,3.605,6.271],C2:[-0.108,-0.184,0.069,3.506,6.891]},
 sexgb:{A:[0.741,0.719,0.547,3.535,9.701],B:[0.729,0.728,0.528,3.808,10.207],C:[0.487,0.481,0.139,4.156,7.94],C2:[-0.075,-0.177,-0.047,3.303,6.594]},
 boltz2:{A:[0.794,0.776,0.595,3.374,9.545],B:[0.761,0.758,0.58,3.699,9.951],C:[0.714,0.703,0.548,3.164,6.368],C2:[0.425,0.485,0.44,3.03,6.548]}};
/* feature-ablation run, src/data/mace_feature_blocks.md — a separate experiment
   from the A/B/C/C2 table, so these are shown on their own terms */
const MACE={boltz:0.603, boltzNode:0.571, node:0.020, nodeTrain:0.768,
            energy:0.430, dims:{node:11520, edge:14450, energy:430}};
const MODELS=[{id:'direct',l:'direct (MLP)',ck:'dir'},{id:'transformer',l:'transformer',ck:'tf'},
 {id:'semlp',l:'SE+MLP',ck:'se'},{id:'sexgb',l:'SE+XGB',ck:'se',dash:[5,3]},{id:'boltz2',l:'boltz2 (frozen)',ck:'bz'}];
/* the 34 NetMHCpan pseudosequence positions, 1-indexed on the 182-aa chain */
const PSEUDO=[7,9,24,45,59,62,63,66,67,69,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171];
const NHLA=182, NPEP=9, NALL=NHLA+NPEP;
const SURV=PSEUDO.map(p=>p-1).concat(Array.from({length:NPEP},(_,i)=>NHLA+i)); // 43 of the 191

/* A real two-residue fragment of the peptide for the MPNN scene: His-Arg at
   P7-P8, 21 atoms and 21 bonds including the imidazole ring. The viewing
   angles were chosen by maximising the closest on-screen approach of any two
   atoms, so nothing stacks. Covalent bonds are drawn as sticks; the 5 A graph
   MACE actually passes messages on is overlaid dashed, which is the point. */
const FRAG=(()=>{
  const want=a=>a.ri===6||a.ri===7;
  const ids=ATOMS.map((a,i)=>i).filter(i=>want(ATOMS[i]));
  const D=(i,j)=>Math.hypot(ATOMS[i].p.x-ATOMS[j].p.x,
                            ATOMS[i].p.y-ATOMS[j].p.y,
                            ATOMS[i].p.z-ATOMS[j].p.z);
  const bonds=[];
  ids.forEach((i,n)=>ids.slice(n+1).forEach(j=>{if(D(i,j)<1.95*SC)bonds.push([i,j]);}));
  // centre on the atom with the busiest 5 A shell inside the fragment
  let c=ids[0],best=-1;
  ids.forEach(i=>{const n=ids.filter(j=>j!==i&&D(i,j)<5.0*SC).length;
    if(n>best){best=n;c=i;}});
  const one=ids.filter(j=>j!==c&&D(c,j)<5.0*SC);
  const two=ids.filter(j=>j!==c&&!one.includes(j)&&one.some(k=>D(k,j)<5.0*SC));
  const near=k=>one.reduce((b2,j)=>D(j,k)<D(b2,k)?j:b2,one[0]);
  return {ids,bonds,c,one,two,
          e1:one.map(j=>[c,j]),
          e2:two.map(k=>[near(k),k])};
})();
const FRAG_VIEW={a:1.04,b:0.70};
function fragLayout(rect){
  const {a,b}=FRAG_VIEW;
  const pts=FRAG.ids.map(i=>{const p=ATOMS[i].p;
    const x1=p.x*Math.cos(a)+p.z*Math.sin(a), z1=-p.x*Math.sin(a)+p.z*Math.cos(a);
    return {i,x:x1,y:p.y*Math.cos(b)-z1*Math.sin(b),z:p.y*Math.sin(b)+z1*Math.cos(b)};});
  let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
  pts.forEach(q=>{x0=Math.min(x0,q.x);x1=Math.max(x1,q.x);y0=Math.min(y0,q.y);y1=Math.max(y1,q.y);});
  const s=Math.min(rect.w/(x1-x0+26),rect.h/(y1-y0+26));
  const ox=rect.x+rect.w/2-(x0+x1)/2*s, oy=rect.y+rect.h/2-(y0+y1)/2*s;
  const P2={}; pts.forEach(q=>P2[q.i]={x:ox+q.x*s,y:oy+q.y*s,z:q.z}); return P2;
}
const ELCOL=()=>[P.bz,P.mut,P.wrn];        // N, C, O

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
  const HW=1.15*SC*zoom, EW=0.85*SC*zoom, CW=0.28*SC*zoom, HEAD=.68;
  const BR=0.26*SC*zoom, SW=0.30*SC*zoom;
  const prims=[];

  RUNS.forEach(r=>{
    const raw=TRACE.slice(r.a,r.b+1); if(raw.length<2)return;
    const sm=chaikin(raw, r.t==='H'?4:r.t==='E'?2:3);
    const sp=sm.map(q=>proj(q,ang,cx,cy,zoom));
    const n=sp.length;
    const wAt=u=>r.t==='H'?HW : r.t==='E'?(u<HEAD?EW:EW*2.8*(1-(u-HEAD)/(1-HEAD))) : CW;
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
    const lit=cl((o.k-.86)*2.7,-.40,.40);          // nearer segments catch the light
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
const RAIL=[['DEFAULT','SEQUENCE'],['FOUNDATION','BOLTZ-2'],['EVIDENCE','RESULTS'],['EXTENSION','MACE'],['FEATURE','SPACES']];
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

/* ---------- scene 1: the three sequence models, side by side ---------- */
const SEQCOLS=[
 {id:'direct', name:'Direct MLP', ck:'dir', kind:'bars',
  input:'34 pocket residues + 9 peptide + allele',
  note:'embeddings, flattened, 3-layer MLP'},
 {id:'semlp',  name:'SE gate',    ck:'se',  kind:'gate',
  input:'the same 704 channels, recalibrated',
  note:'learned per-channel gate, MLP and XGBoost heads'},
 {id:'transformer', name:'Transformer', ck:'tf', kind:'attn',
  input:'the same indices + [CLS]',
  note:'all-pairs attention, d_model 64, trained from scratch'}];

function s1(p,pan){
  txt('THE DEFAULT · LEARN FROM SEQUENCE',pan.x,pan.y+(narrow?16:22),{s:narrow?15:22,w:600,c:P.ink,f:'d'});
  txt(narrow?'three architectures · labels only':'three architectures, all learning interaction patterns from labels alone',pan.x,pan.y+(narrow?32:42),
    {s:narrow?9.5:11,c:P.mut});

  // each column is lit in turn, then all three together for the comparison
  const beats=[[.06,.30],[.30,.52],[.52,.72]];
  const focus=p<.30?0:p<.52?1:p<.72?2:-1;
  const GAP=narrow?10:22, cw=narrow?pan.w:(pan.w-2*GAP)/3;

  SEQCOLS.forEach((col,ci)=>{
    if(narrow&&focus>=0&&focus!==ci)return;          // phone shows one at a time
    if(narrow&&focus<0)return;                       // replaced by the summary below
    const x=narrow?pan.x:pan.x+ci*(cw+GAP), on=focus===ci||focus<0;
    const [b0,b1]=beats[ci], u=sub(p,b0,b1), done=p>=b1;
    const al=on?1:.26;
    const top=pan.y+(narrow?50:58);

    if(focus===ci){
      ctx.save();ctx.globalAlpha=.9;ctx.fillStyle=P[col.ck];
      rr(x-(narrow?5:8),top-10,2.5,narrow?118:150,1.5);ctx.fill();ctx.restore();
    }
    txt(col.name,x,top,{s:narrow?13:18,w:600,c:P[col.ck],f:'d',al});
    txt(col.input,x,top+(narrow?13:16),{s:narrow?8:9.5,c:P.mut,al:al*.95});

    // a compact picture of what this model consumes
    const vy=top+(narrow?24:32), vh=narrow?30:40, N=narrow?16:22;
    const bw=cw/N;
    for(let i=0;i<N;i++){
      const pep=i>=N-5, base=.28+JIT[ci*30+i]*.7;
      let h=base, c2=pep?P.dir:P.mut;
      if(col.kind==='gate'){
        const g=.25+Math.abs(Math.sin(i*.9+ci))*1.1;
        h=cl(lerp(base,base*g,done?1:ease(u)),.08,1);
        c2=(done||u>.1)?(g>1?P.se:hex(P.se,.4)):P.mut;
      }else if(col.kind==='attn'){
        h=lerp(base,.28+Math.abs(Math.sin(i*1.7))*.7,done?1:ease(u));
        c2=pep?P.dir:P.tf;
      }else{
        h=base*(done?1:ease(sub(u,i/N*.5,.4+i/N*.5)));
      }
      ctx.save();ctx.globalAlpha=al*(.45+JIT[ci*30+i+15]*.5);ctx.fillStyle=c2;
      ctx.fillRect(x+i*bw,vy+vh-vh*h,Math.max(1,bw-1),vh*h);ctx.restore();
    }
    if(col.kind==='attn'){
      ctx.save();ctx.lineWidth=1;
      for(let q=0;q<N;q+=3)for(let k=q+4;k<N;k+=7){
        const w=Math.exp(-Math.pow(((q/N+k/N)/2-((done?.5:ease(u))*1.4-.2))/.2,2));
        if(w<.05)continue;
        const ax=x+(q+.5)*bw, bx2=x+(k+.5)*bw;
        ctx.strokeStyle=hex(P.tf,w*.45*al);
        ctx.beginPath();ctx.moveTo(ax,vy-2);
        ctx.quadraticCurveTo((ax+bx2)/2,vy-Math.min(26,(bx2-ax)*.5),bx2,vy-2);ctx.stroke();}
      ctx.restore();
    }
    txt(col.note,x,vy+vh+(narrow?12:15),{s:narrow?7.5:9,c:P.mut,al:al*.9});

    // its four held-out scores, each on the same -0.2 to 1.0 track
    const sy=vy+vh+(narrow?26:34);
    SPLITS.forEach((s,si)=>{
      const v=M[col.id][s.k][0], ry=sy+si*(narrow?17:21);
      const sa=al*(done||u>.5?1:sub(u,.45+si*.05,.7+si*.05));
      txt(s.k,x,ry,{s:narrow?8:9.5,c:P.mut,al:sa});
      txt(s.short,x+(narrow?16:20),ry,{s:narrow?8:9.5,c:P.mut,al:sa*.85});
      txt(v.toFixed(3),x+(narrow?62:78),ry,{s:narrow?9.5:12,w:600,
        c:v>0?P.ink:P.wrn,al:sa});
      const tw=cw-(narrow?104:132), tx=x+(narrow?100:128), th=narrow?3:4;
      const zx=tx+tw*(0.2/1.2), vx=tx+tw*((v+0.2)/1.2);
      ctx.save();ctx.globalAlpha=sa*.45;ctx.fillStyle=P.ln;
      ctx.fillRect(tx,ry-th-1,tw,th);ctx.restore();
      ctx.save();ctx.globalAlpha=sa;ctx.fillStyle=v<0?P.wrn:P[col.ck];
      ctx.fillRect(Math.min(zx,vx),ry-th-1,Math.max(1.5,Math.abs(vx-zx)),th);ctx.restore();
    });
  });

  if(narrow&&focus<0){
    const sy=pan.y+(narrow?232:0);
    SEQCOLS.forEach((col,ci)=>{
      const ry=sy+ci*58;
      txt(col.name,pan.x,ry,{s:10.5,w:600,c:P[col.ck]});
      SPLITS.forEach((s,si)=>{
        const v=M[col.id][s.k][0], vx=pan.x+si*(pan.w/4);
        txt(s.k,vx,ry+16,{s:8,c:P.mut});
        txt(v.toFixed(3),vx,ry+32,{s:11,w:600,c:v>0?P.ink:P.wrn});});});
  }

  const g=sub(p,.78,.88);
  if(g>0){
    const ny=pan.y+pan.h-(narrow?26:22);
    ctx.save();ctx.globalAlpha=g*.5;ctx.strokeStyle=P.ln;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pan.x,ny-30);ctx.lineTo(pan.x+pan.w*ease(g),ny-30);
    ctx.stroke();ctx.restore();
    txt(narrow?'Sequence alone fails on a new groove.'
              :'Different architectures, same limitation: sequence alone fails on an unseen groove.',
      pan.x,ny,{s:narrow?12:18,c:P.ink,f:'s',w:600,al:g});
  }
}

/* ---------- scene 2: boltz-2 ---------- */
function s2(p,pan,ang){
  txt(narrow?'OUR LEVERAGE · PRETRAINED 3D':'OUR LEVERAGE · A PRETRAINED 3D REPRESENTATION',pan.x,pan.y+(narrow?16:22),{s:narrow?15:22,w:600,c:P.bz,f:'d'});
  txt(narrow?'reuse what Boltz-2 learned · frozen':'reuse what Boltz-2 learned about protein complexes · frozen · standardised on train only',pan.x,pan.y+(narrow?32:42),{s:narrow?9.5:11,c:P.mut});
  txt('\u03b11/\u03b12 platform \u00b7 2 helices, 9 strands \u00b7 ALLENIHRV in the groove',
    pan.x,pan.y+pan.h-(narrow?76:72),{s:narrow?9:10.5,c:P.mut,al:sub(p,.14,.26)*(1-sub(p,.34,.46))});

  const show=sub(p,.02,.12), fold=sub(p,.36,.50), vec=sub(p,.50,.62);
  // the complex does not leave: it shrinks into the top right and keeps turning
  const F=ease(fold);
  const cx=lerp(pan.x+pan.w/2, pan.x+pan.w-(narrow?64:92), F);
  const cy=lerp(pan.y+(narrow?160:140), pan.y+(narrow?40:70), F);
  const zoom=lerp(narrow?.50:.78, narrow?.21:.30, F);
  if(show>0)drawCartoon(ang,cx,cy,zoom,show);
  const reg={x:pan.x,y:pan.y+(narrow?214:196),w:pan.w,h:narrow?44:58};
  if(vec>0){
    // the actual 1,547-d vector for this complex, z-scored and bucketed to fit
    const N=EMB.z.length, mid=reg.y+reg.h/2, half=reg.h/2-1, w=reg.w/N;
    ctx.save();ctx.globalAlpha=sub(vec,.1,.4)*.35;ctx.strokeStyle=P.ln;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(reg.x,mid);ctx.lineTo(reg.x+reg.w,mid);ctx.stroke();ctx.restore();
    for(let i=0;i<N;i++){
      const a=sub(vec,(i/N)*.5,.28+(i/N)*.5);
      if(a<=0)continue;
      const h=(EMB.z[i]/2.6)*half;
      ctx.save();ctx.globalAlpha=a*.85;ctx.fillStyle=EMB.z[i]<0?hex(P.bz,.55):P.bz;
      ctx.fillRect(reg.x+i*w, h<0?mid:mid-h, Math.max(.6,w-.5), Math.abs(h));ctx.restore();
    }
    // the nine blocks it is concatenated from
    ctx.save();ctx.globalAlpha=sub(vec,.3,.6)*.4;ctx.strokeStyle=P.sf;ctx.lineWidth=1.4;
    EMB.bounds.forEach(f=>{const x=reg.x+f*reg.w;
      ctx.beginPath();ctx.moveTo(x,reg.y);ctx.lineTo(x,reg.y+reg.h);ctx.stroke();});
    ctx.restore();
    if(!narrow){
      let acc=0;
      EMB.labels.forEach(([n,sz])=>{
        const x0=reg.x+(acc/EMB.n)*reg.w, wd=(sz/EMB.n)*reg.w; acc+=sz;
        if(wd>70)txt(n,x0+3,reg.y-5,{s:9,c:P.mut,al:sub(vec,.35,.6)});});
    }
    txt(narrow?'1,547-d → the same MLP head':'1,547-d, z-scored → the same 2×256 ReLU MLP head as the baseline',
      pan.x,reg.y+reg.h+(narrow?17:20),{s:narrow?10:11.5,c:P.ink,al:sub(vec,.45,.75)});
    txt('backbone frozen',pan.x+pan.w,reg.y+reg.h+(narrow?32:20),{s:narrow?9.5:10.5,c:P.bz,ta:narrow?'left':'right',al:sub(vec,.45,.75)});
  }
  resultStrip('boltz2',p,pan,reg.y+reg.h+(narrow?46:44)+24,.56,null,
    ['0.714 on an unseen allele, against 0.592 for the best sequence model.',
     '0.714 vs 0.592 on an unseen allele.']);
}

/* ---------- shared results strip ---------- */
function resultStrip(id,p,pan,top,at,name,note){
  const a=sub(p,at,at+.12); if(a<=0)return;
  const w=pan.w/4, S=narrow?.82:1;

  ctx.save();ctx.globalAlpha=a*.6;ctx.strokeStyle=P.ln;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pan.x,top);ctx.lineTo(pan.x+pan.w,top);ctx.stroke();ctx.restore();
  txt((narrow?'FOUR HELD-OUT TEST SETS · PEARSON r'
             :'PEARSON r ON FOUR HELD-OUT TEST SETS, HARDER LEFT TO RIGHT')
      +(name?'  ·  '+name:''),pan.x,top-19,{s:narrow?8.5:10,c:P.mut,w:500,al:a});
  txt('1.0 perfect · 0 chance',pan.x+pan.w,top-19,
    {s:narrow?8:9.5,c:P.mut,ta:'right',al:a*.85});

  SPLITS.forEach((s,i)=>{
    const aa=sub(p,at+i*.022,at+.1+i*.022), x=pan.x+i*w, v=M[id][s.k][0];
    // key badge
    const bw=(s.k.length>1?26:17)*S, bh=15*S;
    ctx.save();ctx.globalAlpha=aa;ctx.fillStyle=P.mut;
    rr(x,top+11,bw,bh,3);ctx.fill();ctx.restore();
    txt(s.k,x+bw/2,top+11+bh-4*S,{s:(narrow?8.5:10)*1,c:P.sf,ta:'center',w:600,al:aa});
    txt(s.short,x+bw+6*S,top+11+bh-4*S,{s:narrow?9:11,c:P.ink,w:600,al:aa});
    txt(s.plain,x,top+(narrow?40:42),{s:narrow?8.5:10,c:P.mut,al:aa});
    txt(v.toFixed(3),x,top+(narrow?60:66),{s:narrow?17:24,w:600,c:v>.4?P.ink:P.wrn,al:aa});
    // track runs -0.2 to 1.0 with a tick at zero, so a negative bar reads as negative
    const tw=w-(narrow?14:26), ty=top+(narrow?70:76), th=narrow?4:5;
    const zx=x+tw*(0.2/1.2), vx=x+tw*((v+0.2)/1.2);
    ctx.save();ctx.globalAlpha=aa*.5;ctx.fillStyle=P.ln;
    ctx.fillRect(x,ty,tw,th);ctx.restore();
    ctx.save();ctx.globalAlpha=aa;ctx.fillStyle=v<0?P.wrn:P.bz;
    ctx.fillRect(Math.min(zx,vx),ty,Math.max(1.5,Math.abs(vx-zx)),th);ctx.restore();
    ctx.save();ctx.globalAlpha=aa*.75;ctx.fillStyle=P.mut;
    ctx.fillRect(zx-0.5,ty-2,1,th+4);ctx.restore();
  });

  // what the four numbers mean
  const ny=top+(narrow?92:98), g=sub(p,at+.14,at+.26);
  if(g>0&&note){
    ctx.save();ctx.globalAlpha=g;ctx.fillStyle=P.bz;
    ctx.beginPath();ctx.moveTo(pan.x,ny-7);ctx.lineTo(pan.x+6,ny-3.5);ctx.lineTo(pan.x,ny);
    ctx.closePath();ctx.fill();ctx.restore();
    txt(narrow?note[1]:note[0],pan.x+14,ny,{s:narrow?12:18,c:P.ink,f:'s',w:600,al:g});
  }
}

/* ---------- scene 3: results ---------- */
function s3(p,pan){
  txt('THE GAP OPENS ON HARDER HOLDOUTS',pan.x,pan.y+(narrow?16:22),{s:narrow?15:22,w:600,c:P.ink,f:'d'});
  txt('Pearson r · 14.5k rows, 75 alleles, 22 clusters · increasingly unfamiliar test sets',pan.x,pan.y+(narrow?32:42),{s:narrow?9.5:11,c:P.mut});

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
        {s:narrow?10:11.5,w:600,c:P.bz,ta:i===3?'right':'center',al:sub(p,.44,.54)}));

  });

  // end labels, separated so models that finish close together stay legible
  if(sub(p,.44,.54)>0){
    const gap=narrow?12:14;
    const ls=['transformer','direct','sexgb','semlp','boltz2'].map(id=>{
      const m=MODELS.find(q=>q.id===id);
      return {m,lead:id==='boltz2',y:yOf(M[id]['C2'][0])};});
    ls.sort((a,b2)=>a.y-b2.y);
    for(let i=1;i<ls.length;i++)
      if(ls[i].y-ls[i-1].y<gap) ls[i].y=ls[i-1].y+gap;
    const shift=Math.max(0,ls[ls.length-1].y-(Y1-4));
    ls.forEach(l=>{
      const y=l.y-shift;
      txt(l.m.l,xOf(3)+(narrow?0:6),y+(l.lead?(narrow?30:34):4),
        {s:narrow?8.5:10.5,c:P[l.m.ck],ta:'right',al:(l.lead?1:.6)*sub(p,.44,.54),
         w:l.lead?600:400});});
  }

  const k=sub(p,.54,.64);
  if(k>0){
    const by=Y1+(narrow?62:74);
    ctx.save();ctx.globalAlpha=k*.5;ctx.strokeStyle=P.bz;ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(pan.x,by);ctx.lineTo(pan.x+pan.w*ease(k),by);ctx.stroke();ctx.restore();
    txt('Transferred 3D features generalise where sequence models do not.',pan.x,by+(narrow?19:24),
      {s:narrow?14:24,c:P.ink,f:'s',w:600,al:k});
    txt('0.714 on held-out alleles against 0.592 for the best sequence model, and on unseen',
      pan.x,by+(narrow?34:44),{s:narrow?9.5:12.5,c:P.mut,f:'s',al:sub(p,.60,.70)});
    txt('clusters every sequence model goes negative while Boltz-2 holds 0.425.',
      pan.x,by+(narrow?47:60),{s:narrow?9.5:12.5,c:P.mut,f:'s',al:sub(p,.66,.76)});
    txt('C2 puts ~3 of 22 clusters in test, so treat the size of that gap cautiously.',
      pan.x,by+(narrow?62:78),{s:narrow?9:11,c:P.wrn,f:'s',al:sub(p,.72,.82)});
  }
}

/* ---------- scene 4: MACE as an extension ---------- */
function s4(p,pan){
  txt(narrow?'MACE · CAN WE IMPROVE FURTHER?':'MACE EXTENSION · CAN WE IMPROVE FURTHER?',pan.x,pan.y+(narrow?16:22),{s:narrow?15:22,w:600,c:P.se,f:'d'});
  txt(narrow?'atomistic foundation-model descriptors':'can atomistic-simulation foundation-model descriptors add to Boltz-2?',pan.x,pan.y+(narrow?32:42),{s:narrow?9.5:11,c:P.mut});

  const box={x:pan.x,y:pan.y+(narrow?50:54),w:narrow?pan.w:pan.w*.54,h:narrow?150:190};
  const L=fragLayout(box);
  const build=sub(p,.02,.12), l1=sub(p,.14,.34), l2=sub(p,.34,.54);
  const EC=ELCOL(), AR=narrow?4.2:5.6, CR=narrow?6.6:8.6;

  // the 5 A graph MACE passes messages on, dashed so it reads as a graph
  ctx.save();ctx.setLineDash([2,4]);ctx.lineCap='round';
  FRAG.e2.forEach(([j,k])=>{const a=L[j],c=L[k];
    ctx.globalAlpha=sub(build,.4,1)*.3;ctx.strokeStyle=P.se;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(c.x,c.y);ctx.stroke();});
  FRAG.e1.forEach(([c0,j])=>{const a=L[c0],c=L[j];
    ctx.globalAlpha=build*.55;ctx.strokeStyle=P.se;ctx.lineWidth=1.3;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(c.x,c.y);ctx.stroke();});
  ctx.restore();

  // covalent bonds as sticks, each half taking its own atom's colour
  ctx.save();ctx.lineCap='round';
  FRAG.bonds.forEach(([i,j])=>{
    const a=L[i],c=L[j],mx=(a.x+c.x)/2,my=(a.y+c.y)/2;
    ctx.globalAlpha=build;
    [[a,ATOMS[i].el],[c,ATOMS[j].el]].forEach(([q,el])=>{
      ctx.strokeStyle=hex(P.ink,.22);ctx.lineWidth=narrow?4.4:5.8;
      ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(mx,my);ctx.stroke();
      ctx.strokeStyle=EC[el];ctx.lineWidth=narrow?2.6:3.4;
      ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(mx,my);ctx.stroke();});
  });
  ctx.restore();

  // the receptive field grows one shell per layer
  [[l1,.58],[l2,1.0]].forEach(([u,f])=>{
    if(u<=0)return;
    ctx.save();ctx.globalAlpha=(1-sub(u,.75,1))*.28;ctx.strokeStyle=P.se;
    ctx.setLineDash([3,4]);ctx.lineWidth=1.2;
    ctx.beginPath();ctx.arc(L[FRAG.c].x,L[FRAG.c].y,f*(narrow?62:80)*ease(u),0,7);
    ctx.stroke();ctx.restore();});

  // atoms, coloured by element, drawn back to front
  const order=FRAG.ids.slice().sort((i,j)=>L[i].z-L[j].z);
  order.forEach(i=>{
    const q=L[i], centre=i===FRAG.c;
    const r=(centre?CR:AR)*(centre?1+.10*Math.sin(p*46)*ease(l1)*(1-ease(l2)):1);
    ctx.save();ctx.globalAlpha=build;
    ctx.fillStyle=centre?P.se:EC[ATOMS[i].el];
    ctx.strokeStyle=P.sf;ctx.lineWidth=narrow?1.6:2;
    ctx.beginPath();ctx.arc(q.x,q.y,r,0,7);ctx.fill();ctx.stroke();ctx.restore();});

  // messages travelling inward, one hop per layer
  const flow=(edges,u,rev)=>{
    if(u<=0||u>=1)return;
    edges.forEach(([a0,b0],n)=>{
      const st=(n%5)*.06, w=sub(u,st,st+.55);
      if(w<=0||w>=1)return;
      const A=L[rev?b0:a0], B=L[rev?a0:b0], e=ease(w);
      ctx.save();ctx.globalAlpha=Math.sin(w*Math.PI);ctx.fillStyle=P.se;
      ctx.beginPath();ctx.arc(lerp(B.x,A.x,e),lerp(B.y,A.y,e),narrow?2.4:3.1,0,7);
      ctx.fill();ctx.restore();});
  };
  flow(FRAG.e2,sub(l2,0,.5),true);
  flow(FRAG.e1,l1,false);
  flow(FRAG.e1,sub(l2,.45,1),false);

  // element key, so the colours mean something
  if(build>.5){
    let kx=box.x+(narrow?0:4);
    [['N',0],['C',1],['O',2]].forEach(([n,el])=>{
      ctx.save();ctx.globalAlpha=sub(build,.5,1)*.9;ctx.fillStyle=EC[el];
      ctx.beginPath();ctx.arc(kx+3,box.y+(narrow?8:10),3.2,0,7);ctx.fill();ctx.restore();
      txt(n,kx+9,box.y+(narrow?11:13),{s:narrow?8.5:9.5,c:P.mut,al:sub(build,.5,1)});
      kx+=narrow?24:28;});
    txt('His-Arg at P7-P8',box.x+box.w,box.y+(narrow?11:13),
      {s:narrow?8.5:9.5,c:P.mut,ta:'right',al:sub(build,.5,1)});
  }

  // what the animation to the left is doing, in words
  if(!narrow){
    const tx=pan.x+pan.w*.60, lines=[
      ['every atom starts as its element and position',build],
      ['layer 1  \u2014  sum messages from atoms within 5 \u00c5',l1],
      ['layer 2  \u2014  those neighbours have summed theirs',l2],
      ['so two layers see 10 \u00c5 without ever going global',sub(p,.46,.58)]];
    lines.forEach(([s,u],i)=>{
      const a=sub(u,.05,.45), y=box.y+42+i*34;
      ctx.save();ctx.globalAlpha=a;ctx.fillStyle=i<3?P.se:P.ink;
      ctx.beginPath();ctx.arc(tx+3,y-4,i<3?3.2:0,0,7);ctx.fill();ctx.restore();
      txt(s,tx+(i<3?13:0),y,{s:i<3?11.5:12,c:i<3?P.ink:P.ink,f:'s',w:i<3?400:600,al:a});
    });
  }

  const ly=box.y+box.h+(narrow?14:17);
  txt(l2>.1?'layer 2 · receptive field 10 Å':(l1>.1?'layer 1 · receptive field 5 Å':'invariant scalars per atom'),
    pan.x,ly,{s:narrow?10:11.5,c:P.se,w:600,al:build});
  txt(narrow?'pooled into three blocks':'pooled into node, edge and energy blocks',
    pan.x+pan.w,ly+(narrow?13:0),{s:narrow?9:10.5,c:P.mut,ta:'right',al:sub(p,.5,.62)});

  // the three descriptor blocks, to scale by width
  const bw=sub(p,.48,.58);
  if(bw>0){
    const by=ly+(narrow?12:14), tot=MACE.dims.node+MACE.dims.edge+MACE.dims.energy;
    let x=pan.x;
    [['node',MACE.dims.node],['edge',MACE.dims.edge],['energy',MACE.dims.energy]].forEach(([n,dm],i)=>{
      const w=pan.w*(dm/tot)*ease(bw);
      ctx.save();ctx.globalAlpha=bw*(i===2?.95:.5);ctx.fillStyle=i===2?P.se:hex(P.se,.45);
      ctx.fillRect(x,by,Math.max(1,w-2),narrow?9:11);ctx.restore();
      if(w>(narrow?74:96))txt(n+'  '+dm.toLocaleString(),x+2,by+(narrow?20:23),
        {s:narrow?8.5:9.5,c:P.mut,al:bw});
      x+=w;});
  }
  maceStrip(p,pan,ly+(narrow?54:64),.50);
}

/* explicit answer to whether the atomistic blocks earned their place */
function maceStrip(p,pan,top,at){
  const a=sub(p,at,at+.12); if(a<=0)return;
  ctx.save();ctx.globalAlpha=a*.6;ctx.strokeStyle=P.ln;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pan.x,top);ctx.lineTo(pan.x+pan.w,top);ctx.stroke();ctx.restore();
  txt('DO MACE FEATURES IMPROVE HELD-OUT PEARSON r?',pan.x,top-7,
    {s:narrow?8.5:10,c:P.mut,w:500,al:a});

  const cols=[
    ['Boltz-2 only',   MACE.boltz,     'reference',   P.bz],
    ['Boltz-2 + MACE', MACE.boltzNode, '↓ 0.032',      P.se],
    ['MACE only',      MACE.node,      'train 0.768', P.wrn]];
  const w=pan.w/3;
  cols.forEach(([lab,v,sub2,col],i)=>{
    const aa=sub(p,at+i*.03,at+.1+i*.03), x=pan.x+i*w;
    txt(lab,x,top+(narrow?16:19),{s:narrow?9.5:11,c:P.ink,w:600,al:aa});
    txt(v.toFixed(3),x,top+(narrow?40:46),{s:narrow?17:24,w:600,c:col,al:aa});
    txt(sub2,narrow?x:x+84,top+(narrow?56:46),{s:narrow?9:10.5,c:P.mut,al:aa});
  });
  const g=sub(p,at+.14,at+.26);
  if(g>0){
    const ny=top+(narrow?76:70);
    ctx.save();ctx.globalAlpha=g;ctx.fillStyle=P.wrn;
    ctx.beginPath();ctx.moveTo(pan.x,ny-7);ctx.lineTo(pan.x+6,ny-3.5);ctx.lineTo(pan.x,ny);
    ctx.closePath();ctx.fill();ctx.restore();
    txt(narrow?'NO — MACE does not improve test r.'
              :'NO — MACE does not improve the held-out result.',
      pan.x+14,ny,{s:narrow?12:18,c:P.ink,f:'s',w:600,al:g});
    txt(narrow?'combined 0.603 → 0.571 · MACE-only test 0.020'
              :'Boltz-2 + MACE drops 0.603 → 0.571; MACE alone overfits (train 0.768, test 0.020).',
      pan.x+14,ny+(narrow?15:19),{s:narrow?9:11.5,c:P.mut,f:'s',al:g});
  }
}

/* ---------- scene 5: the feature spaces ---------- */
const UCOLS=[['boltz','Boltz-2 complex','1,547-d','bz'],
             ['node','MACE node','11,520-d','se'],
             ['edge','MACE edge','14,450-d','se']];

function s5(p,pan){
  txt(narrow?'FOUNDATION-MODEL FEATURE SPACES':'INTERROGATING FOUNDATION-MODEL FEATURE SPACES',pan.x,pan.y+(narrow?16:22),{s:narrow?15:22,w:600,c:P.ink,f:'d'});
  txt(narrow?'Boltz-2 and MACE · same complexes':'Boltz-2 and MACE views of the same complexes · learned before the half-life task',
    pan.x,pan.y+(narrow?32:42),{s:narrow?9.5:11,c:P.mut});

  const GAP=narrow?0:20, cw=narrow?pan.w:(pan.w-2*GAP)/3;
  const top=pan.y+(narrow?80:100), ph=narrow?150:Math.min(cw,240);
  const SPC={t:[P.mut,.22,1.3],v:[P.bz,.75,1.6],e:[P.wrn,.85,1.7]};
  const show=narrow?(p<.38?0:p<.68?1:2):-1;      // phone takes them one at a time

  UCOLS.forEach(([tag,name,dim,ck],ci)=>{
    if(narrow&&show!==ci)return;
    const x=narrow?pan.x:pan.x+ci*(cw+GAP);
    const u=sub(p,.06+ci*(narrow?.30:.17),.42+ci*(narrow?.30:.17));
    if(u<=0)return;
    txt(name,x,top-(narrow?26:32),{s:narrow?16:22,w:600,c:P[ck],f:'d',al:u});
    txt(dim,x,top-(narrow?11:14),{s:narrow?9:10.5,c:P.mut,al:u*.9});

    ctx.save();ctx.globalAlpha=u*.5;ctx.strokeStyle=P.ln;ctx.lineWidth=1;
    rr(x,top,cw,ph,5);ctx.stroke();ctx.restore();

    const D=UMAP[tag], n=D.s.length, pad=narrow?10:12;
    const sx=(cw-2*pad)/999, sy=(ph-2*pad)/999;
    // train first, so the held-out points are not buried under it
    for(const want of ['t','v','e']){
      const [col,al,r]=SPC[want];
      ctx.save();ctx.fillStyle=col;
      for(let i=0;i<n;i++){
        if(D.s[i]!==want)continue;
        if(sub(u,(i/n)*.5,.3+(i/n)*.5)<=0)continue;
        ctx.globalAlpha=al*u;
        ctx.beginPath();
        ctx.arc(x+pad+D.xy[i*2]*sx, top+pad+D.xy[i*2+1]*sy, r, 0, 7);
        ctx.fill();
      }
      ctx.restore();
    }
  });

  // legend
  const ly=top+ph+(narrow?22:26), lg=sub(p,.20,.34);
  if(lg>0){
    let lx=pan.x;
    [['train','t'],['val','v'],['test','e']].forEach(([lbl,k])=>{
      const [col,al,r]=SPC[k];
      ctx.save();ctx.globalAlpha=lg*Math.max(al,.5);ctx.fillStyle=col;
      ctx.beginPath();ctx.arc(lx+3,ly-4,3.4,0,7);ctx.fill();ctx.restore();
      txt(lbl,lx+10,ly,{s:narrow?9:10.5,c:P.mut,al:lg});
      lx+=narrow?48:58;});
    txt('held out by sequence-similarity cluster',pan.x+pan.w,ly,
      {s:narrow?8.5:10,c:P.mut,ta:'right',al:lg});
  }
}

/* ---------- compose ---------- */
const CUT=[.26,.46,.66,.84];
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
let tab='C2';
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
