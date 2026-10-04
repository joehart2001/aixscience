// Ca-only secondary structure, P-SEA style: score each window against ideal
// helix and ideal strand geometry and take the better fit.
function assignSS(ca){
  const n=ca.length, d=(i,j)=>Math.hypot(ca[i][0]-ca[j][0],ca[i][1]-ca[j][1],ca[i][2]-ca[j][2]);
  const H={d2:5.5,d3:5.3,d4:6.2}, E={d2:6.7,d3:9.9,d4:12.4};
  const TOL={d2:1.0,d3:1.1,d4:1.3};
  const ss=new Array(n).fill('C');
  const fit=(i,T)=>{
    let s=0;
    for(const k of ['d2','d3','d4']){
      const j=i+(+k[1]); if(j>=n)return 1e9;
      s+=Math.pow((d(i,j)-T[k])/TOL[k],2);
    }
    return s/3;
  };
  for(let i=0;i+4<n;i++){
    const h=fit(i,H), e=fit(i,E);
    if(Math.min(h,e)>1)continue;                 // neither geometry fits
    const t=h<e?'H':'E';
    for(let k=i;k<=i+4;k++) if(ss[k]==='C') ss[k]=t;
  }
  // runs below the minimum length for that element are not real
  const MIN={H:6,E:3};
  for(let pass=0;pass<2;pass++){
    let s=0;
    for(let i=1;i<=n;i++) if(i===n||ss[i]!==ss[s]){
      const t=ss[s], L=i-s;
      if(t!=='C'&&L<MIN[t]) for(let k=s;k<i;k++) ss[k]='C';
      s=i;
    }
  }
  return ss.join('');
}
if(typeof module!=='undefined')module.exports={assignSS};
