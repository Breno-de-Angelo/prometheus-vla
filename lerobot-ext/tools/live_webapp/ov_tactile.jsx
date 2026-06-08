/* ov_tactile_v4.jsx — Dex3-1 tactile, silhueta robótica (blocos retos).
   Corrigido para mapear os 33 canais reais (incluindo palma)
   Vetor [33]: Thumb(4+3), Middle(4+3), Index(4+3), Palm(4+4+4)
   Exposto: window.TactilePanel
 */
(function () {
  const { useRef, useEffect } = React;

  const FINGER_W = 42;
  const SEG_H = 70;
  const PALM_BOX = [94, 198, 132, 132];
  const [PALM_X, PALM_Y, PALM_W, PALM_H] = PALM_BOX;
  const MIDDLE_X = PALM_X + 21;
  const INDEX_X  = PALM_X + PALM_W - 21;
  const THUMB_X  = PALM_X + PALM_W / 2;
  const BOX = [320, 470];

  const FINGERS = [
    {
      label:'Indicador', lx:INDEX_X, ly:38, al:'center',
      bx:INDEX_X, by:PALM_Y, tx:INDEX_X, ty:PALM_Y - 2 * SEG_H,
      segs:[
        { pads:[{ indices:[14,15,16,17], layout:'grid2x2' }] },
        { pads:[{ indices:[18,19,20], layout:'pad3' }] },
      ],
    },
    {
      label:'Médio', lx:MIDDLE_X, ly:38, al:'center',
      bx:MIDDLE_X, by:PALM_Y, tx:MIDDLE_X, ty:PALM_Y - 2 * SEG_H,
      segs:[
        { pads:[{ indices:[7,8,9,10], layout:'grid2x2' }] },
        { pads:[{ indices:[11,12,13], layout:'pad3' }] },
      ],
    },
    {
      label:'Polegar', lx:THUMB_X, ly:PALM_Y + PALM_H + 2 * SEG_H + 16, al:'center',
      bx:THUMB_X, by:PALM_Y + PALM_H, tx:THUMB_X, ty:PALM_Y + PALM_H + 2 * SEG_H,
      segs:[
        { pads:[{ indices:[0,1,2,3], layout:'grid2x2' }] },
        { pads:[{ indices:[4,5,6], layout:'pad3' }] },
      ],
    },
  ];

  function segEnds(f, i){
    const n=f.segs.length, dx=f.tx-f.bx, dy=f.ty-f.by, L=Math.hypot(dx,dy)||1, ax=dx/L, ay=dy/L;
    const t0=i/n, t1=(i+1)/n;
    return {
      sx:f.bx+ax*L*t0, sy:f.by+ay*L*t0,
      ex:f.bx+ax*L*t1, ey:f.by+ay*L*t1,
    };
  }

  function boxSeg(ctx, bx,by, tx,ty, w, sel){
    const dx=tx-bx, dy=ty-by, ax=dx/(Math.hypot(dx,dy)||1), ay=dy/(Math.hypot(dx,dy)||1), px=-ay, py=ax, hw=w/2;
    ctx.beginPath();
    ctx.moveTo(bx+px*hw, by+py*hw);
    ctx.lineTo(tx+px*hw, ty+py*hw);
    ctx.lineTo(tx-px*hw, ty-py*hw);
    ctx.lineTo(bx-px*hw, by-py*hw);
    ctx.closePath();
    const g=ctx.createLinearGradient(bx+px*hw,by+py*hw, bx-px*hw,by-py*hw);
    g.addColorStop(0,'#2b323d'); g.addColorStop(.5,'#161b23'); g.addColorStop(1,'#0a0d12');
    ctx.fillStyle=g; ctx.fill();
    ctx.strokeStyle= sel?'rgba(56,189,248,.45)':'rgba(120,140,166,.26)'; ctx.lineWidth=1.2; ctx.stroke();
  }

  function taxel(ctx,X,Y,R,v){
    ctx.beginPath(); ctx.arc(X,Y,R,0,7); ctx.fillStyle='rgba(150,182,216,.34)'; ctx.fill();
    if (v>0.03){
      const g=ctx.createRadialGradient(X,Y,0,X,Y,R*2.7);
      g.addColorStop(0,`rgba(235,248,255,${0.92*v+0.08})`);
      g.addColorStop(.4,`rgba(70,165,255,${0.8*v})`);
      g.addColorStop(1,'rgba(38,110,230,0)');
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(X,Y,R*2.7,0,7); ctx.fill();
      ctx.beginPath(); ctx.arc(X,Y,R*0.55,0,7); ctx.fillStyle=`rgba(255,255,255,${0.6*v+0.12})`; ctx.fill();
    }
  }

  const gv = (vals,i)=> Math.max(0,Math.min(1, vals?(vals[i]||0):0));

  function padPts(sx,sy,ex,ey,w,layout,at){
    const dx=ex-sx, dy=ey-sy, L=Math.hypot(dx,dy)||1, ax=dx/L, ay=dy/L, nx=-ay, ny=ax;
    const t=at==null?0.5:at;
    const cx=sx+ax*L*t, cy=sy+ay*L*t;
    const p=(u,v)=>[cx+ax*u+nx*v, cy+ay*u+ny*v];
    if (layout==='grid2x2'){
      const u=L*0.18, v=w*0.18;
      return [p(-u,-v), p(u,-v), p(-u,v), p(u,v)];
    }
    if (layout==='pad3'){
      const u=L*0.17, v=w*0.17;
      return [p(-u,-v), p(u,0), p(-u,v)];
    }
    return [];
  }

  function drawSegTaxels(ctx, TX, TY, SC, sx,sy,ex,ey, seg, vals){
    const R=SC(5.5);
    for (const pad of seg.pads){
      const pts=padPts(sx,sy,ex,ey,FINGER_W,pad.layout,pad.at);
      pad.indices.forEach((idx,i)=>{
        if (!pts[i]) return;
        taxel(ctx, TX(pts[i][0]), TY(pts[i][1]), R, gv(vals,idx));
      });
    }
  }

  function drawPalm(ctx, TX, TY, SC, vals) {
    const R=SC(5.5);
    const drawGrid = (cx, cy, indices) => {
        const u = SC(9.5), v = SC(9.5); // Slightly smaller internal grid to highlight separation
        const pts = [
            [cx-v, cy+u], [cx+v, cy+u],
            [cx-v, cy-u], [cx+v, cy-u]
        ];
        indices.forEach((idx, i) => {
            taxel(ctx, pts[i][0], pts[i][1], R, gv(vals, idx));
        });
    };
    // Draw all 3 on the same horizontal line, perfectly spaced
    const py = PALM_Y + 45;
    // ID 6 (Palm near Middle finger) - left side
    drawGrid(TX(MIDDLE_X), TY(py), [21,22,23,24]);
    // ID 8 (Palm near Thumb) - center
    drawGrid(TX(THUMB_X), TY(py), [29,30,31,32]);
    // ID 7 (Palm near Index) - right side
    drawGrid(TX(INDEX_X), TY(py), [25,26,27,28]);
  }

  function draw(ctx, W, H, DPR, vals, sel){
    ctx.save(); ctx.scale(DPR,DPR); ctx.clearRect(0,0,W,H);
    const s=Math.min(W/BOX[0], H/BOX[1]);
    const ox=(W-BOX[0]*s)/2, oy=(H-BOX[1]*s)/2;
    const TX=x=>ox+x*s, TY=y=>oy+y*s, SC=v=>v*s;
    ctx.lineJoin='round'; ctx.lineCap='round';

    const [px,py,pw,ph]=PALM_BOX;
    ctx.beginPath(); ctx.rect(TX(px), TY(py), SC(pw), SC(ph));
    const pg=ctx.createLinearGradient(TX(px),TY(py),TX(px+pw),TY(py+ph));
    pg.addColorStop(0,'#2b323d'); pg.addColorStop(.4,'#161b23'); pg.addColorStop(1,'#0a0d12');
    ctx.fillStyle=pg; ctx.fill();
    ctx.strokeStyle= sel?'rgba(56,189,248,.4)':'rgba(120,140,166,.24)'; ctx.lineWidth=1.3; ctx.stroke();
    ctx.font=`600 10px JetBrains Mono, monospace`; ctx.fillStyle='rgba(207,224,242,.9)';
    ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText('Palma', TX(px)+SC(8), TY(py)+SC(7));

    drawPalm(ctx, TX, TY, SC, vals);

    for (const f of FINGERS){
      f.segs.forEach((seg,i)=>{
        const {sx,sy,ex,ey}=segEnds(f,i);
        boxSeg(ctx, TX(sx),TY(sy), TX(ex),TY(ey), SC(FINGER_W), sel);
        drawSegTaxels(ctx, TX, TY, SC, sx,sy,ex,ey, seg, vals);
      });
      ctx.font=`600 11px JetBrains Mono, monospace`; ctx.fillStyle='#cfe0f2';
      ctx.textAlign=f.al; ctx.textBaseline='alphabetic';
      ctx.fillText(f.label, TX(f.lx), TY(f.ly));
    }
    ctx.restore();
  }

  function HandTactile({ side, stream, frame }) {
    const cvRef=useRef(null), rafRef=useRef(0), ref=useRef({});
    const fr = stream && stream.taxels ? Math.max(0,Math.min(stream.taxels.length-1,Math.round(frame))) : 0;
    const vals = stream && stream.taxels ? stream.taxels[fr] : null;
    const hasData = !!vals && vals.some(v=>v>0.001);
    ref.current={vals};
    useEffect(()=>{
      const cv=cvRef.current, ctx=cv.getContext('2d');
      const DPR=Math.min(2,window.devicePixelRatio||1); let W=0,H=0;
      function resize(){ const r=cv.getBoundingClientRect(); W=r.width;H=r.height;
        cv.width=Math.round(W*DPR); cv.height=Math.round(H*DPR); }
      resize(); const ro=new ResizeObserver(resize); ro.observe(cv);
      function loop(){ draw(ctx,W,H,DPR,ref.current.vals,true); rafRef.current=requestAnimationFrame(loop); }
      rafRef.current=requestAnimationFrame(loop);
      return ()=>{ cancelAnimationFrame(rafRef.current); ro.disconnect(); };
    },[]);
    return (
      <div style={{flex:'1 1 0', minWidth:0, display:'flex', flexDirection:'column',
        background:'#11161f', border:'1px solid rgba(56,189,248,.5)', borderRadius:9, padding:'11px 12px', gap:8}}>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
          <span style={{font:'600 10px JetBrains Mono,monospace', color:'#e6edf6', letterSpacing:'.03em'}}>MÃO DIR · R</span>
          <span style={{font:'600 9px JetBrains Mono,monospace', color:hasData?'#34d399':'#48566a'}}>{hasData?'contato':'— sem sinal'}</span>
        </div>
        <div style={{flex:'1 1 0', minHeight:0}}>
          <canvas ref={cvRef} style={{width:'100%', height:'100%', display:'block'}} />
        </div>
        {!hasData && (
          <div style={{font:'9px JetBrains Mono,monospace', color:'#48566a', textAlign:'center'}}>
            7 links · 33 taxels — aguardando sensor
          </div>
        )}
      </div>
    );
  }

  function TactilePanel({ hands, frame, group, palette='heat' }) {
    return (
      <div className="panel tactile-panel">
        <div className="panel-head">
          <span className="ph-label">TÁTIL · DEX3-1 · sensores por área</span>
          <span className="ph-meta mono">mão dir. · azul→branco ∝ força</span>
        </div>
        <div style={{display:'flex', padding:'10px', alignItems:'stretch', height:'100%', boxSizing:'border-box'}}>
          <HandTactile side="right" stream={hands.right} frame={frame} />
        </div>
      </div>
    );
  }

  window.TactilePanel = TactilePanel;
})();
