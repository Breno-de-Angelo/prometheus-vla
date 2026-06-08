/* ov_ramps.jsx — per-joint ramp charts (commanded / filtered / executed)
   Stacked small-multiples on one canvas. Synced playhead + hover readout.
   Exposes window.Ramps
*/
(function () {
  const { useRef, useEffect, useState } = React;
  const COL = { cmd:'#22d3ee', filt:'#f59e0b', exec:'#34d399' };

  function Ramps(props) {
    // props: tele {action,state,filtered}, frame, group(key in OV.LAYOUT),
    //        signals {cmd,filt,exec}, jointNames[], onScrub(frame), accent
    const canvasRef = useRef(null);
    const wrapRef = useRef(null);
    const propRef = useRef(props); propRef.current = props;
    const [hover, setHover] = useState(null); // {frame}
    const hoverRef = useRef(null); hoverRef.current = hover;
    const rafRef = useRef(0);
    const drawRef = useRef(null);
    // zoom de TEMPO (eixo X): fração do buffer mostrada na largura do painel.
    // 1 = buffer inteiro (mais comprimido) · <1 = janela menor recente (expande o detalhe).
    const xspanRef = useRef(1);
    useEffect(() => { if (drawRef.current) drawRef.current(); });

    useEffect(() => {
      const cv = canvasRef.current, ctx = cv.getContext('2d');
      const DPR = Math.min(2, window.devicePixelRatio||1);
      let W=0,H=0, geom=[];
      function resize(){ const r=cv.getBoundingClientRect(); W=r.width;H=r.height; cv.width=Math.round(W*DPR); cv.height=Math.round(H*DPR); }
      resize(); const ro=new ResizeObserver(resize); ro.observe(cv);

      function draw(){
        const P = propRef.current;
        const grp = OV.LAYOUT[P.group];
        const tele = P.tele;
        const F = tele.action.length;
        const dims = []; for (let i=0;i<grp.len;i++) dims.push(grp.start+i);
        const padL=86, padR=64, padT=8, padB=18, gap=10;
        const n = dims.length;
        const chartH = (H - padT - padB - gap*(n-1)) / n;
        geom = [];
        ctx.save(); ctx.scale(DPR,DPR); ctx.clearRect(0,0,W,H);
        const plotW = W - padL - padR;
        const fr = Math.max(0, Math.min(F-1, Math.round(P.frame)));
        const hv = hoverRef.current ? Math.max(0,Math.min(F-1,hoverRef.current.frame)) : null;
        // janela de tempo visível (zoom X): mostra os últimos `win` frames esticados na largura
        const win = Math.max(2, Math.min(F, Math.round(xspanRef.current * F)));
        const f0 = F - win;

        ctx.font = '10px JetBrains Mono, monospace';
        for (let i=0;i<n;i++){
          const d = dims[i];
          const y0 = padT + i*(chartH+gap);
          geom.push({ d, y0, h: chartH });
          // compute y-range across visible signals
          let mn=1e9,mx=-1e9;
          const series = [];
          if (P.signals.cmd)  series.push(['cmd', tele.action]);
          if (P.signals.filt && tele.filtered) series.push(['filt', tele.filtered]);
          if (P.signals.exec) series.push(['exec', tele.state]);
          for (const [,arr] of series) for (let f=f0;f<F;f++){ const v=arr[f][d]; if(v<mn)mn=v; if(v>mx)mx=v; }
          if (mn===mx){ mn-=0.1; mx+=0.1; }
          const pad=(mx-mn)*0.12; mn-=pad; mx+=pad;
          const X = f => padL + ((f-f0)/(win-1))*plotW;
          const Y = v => y0 + chartH - ((v-mn)/(mx-mn))*chartH;

          // panel bg
          ctx.fillStyle = 'rgba(255,255,255,.015)';
          ctx.fillRect(padL, y0, plotW, chartH);
          // zero line
          if (mn<0 && mx>0){ ctx.strokeStyle='rgba(255,255,255,.10)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(padL,Y(0)); ctx.lineTo(padL+plotW,Y(0)); ctx.stroke(); }
          // baseline border
          ctx.strokeStyle='rgba(255,255,255,.06)'; ctx.strokeRect(padL,y0,plotW,chartH);

          const masked = !grp.active;
          // draw signals (recortados ao gráfico p/ a curva ampliada não vazar no vizinho)
          ctx.save(); ctx.beginPath(); ctx.rect(padL, y0, plotW, chartH); ctx.clip();
          for (const [key,arr] of series){
            ctx.strokeStyle = masked ? 'rgba(120,135,155,.5)' : COL[key];
            ctx.globalAlpha = masked ? .6 : (key==='filt'?0.95:1);
            ctx.lineWidth = key==='exec'?1.8:1.4;
            if (key==='filt') ctx.setLineDash([3,3]); else ctx.setLineDash([]);
            ctx.beginPath();
            for (let f=f0;f<F;f++){ const x=X(f), y=Y(arr[f][d]); f===f0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
            ctx.stroke(); ctx.globalAlpha=1; ctx.setLineDash([]);
          }
          ctx.restore();
          // joint label
          ctx.fillStyle = masked ? '#5b6675' : '#cdd9e6';
          ctx.textAlign='left';
          const nm = (P.jointNames && P.jointNames[i]) || ('dim '+d);
          ctx.fillText(nm, 8, y0+12);
          ctx.fillStyle='#48566a'; ctx.fillText('d'+d, 8, y0+24);
          if (masked){ ctx.fillStyle='#f59e0b'; ctx.fillText('MASK', 8, y0+chartH-2); }

          // current-frame readouts (right gutter)
          ctx.textAlign='right';
          let ry = y0+12;
          const rd=(key,arr)=>{ if(!P.signals[key]||(key==='filt'&&!tele.filtered))return; ctx.fillStyle=masked?'#6b7888':COL[key]; ctx.fillText(arr[fr][d].toFixed(3), W-8, ry); ry+=12; };
          rd('cmd',tele.action); rd('filt',tele.filtered); rd('exec',tele.state);

          // playhead
          const px = X(fr);
          ctx.strokeStyle = P.accent || '#38bdf8'; ctx.globalAlpha=.85; ctx.lineWidth=1;
          ctx.beginPath(); ctx.moveTo(px,y0); ctx.lineTo(px,y0+chartH); ctx.stroke(); ctx.globalAlpha=1;
          // hover line
          if (hv!=null){ const hx=X(hv); ctx.strokeStyle='rgba(255,255,255,.25)'; ctx.setLineDash([2,3]); ctx.beginPath(); ctx.moveTo(hx,y0); ctx.lineTo(hx,y0+chartH); ctx.stroke(); ctx.setLineDash([]); }

          ctx.textAlign='left';
        }
        // grade + rótulos por segundo (tempo até "agora", à direita) — ajuda a ver os recortes no tempo
        const fps = tele.fps||30;
        const yTop = padT, yBot = padT + (n-1)*(chartH+gap) + chartH;
        const winSec = (win-1)/fps;
        const stepSec = winSec>20 ? 5 : (winSec>8 ? 2 : 1);  // evita rótulos colados quando comprimido
        ctx.strokeStyle='rgba(255,255,255,.06)'; ctx.lineWidth=1;
        ctx.fillStyle='#48566a'; ctx.font='9px JetBrains Mono, monospace'; ctx.textAlign='center';
        for (let s=0; s<=winSec+1e-3; s+=stepSec){
          const f=(F-1)-s*fps; if (f<f0) break;
          const x=padL+((f-f0)/(win-1))*plotW;
          ctx.beginPath(); ctx.moveTo(x,yTop); ctx.lineTo(x,yBot); ctx.stroke();
          ctx.fillText(s===0?'agora':('-'+s+'s'), x, H-6);
        }
        // indicador da janela de tempo
        ctx.fillStyle='rgba(56,189,248,.95)'; ctx.font='600 10px JetBrains Mono, monospace'; ctx.textAlign='left';
        ctx.fillText('janela '+winSec.toFixed(1)+'s · scroll = comprimir/expandir · 2× = tudo', padL+6, padT+11);
        ctx.restore();
      }
      function loop(){ draw(); rafRef.current=requestAnimationFrame(loop); }

      function toFrame(e){
        const P=propRef.current; const r=cv.getBoundingClientRect();
        const padL=86, padR=64; const plotW=r.width-padL-padR;
        const F=P.tele.action.length;
        const win=Math.max(2,Math.min(F,Math.round(xspanRef.current*F))); const f0=F-win;
        let u=(e.clientX-r.left-padL)/plotW; u=Math.max(0,Math.min(1,u));
        return f0+Math.round(u*(win-1));
      }
      // scroll = comprimir/expandir o eixo de TEMPO (X) · 2× clique = buffer inteiro
      function mmove(e){ setHover({frame:toFrame(e)}); }
      function mleave(){ setHover(null); }
      function wheel(e){
        e.preventDefault();
        // p/ cima = expande (janela menor, mais detalhe); p/ baixo = comprime (mais tempo na tela)
        const k = e.deltaY<0 ? 1/1.15 : 1.15;
        xspanRef.current = Math.max(0.04, Math.min(1, xspanRef.current*k));
      }
      function dbl(){ xspanRef.current=1; }

      cv.addEventListener('pointermove', mmove);
      cv.addEventListener('pointerleave', mleave);
      cv.addEventListener('wheel', wheel, { passive:false });
      cv.addEventListener('dblclick', dbl);
      rafRef.current=requestAnimationFrame(loop);
      drawRef.current=draw;
      draw();
      return ()=>{ cancelAnimationFrame(rafRef.current); ro.disconnect();
        cv.removeEventListener('pointermove',mmove);
        cv.removeEventListener('pointerleave',mleave);
        cv.removeEventListener('wheel',wheel); cv.removeEventListener('dblclick',dbl);
      };
    }, []);

    return <canvas ref={canvasRef} className="ramps-canvas" />;
  }

  window.Ramps = Ramps;
  window.RAMP_COL = COL;
})();
