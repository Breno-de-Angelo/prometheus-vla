/* ov_traj.jsx — 3D end-effector trajectory (custom canvas, no deps)
   Orbit camera, ground grid, time-gradient executed path, ghost target path,
   and an animated arm skeleton at the current frame.
   Exposes window.Trajectory3D
*/
(function () {
  const { useRef, useEffect } = React;

  function rotPt(p, az, el) {
    // azimuth about Z (up), elevation tilt
    const ca=Math.cos(az), sa=Math.sin(az);
    let x = p[0]*ca - p[1]*sa;
    let y = p[0]*sa + p[1]*ca;
    let z = p[2];
    const ce=Math.cos(el), se=Math.sin(el);
    const y2 = y*ce - z*se;
    const z2 = y*se + z*ce;
    return [x, y2, z2];
  }

  function Trajectory3D(props) {
    const { kin, kinTarget, frame, accent = '#38bdf8', bbox, compact } = props;
    const canvasRef = useRef(null);
    const stateRef = useRef({ az: -0.9, el: 0.55, dist: 2.7, drag: null });
    const rafRef = useRef(0);
    const drawRef = useRef(null);
    const propRef = useRef(props);
    propRef.current = props;

    // draw on every React commit (frame/tweak change) so the canvas is correct
    // even when rAF is throttled (background tabs / screenshot capture)
    useEffect(() => { if (drawRef.current) drawRef.current(); });

    useEffect(() => {
      const cv = canvasRef.current;
      const ctx = cv.getContext('2d');
      let W = 0, H = 0, DPR = Math.min(2, window.devicePixelRatio || 1);

      function resize() {
        const r = cv.getBoundingClientRect();
        W = r.width; H = r.height;
        cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
      }
      resize();
      const ro = new ResizeObserver(resize); ro.observe(cv);

      function project(p) {
        const s = stateRef.current;
        const bb = propRef.current.bbox;
        // normalize around center, scale by bbox radius
        const c = bb.c, scale = 1 / bb.r;
        let q = [(p[0]-c[0])*scale, (p[1]-c[1])*scale, (p[2]-c[2])*scale];
        q = rotPt(q, s.az, s.el);
        const f = 2.2, d = s.dist;
        const zc = d + q[1];               // camera looks along +Y after rot
        const persp = f / Math.max(0.15, zc);
        return [ W/2 + q[0]*persp*H*0.42, H/2 - q[2]*persp*H*0.42, zc ];
      }

      function draw() {
        const P = propRef.current;
        const s = stateRef.current;
        ctx.save();
        ctx.scale(DPR, DPR);
        ctx.clearRect(0,0,W,H);

        // ---- ground grid (plane at min z of bbox) ----
        const bb = P.bbox, c = bb.c, scale = 1/bb.r;
        const zfloor = (bb.lo[2]-c[2])*scale;
        const g = 4, step = 0.5;
        ctx.lineWidth = 1;
        for (let i=-g;i<=g;i++){
          ctx.strokeStyle = i===0 ? 'rgba(120,140,165,.28)' : 'rgba(90,105,125,.12)';
          let a = project([ c[0]+ (i*step)/scale, c[1]+(-g*step)/scale, bb.lo[2] ]);
          let b = project([ c[0]+ (i*step)/scale, c[1]+( g*step)/scale, bb.lo[2] ]);
          ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
          a = project([ c[0]+(-g*step)/scale, c[1]+(i*step)/scale, bb.lo[2] ]);
          b = project([ c[0]+( g*step)/scale, c[1]+(i*step)/scale, bb.lo[2] ]);
          ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
        }

        const fr = Math.min(P.kin.ee.length-1, Math.max(0, Math.round(P.frame)));

        // ---- ghost target path (from action FK) ----
        if (P.kinTarget) {
          ctx.strokeStyle = 'rgba(245,158,11,.30)';
          ctx.lineWidth = 1.4; ctx.setLineDash([4,4]);
          ctx.beginPath();
          for (let i=0;i<P.kinTarget.ee.length;i++){ const p=project(P.kinTarget.ee[i]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }
          ctx.stroke(); ctx.setLineDash([]);
        }

        // ---- executed path, time gradient, traveled portion brighter ----
        const ee = P.kin.ee;
        for (let i=1;i<ee.length;i++){
          const p0=project(ee[i-1]), p1=project(ee[i]);
          const traveled = i<=fr;
          const t = i/ee.length;
          const col = OV.turbo(t);
          ctx.strokeStyle = traveled
            ? `rgba(${col[0]},${col[1]},${col[2]},0.95)`
            : `rgba(${col[0]},${col[1]},${col[2]},0.18)`;
          ctx.lineWidth = traveled ? 2.6 : 1.4;
          ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]); ctx.stroke();
        }

        // ---- arm skeleton at current frame ----
        const sk = P.kin.skel[fr];
        const joints = [sk.shoulder, sk.elbow, sk.wrist, sk.ee];
        const proj = joints.map(project);
        // torso stub
        ctx.strokeStyle = 'rgba(150,165,185,.5)'; ctx.lineWidth = 5; ctx.lineCap='round';
        const torsoTop = project([0,0,sk.shoulder[2]+0.12]);
        ctx.beginPath(); ctx.moveTo(torsoTop[0],torsoTop[1]); ctx.lineTo(proj[0][0],proj[0][1]); ctx.stroke();
        // arm links
        ctx.strokeStyle = accent; ctx.lineWidth = 5;
        ctx.beginPath();
        for (let i=0;i<proj.length;i++){ i?ctx.lineTo(proj[i][0],proj[i][1]):ctx.moveTo(proj[i][0],proj[i][1]); }
        ctx.stroke();
        // joints
        for (let i=0;i<proj.length;i++){
          ctx.fillStyle = i===proj.length-1 ? '#fff' : '#0a0e14';
          ctx.strokeStyle = accent; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(proj[i][0],proj[i][1], i===proj.length-1?5:4, 0, 7); ctx.fill(); ctx.stroke();
        }
        // EE glow marker
        const ep = proj[3];
        const grd = ctx.createRadialGradient(ep[0],ep[1],0, ep[0],ep[1],16);
        grd.addColorStop(0,'rgba(255,255,255,.5)'); grd.addColorStop(1,'rgba(255,255,255,0)');
        ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(ep[0],ep[1],16,0,7); ctx.fill();

        // axes gizmo (bottom-left)
        drawGizmo(ctx, 30, H-30, s.az, s.el);
        ctx.restore();
      }
      function loop(){ draw(); rafRef.current = requestAnimationFrame(loop); }

      function drawGizmo(ctx, ox, oy, az, el) {
        const axes = [[1,0,0,'#f6708a','x'],[0,1,0,'#7ee787','y'],[0,0,1,'#7dd3fc','z']];
        ctx.font = '9px JetBrains Mono, monospace';
        for (const [x,y,z,col,lab] of axes){
          let q = rotPt([x,y,z], az, el);
          const px = ox + q[0]*16, py = oy - q[2]*16;
          ctx.strokeStyle = col; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(px,py); ctx.stroke();
          ctx.fillStyle = col; ctx.fillText(lab, px+1, py+2);
        }
      }

      // pointer orbit
      function down(e){ stateRef.current.drag = { x:e.clientX, y:e.clientY }; }
      function move(e){
        const s = stateRef.current; if (!s.drag) return;
        s.az -= (e.clientX - s.drag.x)*0.008;
        s.el += (e.clientY - s.drag.y)*0.006;
        s.el = Math.max(-1.3, Math.min(1.45, s.el));
        s.drag = { x:e.clientX, y:e.clientY };
      }
      function up(){ stateRef.current.drag = null; }
      function wheel(e){ e.preventDefault(); const s=stateRef.current; s.dist = Math.max(1.4, Math.min(6, s.dist + e.deltaY*0.002)); }

      cv.addEventListener('pointerdown', down);
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
      cv.addEventListener('wheel', wheel, { passive:false });

      rafRef.current = requestAnimationFrame(loop);
      drawRef.current = draw;
      draw();
      return () => {
        cancelAnimationFrame(rafRef.current); ro.disconnect();
        cv.removeEventListener('pointerdown', down);
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        cv.removeEventListener('wheel', wheel);
      };
    }, []);

    return (
      <div className="traj-wrap">
        <canvas ref={canvasRef} className="traj-canvas" style={{cursor:'grab'}} />
        <div className="traj-legend">
          <span><i style={{background:accent}} />executado (FK·state)</span>
          <span><i style={{background:'#f59e0b',opacity:.7}} className="dash" />alvo (FK·action)</span>
        </div>
        <div className="traj-hint">arraste p/ orbitar · scroll p/ zoom</div>
      </div>
    );
  }

  window.Trajectory3D = Trajectory3D;
})();
