/* inference_app.jsx — OmniView · Inference Live.
   Dashboard dedicado de inferência (estética "mission control"). Consome o mesmo
   websocket /ws (img + tele) e reusa window.OV (deriveFiltered) + window.Ramps.
   Layout 2×3: [RGB · Depth · Atenção] / [Sim 3ª · Sim 1ª · Recebido/Filtrado/Executado]
   + gauge de Grasp (%) no header. Sem bloco tátil.
   As cams do sim vêm do offline_sim_host (MJPEG em :8014 /global.mjpg e /head.mjpg). */
(function () {
  const { useState, useEffect, useRef, useMemo } = React;
  const FPS = 30, MAXLEN = 30 * FPS, ACCENT = '#22d3ee';
  // host MJPEG (3ª/1ª pessoa do sim) — mesmo hostname do dashboard, porta 8014
  const SIMCAM = `http://${location.hostname}:8014`;

  OV.LAYOUT = {
    leftArm:   { start: 0,  len: 7, active: false, label: 'Braço esq.', kind: 'arm' },
    rightArm:  { start: 7,  len: 7, active: true,  label: 'Braço dir.', kind: 'arm' },
    leftHand:  { start: 14, len: 7, active: false, label: 'Mão esq.',   kind: 'hand' },
    rightHand: { start: 21, len: 7, active: true,  label: 'Mão dir.',   kind: 'hand' },
  };
  OV.ARM_JOINTS = ['Shoulder·Pitch','Shoulder·Roll','Shoulder·Yaw','Elbow','Wrist·Roll','Wrist·Pitch','Wrist·Yaw'];
  OV.RHAND_JOINTS = ['Thumb·0','Thumb·1','Thumb·2','Index·0','Index·1','Middle·0','Middle·1'];
  OV.LHAND_JOINTS = ['Thumb·0','Thumb·1','Thumb·2','Middle·0','Middle·1','Index·0','Index·1'];

  function flat28(g) {
    const z7 = [0,0,0,0,0,0,0];
    return [].concat(g.leftArm||z7, g.rightArm||z7, g.leftHand||z7, g.rightHand||z7);
  }
  function dataUrlToBlobUrl(d) {
    const i = d.indexOf(','); const bin = atob(d.slice(i + 1));
    const b = new Uint8Array(bin.length); for (let j = 0; j < bin.length; j++) b[j] = bin.charCodeAt(j);
    return URL.createObjectURL(new Blob([b], { type: 'image/jpeg' }));
  }

  function useLiveData() {
    const histRef = useRef([]), graspRef = useRef({ right: 0, rightTrigger: 0, left: 0 });
    const graspHistRef = useRef([]), imgRef = useRef({}), metaRef = useRef({ episode: 0, frame: 0 });
    const lastTeleRef = useRef(0), phaseRef = useRef('unlocked');
    // taxas: cmd = comandos enviados ao robô (msgs 'tele', ~loop de controle);
    // attn = atualização do mapa de atenção (= taxa de inferência da VLA, msgs com attn_hm).
    const cmdHzRef = useRef({ last: performance.now(), n: 0, hz: 0 });
    const attnHzRef = useRef({ last: performance.now(), n: 0, hz: 0 });
    const chunkRef = useRef(null);   // último chunk previsto (N×7 juntas do braço, rad físico)
    const [, setTick] = useState(0); const [conn, setConn] = useState(false); const dirty = useRef(false);
    useEffect(() => {
      let ws, retry, alive = true;
      function connect() {
        ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onopen = () => setConn(true);
        ws.onclose = () => { setConn(false); if (alive) retry = setTimeout(connect, 1000); };
        ws.onerror = () => { try { ws.close(); } catch (e) {} };
        ws.onmessage = (ev) => {
          let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
          if (m.type === 'img') {
            const p = imgRef.current;
            const n = { rgb: p.rgb, depth: p.depth, attnHm: p.attnHm, rgbSize: m.rgbSize || p.rgbSize, depthMeta: m.depthMeta || p.depthMeta };
            if (m.rgb)     { n.rgb = dataUrlToBlobUrl(m.rgb);       if (p.rgb)    URL.revokeObjectURL(p.rgb); }
            if (m.depth)   { n.depth = dataUrlToBlobUrl(m.depth);   if (p.depth)  URL.revokeObjectURL(p.depth); }
            if (m.attn_hm) { n.attnHm = dataUrlToBlobUrl(m.attn_hm);if (p.attnHm) URL.revokeObjectURL(p.attnHm); attnHzRef.current.n++; }
            if (m.chunk) { chunkRef.current = m.chunk; }
            imgRef.current = n; dirty.current = true;
          } else if (m.type === 'tele') {
            lastTeleRef.current = performance.now();
            cmdHzRef.current.n++;
            if (m.robot && m.robot.phase) phaseRef.current = m.robot.phase;
            const h = histRef.current;
            h.push({ s28: flat28(m.state || {}), a28: flat28(m.action || {}) });
            if (h.length > MAXLEN) h.shift();
            if (m.grasp) {
              graspRef.current = m.grasp;
              const gh = graspHistRef.current; gh.push(+m.grasp.right || 0); if (gh.length > MAXLEN) gh.shift();
            }
            metaRef.current = { episode: m.episode || 0, frame: m.frame || 0 };
            dirty.current = true;
          }
        };
      }
      connect();
      let raf, lastT = 0;
      function tickHz(r, now) { if (now - r.last >= 1000) { r.hz = r.n * 1000 / (now - r.last); r.n = 0; r.last = now; } }
      function loop(ts) {
        const now = performance.now(); tickHz(cmdHzRef.current, now); tickHz(attnHzRef.current, now);
        if (dirty.current && (ts - lastT) >= 33) { dirty.current = false; lastT = ts; setTick(t => t + 1); }
        raf = requestAnimationFrame(loop);
      }
      raf = requestAnimationFrame(loop);
      return () => { alive = false; clearTimeout(retry); cancelAnimationFrame(raf); try { ws.close(); } catch (e) {} };
    }, []);
    return { histRef, graspRef, graspHistRef, imgRef, metaRef, lastTeleRef, phaseRef, conn, cmdHzRef, attnHzRef, chunkRef };
  }

  // ───────── header pieces ─────────
  const PHASE = { unlocked: ['', 'EXECUTANDO'], locked: ['off', 'BLOQUEADO'], softstart: ['warn', 'INICIANDO'] };
  function Led({ label, state }) {
    const cls = state === true ? 'on' : state === false ? 'off' : 'unk';
    return <span className={'led ' + cls}><span className="d" />{label}</span>;
  }
  function Sparkline({ hist, color }) {
    const W = 170, H = 18, n = hist.length;
    if (n < 2) return <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" />;
    // downsample p/ ~140 pts (polyline leve a 30Hz, sobretudo no Firefox)
    const MAXP = 140, stride = n > MAXP ? Math.ceil(n / MAXP) : 1;
    let pts = '';
    for (let i = 0; i < n; i += stride) {
      const x = (i / (n - 1)) * W, y = H - Math.max(0, Math.min(1, hist[i])) * (H - 2) - 1;
      pts += x.toFixed(1) + ',' + y.toFixed(1) + ' ';
    }
    return <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>;
  }
  function GraspGauge({ grasp, hist }) {
    const v = Math.max(0, Math.min(1, +grasp.right || 0)), pct = (v * 100).toFixed(0);
    const trig = Math.max(0, Math.min(1, +grasp.rightTrigger || 0));
    return (
      <div className="grasp">
        <div className="lbl"><b>Grasp</b><small>SQUEEZE · MÃO DIR</small></div>
        <div className="gauge">
          <div className="gmeter"><div className="fill" style={{ width: pct + '%' }} /><div className="ticks" /></div>
          <Sparkline hist={hist} color="#ff9e3d" />
          <span className="gtrig">trigger <b>{(trig * 100).toFixed(0)}%</b> · 0=aberto · 1=fechado</span>
        </div>
        <div className="gnum">{pct}<span style={{ fontSize: 12, opacity: .7 }}>%</span></div>
      </div>
    );
  }

  // ───────── media panels ─────────
  function MediaPanel({ tag, meta, amber, badge, children }) {
    return (
      <div className={'pnl' + (amber ? ' amber' : '')}>
        <div className="phead"><span className="ptag"><i />{tag}</span><span className="pmeta">{meta}</span></div>
        <div className="pbody">{children}{badge && <span className={'badge' + (amber ? ' amber' : '')}><span className="ld" />{badge}</span>}<div className="scan" /></div>
      </div>
    );
  }
  const Rgb = React.memo(function Rgb({ url, size }) {
    return <MediaPanel tag="RGB · head cam" meta={(size ? size[0] + '×' + size[1] : '—') + ' · entrada · agora'} badge="LIVE">
      {url ? <img className="feed" src={url} draggable="false" /> : <span className="wait">aguardando câmera</span>}
    </MediaPanel>;
  });
  // O modelo vê a imagem via resize_with_pad → 224×224 QUADRADO (aspecto preservado +
  // padding preto em cima/baixo p/ a head_camera 848×480). O heatmap 16×16 cobre esse
  // quadrado padado. Pra alinhar com o RGB largo: recorto a BANDA central (a imagem real)
  // e estico pra preencher. band = H/W (fração vertical ocupada); o resto é padding.
  const Attn = React.memo(function Attn({ url, hm, size }) {
    const W = (size && size[0]) || 848, H = (size && size[1]) || 480;
    const band = Math.min(1, H / W);                 // landscape: imagem ocupa essa fração vertical
    const heightPct = 100 / band;                    // estica a banda p/ preencher a altura
    const topPct = -((1 - band) / 2 / band) * 100;   // desloca p/ tirar o padding (simétrico)
    return <MediaPanel tag="Atenção · VLA" meta={hm ? 'heatmap · ~1s atrás (chunk)' : '—'} badge="ATN">
      {url ? (
        <div className="attwrap" style={{ aspectRatio: W + ' / ' + H }}>
          <img src={url} draggable="false" style={{ width: '100%', height: '100%', objectFit: 'fill', display: 'block' }} />
          {hm && <div className="heatclip">
            <img src={hm} draggable="false" style={{ position: 'absolute', left: 0, width: '100%',
              height: heightPct.toFixed(1) + '%', top: topPct.toFixed(1) + '%', opacity: .6, mixBlendMode: 'screen' }} />
          </div>}
        </div>
      ) : <span className="wait">aguardando inferência</span>}
    </MediaPanel>;
  });
  const Depth = React.memo(function Depth({ url, meta }) {
    const m = meta || {};
    return <MediaPanel tag="Depth · turbo" meta={m.min != null ? `${m.min}–${m.max} mm · ${m.valid}%` : '—'}>
      {url ? <img className="feed" src={url} draggable="false" /> : <span className="wait">aguardando depth</span>}
      <div className="dscale"><span>perto</span><div className="bar" /><span>longe</span></div>
    </MediaPanel>;
  });
  // props estáticas → renderiza 1× e o MJPEG decoda sozinho fora do React
  const SimCam = React.memo(function SimCam({ tag, path, badge }) {
    const [err, setErr] = useState(false);
    return <MediaPanel tag={tag} meta={err ? 'host --mujoco :8014 ?' : 'MuJoCo · execução defasada'} amber badge={err ? null : badge}>
      <img className="feed" src={SIMCAM + path} draggable="false" onError={() => setErr(true)} onLoad={() => setErr(false)}
           style={err ? { opacity: .12 } : null} />
      {err && <span className="wait">aguardando render do sim</span>}
    </MediaPanel>;
  });

  // ───────── arm joint blocks (Shoulder / Elbow / Wrist como painéis próprios) ─────────
  // Cada grupo de juntas vira um painel (igual aos de câmera), com as juntas sobrepostas
  // (cor por junta) e auto-escala do BLOCO — assim cada bloco tem altura p/ respirar.
  // cor ÚNICA por junta do braço (dim 7..13) — não se repete entre blocos
  const JOINT_COLS = ['#38bdf8', '#34d399', '#a78bfa', '#fb923c', '#f472b6', '#facc15', '#f87171'];
  const colOf = (d, base) => JOINT_COLS[(((d - base) % 7) + 7) % 7];  // cor por posição da junta no braço (0..6)
  const JointBlock = React.memo(function JointBlock({ tele, frame, dims, subNames, base, showCmd }) {
    const cref = useRef(null);
    const pref = useRef(null); pref.current = { tele, frame, dims, subNames, base, showCmd };
    useEffect(() => {
      const cv = cref.current, ctx = cv.getContext('2d');
      const DPR = Math.min(2, window.devicePixelRatio || 1);
      let W = 0, H = 0;
      function resize() { const r = cv.getBoundingClientRect(); W = r.width; H = r.height; cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR); }
      resize(); const ro = new ResizeObserver(resize); ro.observe(cv);
      let raf;
      function draw() {
        const P = pref.current, tl = P.tele;
        ctx.save(); ctx.scale(DPR, DPR); ctx.clearRect(0, 0, W, H);
        if (!tl || !tl.state.length) { ctx.restore(); return; }
        const F = tl.state.length;
        const padL = 10, padR = 50, padT = 22, padB = 16;
        const plotW = W - padL - padR, plotH = H - padT - padB;
        let mn = 1e9, mx = -1e9;
        const series = [['exec', tl.state]]; if (P.showCmd) series.push(['cmd', tl.action]);
        for (const [, arr] of series) for (const d of P.dims) for (let f = 0; f < F; f++) { const v = arr[f][d]; if (v < mn) mn = v; if (v > mx) mx = v; }
        if (mn === mx) { mn -= 0.1; mx += 0.1; } const pd = (mx - mn) * 0.14 || 0.05; mn -= pd; mx += pd;
        const X = f => padL + (F < 2 ? 0 : (f / (F - 1)) * plotW), Y = v => padT + plotH - ((v - mn) / (mx - mn)) * plotH;
        // zero line
        if (mn < 0 && mx > 0) { ctx.strokeStyle = 'rgba(255,255,255,.10)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(padL + plotW, Y(0)); ctx.stroke(); }
        // each joint
        P.dims.forEach((d, j) => {
          const col = colOf(d, P.base);
          if (P.showCmd) { ctx.strokeStyle = col; ctx.globalAlpha = .4; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); for (let f = 0; f < F; f++) { const x = X(f), y = Y(tl.action[f][d]); f ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]); }
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let f = 0; f < F; f++) { const x = X(f), y = Y(tl.state[f][d]); f ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
          // valor atual (à direita)
          ctx.fillStyle = col; ctx.textAlign = 'right'; ctx.font = '10px JetBrains Mono, monospace';
          ctx.fillText(tl.state[F - 1][d].toFixed(2), W - 5, padT + 11 + j * 13);
        });
        // legenda (topo) — identifica por junta + dim, cor única
        ctx.textAlign = 'left'; ctx.font = '10px JetBrains Mono, monospace'; let lx = padL + 1;
        P.dims.forEach((d, j) => { const col = colOf(d, P.base); ctx.fillStyle = col; ctx.fillRect(lx, 5, 10, 3); ctx.fillStyle = '#9fb2d4'; const nm = P.subNames[j] + ' d' + d; ctx.fillText(nm, lx + 14, 9); lx += 14 + ctx.measureText(nm).width + 13; });
        // playhead
        const fr = Math.max(0, Math.min(F - 1, Math.round(P.frame)));
        ctx.strokeStyle = '#38bdf8'; ctx.globalAlpha = .65; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(fr), padT); ctx.lineTo(X(fr), padT + plotH); ctx.stroke(); ctx.globalAlpha = 1;
        // eixo de tempo
        const fps = tl.fps || 30, winSec = (F - 1) / fps, step = winSec > 20 ? 5 : (winSec > 8 ? 2 : 1);
        ctx.fillStyle = '#48566a'; ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'center';
        for (let s = 0; s <= winSec + 1e-3; s += step) { const f = (F - 1) - s * fps; if (f < 0) break; ctx.fillText(s === 0 ? 'agora' : ('-' + s + 's'), X(f), H - 4); }
        ctx.restore();
      }
      function loop() { draw(); raf = requestAnimationFrame(loop); } raf = requestAnimationFrame(loop);
      return () => { cancelAnimationFrame(raf); ro.disconnect(); };
    }, []);
    return <canvas ref={cref} className="ramps-canvas" />;
  });
  function ArmBlockPanel({ label, sideLabel, tele, frame, dims, subNames, base, showCmd }) {
    return (
      <div className="pnl amber">
        <div className="phead"><span className="ptag"><i />{label}</span><span className="pmeta">{sideLabel} · rad</span></div>
        <div className="pbody" style={{ background: '#05080d', display: 'block' }}>
          {tele ? <JointBlock tele={tele} frame={frame} dims={dims} subNames={subNames} base={base} showCmd={showCmd} />
                : <span className="wait" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>aguardando telemetria</span>}
        </div>
      </div>
    );
  }

  // ───────── trajetória 3D (efetuador) ─────────
  // executado = FK do state medido (traço com gradiente de tempo); chunk previsto = FK
  // do chunk da VLA (traço fantasma tracejado). Reusa window.Trajectory3D (ov_traj.jsx).
  function TrajPanel({ traj, frame }) {
    return (
      <div className="pnl traj-panel">
        <div className="phead"><span className="ptag"><i />Trajetória 3D · efetuador dir.</span>
          <span className="pmeta">cheia = executado (robô) · tracejado = chunk previsto (VLA)</span></div>
        <div className="pbody" style={{ background: '#05080d', display: 'block' }}>
          {traj
            ? React.createElement(window.Trajectory3D, { kin: traj.kin, kinTarget: traj.kinChunk, frame, bbox: traj.bbox, accent: '#34d399' })
            : <span className="wait" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>aguardando telemetria</span>}
        </div>
      </div>
    );
  }

  // ───────── App ─────────
  function App() {
    const { histRef, graspRef, graspHistRef, imgRef, metaRef, lastTeleRef, phaseRef, conn, cmdHzRef, attnHzRef, chunkRef } = useLiveData();
    const [showCmd, setShowCmd] = useState(false); // mostra o "recebido" (ação da VLA) tracejado além do executado
    const [armSide, setArmSide] = useState('right'); // 'right' (dims 7-13, controlado) | 'left' (dims 0-6, limp/medido)
    const base = armSide === 'right' ? 7 : 0;
    const sideLabel = armSide === 'right' ? 'Braço dir' : 'Braço esq';
    const hist = histRef.current, F = hist.length, img = imgRef.current, meta = metaRef.current;

    const ready = F >= 2;
    const tele = useMemo(() => {
      if (!ready) return null;
      const action = hist.map(s => s.a28), state = hist.map(s => s.s28);
      return { action, state, filtered: OV.deriveFiltered(action, FPS), fps: FPS };
    }, [meta.frame, F, conn]); // eslint-disable-line

    // trajetória 3D: executado = FK do state medido; chunk = FK do chunk previsto (N×7).
    const traj = useMemo(() => {
      if (!ready) return null;
      const kin = OV.buildKinematics(hist.map(s => s.s28));
      let kinChunk = null;
      const ch = chunkRef.current;
      if (ch && ch.length && ch[0] && ch[0].length >= 7) {
        kinChunk = { ee: ch.map(q => OV.fkArm(q).ee), skel: ch.map(q => OV.fkArm(q)) };
      }
      const pts = kin.ee.concat(kinChunk ? kinChunk.ee : []);
      return { kin, kinChunk, bbox: OV.bbox(pts) };
    }, [meta.frame, F, conn]); // eslint-disable-line

    const ph = PHASE[phaseRef.current] || PHASE.unlocked;
    const g1 = (performance.now() - lastTeleRef.current) < 1500;

    return (
      <div className="app">
        <div className="top">
          <div className="brand">
            <div className="reactor" />
            <div className="title"><b>OmniView</b><small>Inference&nbsp;Live</small></div>
          </div>
          <div className="chips">
            <span className="chip">π0.5 · <b>armstate7-8k</b></span>
            <span className="chip" title="passo cumulativo da VLA (não reseta); o buffer do gráfico é capado em ~30s">ep <b>{String(meta.episode).padStart(2, '0')}</b> · step <b>{meta.frame}</b></span>
            <span className="chip" title="comandos enviados ao robô (loop de controle)">cmd <b>{cmdHzRef.current.hz.toFixed(0)}</b> Hz</span>
            <span className="chip" title="atualização do mapa de atenção (= taxa de inferência da VLA)">atn <b>{attnHzRef.current.hz.toFixed(1)}</b> Hz</span>
            <span className={'chip phase ' + ph[0]}><span className="pdot" />{ph[1]}</span>
          </div>
          <div className="armsel">
            {[['right', 'Braço Direito'], ['left', 'Braço Esquerdo']].map(([s, l]) =>
              <button key={s} className={'armbtn' + (armSide === s ? ' on' : '')} onClick={() => setArmSide(s)}>{l}</button>)}
          </div>
          <div className="spacer" />
          <div className="leds"><Led label="WS" state={conn} /><Led label="VLA" state={g1} /></div>
          <GraspGauge grasp={graspRef.current} hist={graspHistRef.current} />
        </div>

        <div className="grid">
          <div className="cell"><Rgb url={img.rgb} size={img.rgbSize} /></div>
          <div className="cell"><Depth url={img.depth} meta={img.depthMeta} /></div>
          <div className="cell"><Attn url={img.rgb} hm={img.attnHm} size={img.rgbSize} /></div>
          <div className="cell"><ArmBlockPanel label="Shoulder" sideLabel={sideLabel} tele={tele} frame={F - 1} dims={[base, base + 1, base + 2]} subNames={['Pitch', 'Roll', 'Yaw']} base={base} showCmd={showCmd} /></div>
          <div className="cell"><ArmBlockPanel label="Elbow" sideLabel={sideLabel} tele={tele} frame={F - 1} dims={[base + 3]} subNames={['Elbow']} base={base} showCmd={showCmd} /></div>
          <div className="cell"><ArmBlockPanel label="Wrist" sideLabel={sideLabel} tele={tele} frame={F - 1} dims={[base + 4, base + 5, base + 6]} subNames={['Roll', 'Pitch', 'Yaw']} base={base} showCmd={showCmd} /></div>
          <div className="cell traj-wide"><TrajPanel traj={traj} frame={F - 1} /></div>
        </div>

        <div className="foot">
          <span className={'rec' + (conn ? ' on' : '')} />
          <span>{conn ? 'AO VIVO' : 'sem conexão'}</span>
          <span className="sp" />
          <span>buffer <b>{F}</b>/{MAXLEN}f</span>
          <span>braço dir · linha cheia = executado (robô)</span>
          <button className={'sig' + (showCmd ? ' on' : '')} onClick={() => setShowCmd(s => !s)} style={{ marginLeft: 8 }}><i style={{ borderColor: '#9fb2d4', background: showCmd ? '#9fb2d4' : 'transparent' }} />recebido (tracejado)</button>
        </div>
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
})();
