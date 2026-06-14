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
            if (m.attn_hm) { n.attnHm = dataUrlToBlobUrl(m.attn_hm);if (p.attnHm) URL.revokeObjectURL(p.attnHm); }
            imgRef.current = n; dirty.current = true;
          } else if (m.type === 'tele') {
            lastTeleRef.current = performance.now();
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
      function loop(ts) { if (dirty.current && (ts - lastT) >= 33) { dirty.current = false; lastT = ts; setTick(t => t + 1); } raf = requestAnimationFrame(loop); }
      raf = requestAnimationFrame(loop);
      return () => { alive = false; clearTimeout(retry); cancelAnimationFrame(raf); try { ws.close(); } catch (e) {} };
    }, []);
    return { histRef, graspRef, graspHistRef, imgRef, metaRef, lastTeleRef, phaseRef, conn };
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

  // ───────── ramps panel (reusa window.Ramps) ─────────
  function RampsPanel({ tele, frame, group, setGroup, sig, setSig }) {
    const names = OV.LAYOUT[group].kind === 'arm' ? OV.ARM_JOINTS : (group === 'rightHand' ? OV.RHAND_JOINTS : OV.LHAND_JOINTS);
    const Toggle = ({ k, c, l }) => <button className={'sig' + (sig[k] ? ' on' : '')} style={{ color: sig[k] ? c : undefined }}
      onClick={() => setSig(s => ({ ...s, [k]: !s[k] }))}><i style={{ borderColor: c, background: sig[k] ? c : 'transparent' }} />{l}</button>;
    return (
      <div className="pnl amber" style={{ gridColumn: '3', gridRow: '2' }}>
        <div className="phead">
          <span className="ptag"><i />Comando</span>
          <div className="rcontrols">
            {['rightArm', 'rightHand'].map(g => <button key={g} className={'rtab' + (group === g ? ' on' : '')}
              onClick={() => setGroup(g)}>{OV.LAYOUT[g].label}</button>)}
          </div>
        </div>
        <div className="phead" style={{ borderTop: 'none', paddingTop: 5, paddingBottom: 5, background: 'none' }}>
          <span className="pmeta">recebido / filtrado / executado</span>
          <div className="rleg"><Toggle k="cmd" c="#22d3ee" l="recebido" /><Toggle k="filt" c="#f59e0b" l="filtrado" /><Toggle k="exec" c="#34d399" l="executado" /></div>
        </div>
        <div className="pbody" style={{ background: '#05080d', display: 'block' }}>
          {tele ? <Ramps tele={tele} frame={frame} group={group} jointNames={names} signals={sig} onScrub={() => {}} accent={ACCENT} />
                : <span className="wait" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>aguardando telemetria</span>}
        </div>
        <div className="rfoot">scroll = comprime/expande o tempo · "filtrado" derivado ao vivo · "executado" = pós-clamp</div>
      </div>
    );
  }

  // ───────── App ─────────
  function App() {
    const { histRef, graspRef, graspHistRef, imgRef, metaRef, lastTeleRef, phaseRef, conn } = useLiveData();
    const [group, setGroup] = useState('rightArm');
    const [sig, setSig] = useState({ cmd: false, filt: true, exec: true });
    const hzRef = useRef({ last: performance.now(), n: 0, hz: 0 });
    const hist = histRef.current, F = hist.length, img = imgRef.current, meta = metaRef.current;
    { const h = hzRef.current; h.n++; const now = performance.now(); if (now - h.last > 1000) { h.hz = h.n * 1000 / (now - h.last); h.n = 0; h.last = now; } }

    const ready = F >= 2;
    const tele = useMemo(() => {
      if (!ready) return null;
      const action = hist.map(s => s.a28), state = hist.map(s => s.s28);
      return { action, state, filtered: OV.deriveFiltered(action, FPS), fps: FPS };
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
            <span className="chip">{hzRef.current.hz.toFixed(0)} <b>Hz</b></span>
            <span className={'chip phase ' + ph[0]}><span className="pdot" />{ph[1]}</span>
          </div>
          <div className="spacer" />
          <div className="leds"><Led label="WS" state={conn} /><Led label="VLA" state={g1} /></div>
          <GraspGauge grasp={graspRef.current} hist={graspHistRef.current} />
        </div>

        <div className="grid">
          <div className="cell"><Rgb url={img.rgb} size={img.rgbSize} /></div>
          <div className="cell"><Depth url={img.depth} meta={img.depthMeta} /></div>
          <div className="cell"><Attn url={img.rgb} hm={img.attnHm} size={img.rgbSize} /></div>
          <div className="cell"><SimCam tag="Robô · 3ª pessoa" path="/global.mjpg" badge="SIM" /></div>
          <div className="cell"><SimCam tag="Robô · 1ª pessoa" path="/head.mjpg" badge="POV" /></div>
          <div className="cell"><RampsPanel tele={tele} frame={F - 1} group={group} setGroup={setGroup} sig={sig} setSig={setSig} /></div>
        </div>

        <div className="foot">
          <span className={'rec' + (conn ? ' on' : '')} />
          <span>{conn ? 'AO VIVO' : 'sem conexão'}</span>
          <span className="sp" />
          <span>buffer <b>{F}</b>/{MAXLEN}f</span>
          <span>replay aberto · a VLA vê a imagem real · o robô do sim visualiza o comando</span>
        </div>
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
})();
