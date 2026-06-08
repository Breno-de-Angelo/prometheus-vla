/* live_app.jsx — versão AO VIVO da tela de detalhe do OmniView.
   Lê um websocket (/ws) com:
     • {type:'img', rgb, depth, depthMeta, rgbSize}  — frames de câmera
     • {type:'tele', episode, frame, state{...}, action{...}, pressure{left,right}}
   Mantém um buffer rolante das últimas N amostras de telemetria e alimenta os
   MESMOS componentes do OmniView (Trajectory3D, Ramps, TactilePanel) montando
   arrays [F][28] + cinemática + mapa tátil em tempo real.
*/
(function () {
  const { useState, useEffect, useRef, useMemo } = React;

  // ----- layout das 28 juntas REAIS do G1 (sobrescreve o demo do ov_data) -----
  // packing: [0-6 braço esq] [7-13 braço dir ATIVO] [14-20 mão esq] [21-27 mão dir ATIVA]
  OV.LAYOUT = {
    leftArm:   { start: 0,  len: 7, active: false, label: 'Braço esq.', kind: 'arm' },
    rightArm:  { start: 7,  len: 7, active: true,  label: 'Braço dir.', kind: 'arm' },
    leftHand:  { start: 14, len: 7, active: false, label: 'Mão esq.',   kind: 'hand' },
    rightHand: { start: 21, len: 7, active: true,  label: 'Mão dir.',   kind: 'hand' },
  };
  OV.ARM_JOINTS = ['Shoulder·Pitch','Shoulder·Roll','Shoulder·Yaw','Elbow','Wrist·Roll','Wrist·Pitch','Wrist·Yaw'];
  // mão direita Dex3: thumb0,1,2 · index0,1 · middle0,1
  OV.RHAND_JOINTS = ['Thumb·0','Thumb·1','Thumb·2','Index·0','Index·1','Middle·0','Middle·1'];
  OV.LHAND_JOINTS = ['Thumb·0','Thumb·1','Thumb·2','Middle·0','Middle·1','Index·0','Index·1'];

  const FPS = 30;
  const MAXLEN = 30 * FPS;         // ~30 s de histórico (dá margem pra comprimir o tempo nas rampas)
  const ACCENT = '#38bdf8';

  function flat28(g) {
    // concatena na ordem do LAYOUT; tolera grupos faltando
    const z7 = [0,0,0,0,0,0,0];
    return [].concat(g.leftArm||z7, g.rightArm||z7, g.leftHand||z7, g.rightHand||z7);
  }

  // "data:image/jpeg;base64,XXXX" -> Blob URL (revogável; não acumula memória)
  function dataUrlToBlobUrl(dataUrl) {
    const i = dataUrl.indexOf(',');
    const bin = atob(dataUrl.slice(i + 1));
    const bytes = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
    return URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
  }

  // ---- pressão: 108 (9 áreas × 12) -> 33 taxels do display do Dex3 ----------
  // Áreas (ordem do firmware = índices do sensor Dex3):
  //   0 ThumbBase 1 ThumbTip 2 MiddleBase 3 MiddleTip 4 IndexBase 5 IndexTip
  //   6 Palm0 7 Palm1 8 Palm2  → cada área tem 12 taxels.
  // O componente tátil espera um vetor[33] (ver ov_tactile.jsx):
  //   Thumb base[0-3] tip[4-6] · Middle base[7-10] tip[11-13] · Index base[14-17]
  //   tip[18-20] · Palm[21-24 / 25-28 / 29-32].
  const AREA_TO_TAXELS = [
    [0,1,2,3],     // 0 ThumbBase
    [4,5,6],       // 1 ThumbTip
    [7,8,9,10],    // 2 MiddleBase
    [11,12,13],    // 3 MiddleTip
    [14,15,16,17], // 4 IndexBase
    [18,19,20],    // 5 IndexTip
    [21,22,23,24], // 6 Palm0 (perto do médio)
    [25,26,27,28], // 7 Palm1 (perto do indicador)
    [29,30,31,32], // 8 Palm2 (perto do polegar)
  ];

  function pressureToTaxels(p108, scale) {
    const out = new Float32Array(33);
    if (!p108 || p108.length < 108) return out;
    for (let a = 0; a < 9; a++) {
      const dst = AREA_TO_TAXELS[a];
      const base = a * 12, n = dst.length;
      // downsample 12 -> n por média de blocos
      for (let k = 0; k < n; k++) {
        const i0 = Math.floor(k * 12 / n), i1 = Math.floor((k + 1) * 12 / n);
        let s = 0, c = 0;
        for (let i = i0; i < i1; i++) { s += p108[base + i]; c++; }
        out[dst[k]] = c ? Math.max(0, Math.min(1, (s / c) / scale)) : 0;
      }
    }
    return out;
  }

  // ----------------------------------------------------------------- websocket
  function useLiveData() {
    const histRef = useRef([]);          // amostras {s28, a28, pl, pr}
    const imgRef = useRef({});           // {rgb, depth, depthMeta, rgbSize}
    const metaRef = useRef({ episode: 0, frame: 0 });
    const pmaxRef = useRef(120);         // auto-ganho da pressão
    const lastTeleRef = useRef(0);       // performance.now() do último pacote de telemetria (freshness do G1)
    const questRef = useRef(null);       // status do Quest vindo do servidor (null=desconhecido)
    const phaseRef = useRef('unlocked'); // estado do robô: softstart | locked | unlocked
    const [, setTick] = useState(0);
    const [conn, setConn] = useState(false);
    const dirty = useRef(false);

    useEffect(() => {
      let ws, retry, alive = true;
      function connect() {
        ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onopen = () => setConn(true);
        ws.onclose = () => { setConn(false); if (alive) retry = setTimeout(connect, 1000); };
        ws.onerror = () => { try { ws.close(); } catch (e) {} };
        ws.onmessage = (ev) => {
          let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
          if (m.type === 'links') {
            questRef.current = m.quest;     // true/false/null vindo do servidor (adb do Quest)
            dirty.current = true;
          } else if (m.type === 'img') {
            // converte cada frame p/ Blob URL e REVOGA o anterior — senão o browser
            // acumula data-URLs decodificados (848×480 a 30Hz estourava a memória/aba).
            const prev = imgRef.current;
            const next = {
              rgb: prev.rgb, depth: prev.depth,
              rgbSize: m.rgbSize || prev.rgbSize,
              depthMeta: m.depthMeta || prev.depthMeta,
            };
            if (m.rgb)   { next.rgb   = dataUrlToBlobUrl(m.rgb);   if (prev.rgb)   URL.revokeObjectURL(prev.rgb); }
            if (m.depth) { next.depth = dataUrlToBlobUrl(m.depth); if (prev.depth) URL.revokeObjectURL(prev.depth); }
            imgRef.current = next;
            dirty.current = true;
          } else if (m.type === 'tele') {
            lastTeleRef.current = performance.now();
            if (m.robot && m.robot.phase) phaseRef.current = m.robot.phase;
            const s28 = flat28(m.state || {});
            const a28 = flat28(m.action || {});
            const pr = (m.pressure && m.pressure.right) || null;
            const pl = (m.pressure && m.pressure.left) || null;
            if (pr) { let mx = 0; for (const v of pr) if (v > mx) mx = v; if (mx > pmaxRef.current) pmaxRef.current = mx; }
            const h = histRef.current;
            h.push({ s28, a28, pr, pl });
            if (h.length > MAXLEN) h.shift();
            metaRef.current = { episode: m.episode || 0, frame: m.frame || 0 };
            dirty.current = true;
          }
        };
      }
      connect();
      // repinta no máx. ~30 Hz quando houver dado novo (desacopla render do socket)
      let raf, lastT = 0;
      function loop(ts) {
        if (dirty.current && (ts - lastT) >= 33) { dirty.current = false; lastT = ts; setTick((t) => t + 1); }
        raf = requestAnimationFrame(loop);
      }
      raf = requestAnimationFrame(loop);
      return () => { alive = false; clearTimeout(retry); cancelAnimationFrame(raf); try { ws.close(); } catch (e) {} };
    }, []);

    return { histRef, imgRef, metaRef, pmaxRef, lastTeleRef, questRef, phaseRef, conn };
  }

  // --------------------------------------------------------------- bloco de conexões
  function Dot({ label, state }) {
    // state: true=verde (conectado) · false=vermelho (offline) · null=cinza (desconhecido)
    const cls = state === true ? 'on' : (state === false ? 'off' : 'unk');
    const txt = state === true ? 'conectado' : (state === false ? 'offline' : 'n/d');
    return (
      <div className="conn-row">
        <span className={"conn-dot " + cls} />
        <span className="conn-name mono">{label}</span>
        <span className={"conn-state mono " + cls}>{txt}</span>
      </div>
    );
  }

  const PHASE = {
    unlocked:  { cls: 'on',   txt: 'DESBLOQUEADO', hint: 'teleoperando' },
    locked:    { cls: 'off',  txt: 'BLOQUEADO',    hint: 'aperte X para liberar' },
    softstart: { cls: 'warn', txt: 'INICIANDO…',   hint: 'aguarde · não aperte X ainda' },
  };

  function EstadoPanel({ phase }) {
    const ph = PHASE[phase] || PHASE.unlocked;
    return (
      <div className="panel estado-panel">
        <div className="panel-head">
          <span className="ph-label">ESTADO</span>
          <span className="ph-meta mono">trava do robô · X</span>
        </div>
        <div className="estado-body">
          <div className={"estado-pill " + ph.cls}>
            <span className={"estado-dot " + ph.cls} />
            <span className="estado-txt">{ph.txt}</span>
          </div>
          <div className="estado-hint mono">{ph.hint}</div>
        </div>
      </div>
    );
  }

  function ConnPanel({ pc, quest, g1 }) {
    return (
      <div className="panel conn-panel">
        <div className="panel-head">
          <span className="ph-label">CONEXÕES</span>
          <span className="ph-meta mono">status dos links</span>
        </div>
        <div className="conn-body">
          <Dot label="PC (dashboard)" state={pc} />
          <Dot label="Oculus Quest" state={quest} />
          <Dot label="G1 (robô)" state={g1} />
        </div>
      </div>
    );
  }

  // --------------------------------------------------------------- DoF mask map
  function DofMap({ group }) {
    const segs = ['leftArm','rightArm','leftHand','rightHand'].map(k => ({ k, ...OV.LAYOUT[k] }));
    return (
      <div className="dofmap">
        {segs.map(s => (
          <div key={s.k} className={"dof-seg" + (s.active ? ' active' : ' masked') + (group === s.k ? ' sel' : '')}>
            <span className="dof-lab mono">{s.label}{!s.active && <em> · mask</em>}</span>
            <div className="dof-cells">
              {Array.from({ length: s.len }).map((_, i) => <span key={i} className="dof-cell" />)}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // --------------------------------------------------------------- RGB ao vivo
  function LiveRGB({ url, size }) {
    return (
      <div className="media-panel">
        <div className="media-head">
          <span className="mh-label">RGB · head_camera</span>
          <span className="mh-meta">{size ? `${size[0]}×${size[1]}` : '—'} · live</span>
        </div>
        <div className="media-body">
          {url
            ? <img src={url} className="rgb-canvas" draggable="false" />
            : <div className="wait mono">aguardando câmera…</div>}
          <span className="real-badge">LIVE</span>
          <div className="scanline" />
        </div>
      </div>
    );
  }

  // --------------------------------------------------------------- Depth ao vivo
  function LiveDepth({ url, meta }) {
    const m = meta || {};
    return (
      <div className="media-panel">
        <div className="media-head">
          <span className="mh-label">DEPTH · turbo</span>
          <span className="mh-meta">{m.min != null ? `${m.min}–${m.max} mm · ${m.valid}% válido` : '—'}</span>
        </div>
        <div className="media-body depth-body">
          {url
            ? <img src={url} className="depth-img" draggable="false" />
            : <div className="wait mono">aguardando depth…</div>}
          <span className="depth-frame-note">
            escala fixa {m.visMin != null ? `${m.visMin}-${m.visMax} m` : ''} · perto=verm. longe=azul
          </span>
          <div className="depth-scale"><span>near</span><div className="ds-bar" /><span>far</span></div>
        </div>
      </div>
    );
  }

  // --------------------------------------------------------------- status (rodapé)
  function StatusBar({ conn, episode, frames, hz }) {
    return (
      <div className="transport live-status">
        <span className={"rec-dot" + (conn ? '' : ' off')} />
        <span className="tp-time mono">{conn ? 'AO VIVO' : 'sem conexão'}</span>
        <div className="dt-spacer" />
        <span className="tp-frame mono">ep <b>{String(episode).padStart(2, '0')}</b></span>
        <span className="tp-frame mono">buffer <span>{frames}f / {MAXLEN}f</span></span>
        <span className="tp-frame mono">{hz.toFixed(0)} Hz</span>
      </div>
    );
  }

  function SigToggle({ on, label, color, onClick }) {
    return <button className={"sig-tog" + (on ? ' on' : '')} onClick={onClick}>
      <i style={{ background: on ? color : 'transparent', borderColor: color }} /> {label}
    </button>;
  }

  // ----------------------------------------------------------------------- App
  function App() {
    const { histRef, imgRef, metaRef, pmaxRef, lastTeleRef, questRef, phaseRef, conn } = useLiveData();
    const [group, setGroup] = useState('rightArm');
    const [sig, setSig] = useState({ cmd: false, filt: true, exec: true });
    const hzRef = useRef({ last: performance.now(), n: 0, hz: 0 });

    const hist = histRef.current;
    const F = hist.length;
    const img = imgRef.current;
    const meta = metaRef.current;

    // taxa de chegada de frames (Hz) — só visual
    {
      const h = hzRef.current; h.n++;
      const now = performance.now();
      if (now - h.last > 1000) { h.hz = h.n * 1000 / (now - h.last); h.n = 0; h.last = now; }
    }

    const ready = F >= 2;
    const tele = useMemo(() => {
      if (!ready) return null;
      const action = hist.map(s => s.a28);
      const state = hist.map(s => s.s28);
      const filtered = OV.deriveFiltered(action, FPS);
      const scale = Math.max(pmaxRef.current, 60);
      const last = hist[hist.length - 1];
      const taxR = pressureToTaxels(last.pr, scale);
      return { action, state, filtered, hands: { right: { taxels: [taxR] } } };
      // frame muda a cada telemetria → recomputa mesmo com buffer cheio (length fixo em 900)
    }, [meta.frame, F, conn]);  // eslint-disable-line

    const groupDef = { names: OV.LAYOUT[group].kind === 'arm' ? OV.ARM_JOINTS
                              : (group === 'rightHand' ? OV.RHAND_JOINTS : OV.LHAND_JOINTS) };
    const frame = F - 1;

    // estados de conexão: PC = ws aberto · G1 = telemetria fresca (<1.5s) · Quest = adb (servidor)
    const pcLink = conn;
    const g1Link = (performance.now() - lastTeleRef.current) < 1500;
    const questLink = questRef.current;

    return (
      <div className="detail-view live-grid">
        <header className="detail-top">
          <div className="dt-id">
            <span className="dt-ep mono">ep{String(meta.episode).padStart(2, '0')}</span>
            <span className="tag-real mono">LIVE</span>
          </div>
          <div className="dt-task">G1 · Dex3-1 · teleop</div>
          <div className="dt-meta mono">{F} frames no buffer · {FPS}fps</div>
          <div className="dt-spacer" />
          <DofMap group={group} />
        </header>

        <div className="detail-grid">
          <div className="cell cell-rgb"><LiveRGB url={img.rgb} size={img.rgbSize} /></div>
          <div className="cell cell-depth"><LiveDepth url={img.depth} meta={img.depthMeta} /></div>
          <div className="cell cell-tactile">
            {ready
              ? <TactilePanel hands={tele.hands} frame={0} group={group} palette="heat" />
              : <div className="panel tactile-panel"><div className="panel-head"><span className="ph-label">TÁTIL · DEX3-1</span></div><div className="wait mono">aguardando sensor…</div></div>}
          </div>
          <div className="cell cell-ramps">
            <div className="panel ramps-panel">
              <div className="panel-head">
                <div className="rg-tabs">
                  {['rightArm','rightHand','leftArm','leftHand'].map(g => {
                    const def = OV.LAYOUT[g];
                    return <button key={g} className={"rg-tab" + (group === g ? ' on' : '') + (def.active ? '' : ' masked')}
                      onClick={() => setGroup(g)}>{def.label}{def.active ? '' : ' ·mask'}</button>;
                  })}
                </div>
                <div className="rg-legend mono">
                  <SigToggle on={sig.cmd} label="recebido" color="#22d3ee" onClick={() => setSig(s => ({ ...s, cmd: !s.cmd }))} />
                  <SigToggle on={sig.filt} label="filtrado*" color="#f59e0b" onClick={() => setSig(s => ({ ...s, filt: !s.filt }))} />
                  <SigToggle on={sig.exec} label="executado" color="#34d399" onClick={() => setSig(s => ({ ...s, exec: !s.exec }))} />
                </div>
              </div>
              {ready
                ? <Ramps tele={{ action: tele.action, state: tele.state, filtered: tele.filtered, fps: FPS }}
                         frame={frame} group={group} jointNames={groupDef.names}
                         signals={sig} onScrub={() => {}} accent={ACCENT} />
                : <div className="wait mono">aguardando telemetria…</div>}
              <div className="ramps-foot mono">scroll = comprimir/expandir o tempo (X) · 2× clique = buffer inteiro &nbsp;·&nbsp; * "filtrado" derivado ao vivo</div>
            </div>
          </div>
          <div className="cell cell-conn">
            <EstadoPanel phase={phaseRef.current} />
            <ConnPanel pc={pcLink} quest={questLink} g1={g1Link} />
          </div>
        </div>

        <StatusBar conn={conn} episode={meta.episode} frames={F} hz={hzRef.current.hz} />
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
})();
