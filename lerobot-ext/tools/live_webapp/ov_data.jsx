/* ov_data.jsx — OmniView data layer
   - decodes real ep4 (action/state) from gzip+base64
   - approximate forward-kinematics for the G1 right arm end-effector
   - derives a plausible "filtered" (rate-limited) command stream
   - synthesizes sibling episodes for the grid/detail demo
   Exposes window.OV (namespace).
*/
(function () {
  const OV = (window.OV = window.OV || {});

  // ---- joint layout (28 dims) ----------------------------------------------
  // 0-6 left arm (masked: near-static reference), 7-13 right arm (ACTIVE),
  // 14-21 left hand (masked: zero), 22-27 right hand (ACTIVE)
  OV.LAYOUT = {
    leftArm:   { start: 0,  len: 7,  active: false, label: 'Braço esq.', kind: 'arm' },
    rightArm:  { start: 7,  len: 7,  active: true,  label: 'Braço dir.', kind: 'arm' },
    leftHand:  { start: 14, len: 8,  active: false, label: 'Mão esq.',   kind: 'hand' },
    rightHand: { start: 22, len: 6,  active: true,  label: 'Mão dir.',   kind: 'hand' },
  };
  OV.ARM_JOINTS = ['Shoulder·Pitch','Shoulder·Roll','Shoulder·Yaw','Elbow','Wrist·Roll','Wrist·Pitch','Wrist·Yaw'];
  OV.RHAND_JOINTS = ['Thumb·0','Thumb·1','Thumb·2','Index','Middle','Ring'];

  // ---- gzip+base64 decode ---------------------------------------------------
  OV.decodeReal = async function () {
    const b64 = window.__EP4_GZB64;
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const ds = new DecompressionStream('gzip');
    const w = ds.writable.getWriter(); w.write(bytes); w.close();
    const ab = await new Response(ds.readable).arrayBuffer();
    const json = new TextDecoder().decode(ab);
    return JSON.parse(json); // {nframes,fps,ts,action,state,ranges}
  };

  // ---- REAL forward-kinematics (quaternions, from g1_29dof_with_hand.xml) ---
  // Cadeia real do braço DIREITO: pos (offset do pai, m) + quat (montagem, wxyz)
  // + axis (eixo da junta no frame do link). Substitui a FK ilustrativa antiga.
  const RIGHT_ARM = [
    { pos:[0.003956,-0.10021,0.23778], quat:[0.990264,-0.139201,0.000014,0.000099], axis:[0,1,0] }, // shoulder pitch
    { pos:[0,-0.038,-0.013831],        quat:[0.990268,0.139172,0,0],                axis:[1,0,0] }, // shoulder roll
    { pos:[0,-0.00624,-0.1032],        quat:[1,0,0,0],                              axis:[0,0,1] }, // shoulder yaw
    { pos:[0.015783,0,-0.080518],      quat:[1,0,0,0],                              axis:[0,1,0] }, // elbow
    { pos:[0.1,-0.001888,-0.01],       quat:[1,0,0,0],                              axis:[1,0,0] }, // wrist roll
    { pos:[0.038,0,0],                 quat:[1,0,0,0],                              axis:[0,1,0] }, // wrist pitch
    { pos:[0.046,0,0],                 quat:[1,0,0,0],                              axis:[0,0,1] }, // wrist yaw
  ];
  const PALM_OFF = [0.0415,0.003,0]; // wrist_yaw -> centro da palma (end-effector)

  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  function qmul(a,b){ // Hamilton, [w,x,y,z]
    return [ a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
             a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
             a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
             a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] ];
  }
  function qrot(q,v){ // rotaciona vetor v por quat q
    const t=[ 2*(q[2]*v[2]-q[3]*v[1]), 2*(q[3]*v[0]-q[1]*v[2]), 2*(q[1]*v[1]-q[2]*v[0]) ];
    return [ v[0]+q[0]*t[0]+(q[2]*t[2]-q[3]*t[1]),
             v[1]+q[0]*t[1]+(q[3]*t[0]-q[1]*t[2]),
             v[2]+q[0]*t[2]+(q[1]*t[1]-q[2]*t[0]) ];
  }
  function axisQuat(ax,ang){ const h=ang/2,s=Math.sin(h); return [Math.cos(h),ax[0]*s,ax[1]*s,ax[2]*s]; }

  // FK real: 7 ângulos do braço dir. -> pontos 3D (shoulder/elbow/wrist/ee) no frame do torso
  OV.fkArm = function (q) {
    let T = { p:[0,0,0], q:[1,0,0,0] };
    const names = ['shoulder',null,null,'elbow','wrist',null,null];
    const pts = {};
    for (let i=0;i<7;i++){
      const L = RIGHT_ARM[i];
      T = { p: add(T.p, qrot(T.q, L.pos)), q: qmul(T.q, L.quat) }; // frame do link
      if (names[i]) pts[names[i]] = T.p.slice();                  // ponto da junta
      T.q = qmul(T.q, axisQuat(L.axis, q[i]));                    // rotação da junta
    }
    const ee = add(T.p, qrot(T.q, PALM_OFF));
    return { shoulder: pts.shoulder, elbow: pts.elbow, wrist: pts.wrist, ee };
  };

  // build full EE path + per-frame skeleton from a [F][28] joint array
  OV.buildKinematics = function (joints28) {
    const F = joints28.length;
    const ee = new Array(F), skel = new Array(F);
    for (let f=0; f<F; f++) {
      const q = joints28[f].slice(7, 14);
      const k = OV.fkArm(q);
      ee[f] = k.ee; skel[f] = k;
    }
    return { ee, skel };
  };

  // bounding box of an array of [x,y,z]
  OV.bbox = function (pts) {
    const lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
    for (const p of pts) for (let i=0;i<3;i++){ if(p[i]<lo[i])lo[i]=p[i]; if(p[i]>hi[i])hi[i]=p[i]; }
    const c=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
    const r=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],0.001);
    return { lo, hi, c, r };
  };

  // ---- derive a plausible "filtered" command -------------------------------
  // received(action) -> safety filter: per-step slew-rate clamp + light EMA.
  // Clearly an ILLUSTRATIVE derivation (raw filtered stream not in this export).
  OV.deriveFiltered = function (action, fps) {
    const F = action.length, D = action[0].length;
    const out = new Array(F);
    const maxRate = 0.9 / fps;      // rad per frame slew limit
    const alpha = 0.35;             // EMA smoothing
    let prev = action[0].slice();
    for (let f=0; f<F; f++) {
      const row = new Array(D);
      for (let d=0; d<D; d++) {
        const target = action[f][d];
        let step = target - prev[d];
        if (step >  maxRate) step =  maxRate;
        if (step < -maxRate) step = -maxRate;
        const slewed = prev[d] + step;
        row[d] = prev[d] + alpha * (slewed - prev[d]) + (1 - alpha) * (slewed - prev[d]);
        // simpler: EMA toward slewed
        row[d] = prev[d] + (slewed - prev[d]); // keep slewed
        prev[d] = row[d];
      }
      out[f] = row;
    }
    // second pass: gentle low-pass for the "tracking lag" feel
    const sm = new Array(F);
    let p2 = out[0].slice();
    for (let f=0; f<F; f++){
      const row=new Array(D);
      for(let d=0; d<D; d++){ p2[d]=p2[d]+alpha*(out[f][d]-p2[d]); row[d]=p2[d]; }
      sm[f]=row;
    }
    return sm;
  };

  // ---- per-dim min/max over frames (for chart scaling) ----------------------
  OV.dimRange = function (rows, d) {
    let mn=1e9, mx=-1e9;
    for (let f=0; f<rows.length; f++){ const v=rows[f][d]; if(v<mn)mn=v; if(v>mx)mx=v; }
    if (mn===mx){ mn-=0.05; mx+=0.05; }
    return [mn, mx];
  };

  // ---- deterministic PRNG ---------------------------------------------------
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

  // resample a [F][D] array to length N with optional time-warp(0..1->0..1)
  function resample(rows, N, warp) {
    const F = rows.length, D = rows[0].length, out = new Array(N);
    for (let i=0;i<N;i++){
      let u = i/(N-1); if (warp) u = warp(u);
      const x = u*(F-1); const a=Math.floor(x), b=Math.min(F-1,a+1), t=x-a;
      const row=new Array(D);
      for (let d=0; d<D; d++) row[d]=rows[a][d]*(1-t)+rows[b][d]*t;
      out[i]=row;
    }
    return out;
  }

  // ---- synthesize a sibling episode from real ep4 --------------------------
  // returns {action,state,ts,fps,nframes}
  OV.synthTelemetry = function (real, ep) {
    const rnd = mulberry32(ep.seed);
    const N = ep.nframes, fps = ep.fps;
    // time-warp: ease-in/out variation so motion timing differs
    const k = 0.7 + rnd()*0.7;
    const warp = (u) => Math.pow(u, k);
    let action = resample(real.action, N, warp);
    let state  = resample(real.state,  N, warp);
    // per-dim bias + noise on ACTIVE dims only; keep masked dims as-is
    const activeDims = [7,8,9,10,11,12,13,22,23,24,25,26,27];
    const bias = {}; activeDims.forEach(d => bias[d] = (rnd()-0.5)*0.12);
    for (let f=0; f<N; f++){
      for (const d of activeDims){
        const n = (rnd()-0.5)*0.02;
        action[f][d] += bias[d] + n*0.6;
        state[f][d]  += bias[d] + n;
      }
      // a failed grasp: right-hand close dims don't reach full range
      if (ep.fail){
        for (const d of [22,23,24,25,26,27]){
          action[f][d] *= 0.45; state[f][d] *= 0.4;
        }
      }
    }
    const ts = new Array(N); for (let i=0;i<N;i++) ts[i]=+(i/fps).toFixed(3);
    return { action, state, ts, fps, nframes: N };
  };

  // ---- episode catalog ------------------------------------------------------
  OV.buildCatalog = function () {
    const COUNT = 23, REAL_IDX = 4, fps = 30;
    const rnd = mulberry32(0xC0FFEE);
    const eps = [];
    for (let i=0;i<COUNT;i++){
      const isReal = i===REAL_IDX;
      const durSec = isReal ? 11.2 : +(8 + rnd()*6).toFixed(1);
      const nframes = isReal ? 337 : Math.round(durSec*fps);
      // a couple of flagged/failed takes among the synth ones
      const fail = !isReal && (i===9 || i===17);
      eps.push({
        idx: i,
        id: 'ep' + String(i).padStart(2,'0'),
        isReal, fail,
        durSec, nframes, fps,
        seed: 1000 + i*97,
        task: 'Pick up the white cup',
        hue: isReal ? 0 : Math.round((rnd()-0.5)*44), // tile tint for demo variety
      });
    }
    return { eps, REAL_IDX, fps, COUNT };
  };

  // turbo-ish colormap (t in 0..1) -> [r,g,b]
  OV.turbo = function (t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255*Math.min(1, Math.max(0, 1.5 - Math.abs(2*t-1.6)*2.2 + 0.2)));
    const g = Math.round(255*Math.min(1, Math.max(0, 1.1 - Math.abs(2*t-1.0)*1.9)));
    const b = Math.round(255*Math.min(1, Math.max(0, 1.4 - Math.abs(2*t-0.3)*2.2)));
    return [r,g,b];
  };

  // ---- tactile: derive per-hand contact force from hand-joint closure -------
  // A hand only registers contact when its joints actually move (close). Static
  // / masked hands (left) produce ~0 force. Failed grasps (synth) close less,
  // so contact stays low — the tactile map tells success from failure.
  OV.smoothstep = function (a, b, x) { x = Math.max(0, Math.min(1, (x - a) / (b - a))); return x * x * (3 - 2 * x); };

  OV.handStreams = function (state, refState) {
    const F = state.length;
    const ref = refState || state;          // fixed open→closed reference (real ep4)
    const groups = { right: [22,23,24,25,26,27], left: [14,15,16,17,18,19,20,21] };
    const out = {};
    for (const key in groups) {
      const dims = groups[key];
      // per-dim range taken from the REFERENCE episode, so a weaker/partial
      // grasp normalizes below 1.0 (failed grasp → genuinely lower contact)
      const ranges = dims.map(d => { let mn=1e9, mx=-1e9; for (let f=0;f<ref.length;f++){ const v=ref[f][d]; if(v<mn)mn=v; if(v>mx)mx=v; } return [mn,mx]; });
      const raw = new Float32Array(F);
      for (let f=0; f<F; f++) {
        let s=0, c=0;
        for (let k=0;k<dims.length;k++){ const [mn,mx]=ranges[k]; const rng=mx-mn; if (rng<1e-3) continue; s += Math.max(0, Math.min(1, (state[f][dims[k]]-mn)/rng)); c++; }
        raw[f] = c ? s/c : 0;
      }
      // temporal smoothing (sensor + mechanical lag)
      const clo = new Float32Array(F); let p = raw[0]||0;
      for (let f=0; f<F; f++){ p += 0.22*(raw[f]-p); clo[f]=p; }
      // contact engages only past a closure threshold (object seated in grasp)
      const force = new Float32Array(F); let peak=0;
      for (let f=0; f<F; f++){ force[f]=OV.smoothstep(0.30, 0.84, clo[f]); if(force[f]>peak)peak=force[f]; }
      out[key] = { closure: clo, force, peak, active: peak > 0.05 };
    }
    return out;
  };

})();
