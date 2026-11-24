'use strict';

/* ───────────────────────── palette & helpers ───────────────────────── */
const C = {
  bg: '#10141b', panel: '#151a22', ink: '#e8e3d8', mute: '#566070', dim: '#2a313d',
  line: '#242c38', amber: '#e3a857', teal: '#6cc0d0', violet: '#b99cf0', rose: '#e3798c', green: '#93cf8f',
};
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function mix(a, b, t) {
  t = Math.max(0, Math.min(1, t));
  const A = hex2rgb(a), B = hex2rgb(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
}
function rgba(h, a) { const [r, g, b] = hex2rgb(h); return `rgba(${r},${g},${b},${a})`; }
function diverge(v) { v = Math.max(-1, Math.min(1, v)); return v >= 0 ? mix('#141922', C.amber, v) : mix('#141922', C.teal, -v); }
function ramp(t) { return t < 0.5 ? mix(C.rose, C.amber, t * 2) : mix(C.amber, C.teal, (t - 0.5) * 2); }
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let anim = null;
function stopAnim() { if (anim) cancelAnimationFrame(anim); anim = null; }
function loop(fn) {
  if (REDUCED) { fn(0); return; }
  const t0 = performance.now();
  const step = t => { fn((t - t0) / 1000); anim = requestAnimationFrame(step); };
  anim = requestAnimationFrame(step);
}

function mkCanvas(host, w, h) {
  const c = document.createElement('canvas');
  const d = window.devicePixelRatio || 1;
  c.width = w * d; c.height = h * d;
  c.style.aspectRatio = `${w} / ${h}`;
  host.appendChild(c);
  const x = c.getContext('2d');
  x.scale(d, d);
  x.font = '12px ui-sans-serif, system-ui';
  x.textBaseline = 'middle';
  return x;
}
function txt(x, s, px, py, color = C.mute, align = 'left', font) {
  x.save(); if (font) x.font = font; x.fillStyle = color; x.textAlign = align; x.fillText(s, px, py); x.restore();
}
function bars(x, px, py, w, h, vals, color, opts = {}) {
  const n = vals.length, gap = opts.gap ?? 3, bw = (w - gap * (n - 1)) / n;
  const max = opts.max ?? Math.max(...vals.map(Math.abs));
  x.strokeStyle = C.line; x.beginPath(); x.moveTo(px, py + h); x.lineTo(px + w, py + h); x.stroke();
  vals.forEach((v, i) => {
    const bh = (Math.abs(v) / max) * h;
    x.fillStyle = typeof color === 'function' ? color(i, v) : color;
    x.fillRect(px + i * (bw + gap), py + h - bh, bw, bh);
  });
  if (opts.label) txt(x, opts.label, px, py - 12, C.mute);
}
function arrow(x, x0, y0, x1, y1, color, lw = 1.6, head = 7) {
  x.strokeStyle = color; x.fillStyle = color; x.lineWidth = lw;
  x.beginPath(); x.moveTo(x0, y0); x.lineTo(x1, y1); x.stroke();
  const a = Math.atan2(y1 - y0, x1 - x0);
  x.beginPath(); x.moveTo(x1, y1);
  x.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4));
  x.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4));
  x.closePath(); x.fill();
}
const svg = (w, h, inner) => `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>
  <filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>${inner}</svg>`;
const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.c || C.ink}" font-size="${o.s || 13}" text-anchor="${o.a || 'middle'}" font-family="${o.f || 'Iowan Old Style, Palatino, Georgia, serif'}" ${o.i ? 'font-style="italic"' : ''} dominant-baseline="middle">${s}</text>`;

/* ───────────────────────── the ten choices ───────────────────────── */
const R = String.raw;
const CHOICES = [
  {
    id: 'locus', title: 'Locus', viz: 'locus',
    q: 'Which object does S-space live in?',
    context: R`Mamba-2, per head: $h_t = a_t\,h_{t-1} + x_t B_t^{\top}$, $\;y_t = h_t C_t$, with $h_t \in \mathbb{R}^{P\times N}$. J-space lives in the residual stream $r_t \in \mathbb{R}^{d}$. The question is where the S-space boundary sits in this diagram.`,
    options: [
      { k: 'raw', label: 'Raw recurrent state', rec: true, tex: R`\mathcal{S} \subseteq \mathbb{R}^{P\times N},\qquad h_t = a_t\,h_{t-1} + x_t B_t^{\top}`, note: 'Keeps S-space distinct from J-space. The readout is a lens onto the space, not the space itself.' },
      { k: 'readout', label: 'Read-out state', tex: R`\mathcal{S} \subseteq \operatorname{im}\big(h \mapsto h\,C_t\big) \subseteq \mathbb{R}^{P}`, note: 'Only what is exposed at this step. Easy to compare with J-space, but it tends to collapse into it.' },
      { k: 'gates', label: 'Selective gates', tex: R`(\Delta_t,\,B_t,\,C_t) = f_\theta(x_t)`, note: 'Describes the control policy (what gets written and read), not the content being held.' },
      { k: 'joint', label: 'Joint state ⊕ residual', tex: R`\mathcal{S} \subseteq \mathbb{R}^{d} \oplus \mathbb{R}^{PN},\qquad (r_t,\ \operatorname{vec} h_t)`, note: 'Matches Wang & Reid’s joint readout, but you can no longer say what is state-only.' },
    ],
  },
  {
    id: 'criterion', title: 'Membership criterion', viz: 'criterion',
    q: 'What property qualifies a direction for S-space?',
    context: R`Wang & Reid report that recovering content, keeping it over time, and redirecting the model are related but not identical. Each property defines a different set of directions, and S-space is some region of the diagram.`,
    options: [
      { k: 'readable', label: 'Readable', tex: R`\exists\,L:\ \Pr\big[\arg\max L(h_t) = c\big] \ge \gamma`, note: 'The weakest test. Anything a probe can decode qualifies.' },
      { k: 'persistent', label: 'Persistent', tex: R`\frac{\lVert P_{\mathcal S}\,\Phi_{t+k,t}\,h_t\rVert}{\lVert P_{\mathcal S}\,h_t\rVert} \ge \rho\quad \forall k \le K`, note: 'Content that survives the decay over a horizon K.' },
      { k: 'causal', label: 'Causal', tex: R`\big|\,\mathbb{E}\,\Delta\log p\big(y \mid h_t + \alpha v\big)\big| \ge \delta,\quad v\in\mathcal S`, note: 'Steering along the direction changes the output.' },
      { k: 'verbalizable', label: 'Verbalizable', tex: R`\mathcal S = \operatorname{cone}\Big\{ \big(\tfrac{\partial \ell_{t+k}}{\partial h_t}\big)^{\!\top} e_w \Big\}`, note: 'J-space’s own criterion, applied to the state.' },
      { k: 'pc', label: 'Persistent ∧ causal', rec: true, tex: R`\mathcal S = \mathcal S_{\text{persist}} \cap \mathcal S_{\text{causal}}`, note: 'Leaves out verbalizability, which is the property that would make S-space just “J-space on the state”.' },
      { k: 'all', label: 'All four', tex: R`\mathcal S = \mathcal S_{\text{read}} \cap \mathcal S_{\text{persist}} \cap \mathcal S_{\text{causal}} \cap \mathcal S_{\text{verbal}}`, note: 'The strictest version. It may come out empty, or end up restating J-space.' },
    ],
  },
  {
    id: 'instrument', title: 'Instrument', viz: 'instrument',
    q: 'How is S-space measured?',
    context: R`Each instrument returns a basis $U$ for $\mathcal S$. They differ in what they reference: the vocabulary, the dynamics, the data distribution, or a set of labels.`,
    options: [
      { k: 'jacobian', label: 'State Jacobian lens', rec: true, tex: R`J^{(k)}_t = \frac{\partial\,\operatorname{logits}_{t+k}}{\partial \operatorname{vec} h_t} \in \mathbb{R}^{|V|\times PN}`, note: 'Closest to J-space, and Wang & Reid have shown it works. The spectral analysis goes on top of it.' },
      { k: 'gramian', label: 'Gramians', tex: R`\begin{gathered}W_c = \textstyle\sum_{s\le t}\Phi_{t,s}B_sB_s^{\top}\Phi_{t,s}^{\top} \\ W_o = \textstyle\sum_{u\ge t}\Phi_{u,t}^{\top}C_uC_u^{\top}\Phi_{u,t}\end{gathered}`, note: 'Uses only the dynamics, with no reference to the vocabulary. Non-trivial even though Ā is diagonal.' },
      { k: 'covariance', label: 'State covariance', tex: R`\Sigma = \mathbb{E}\big[\operatorname{vec}h_t \operatorname{vec}h_t^{\top}\big] = U\Lambda U^{\top}`, note: 'Cheap to compute, but high variance does not mean importance.' },
      { k: 'sae', label: 'Sparse autoencoder', tex: R`\operatorname{vec}h_t \approx D f_t,\quad f_t \ge 0,\ \lVert f_t\rVert_0 \ll PN`, note: 'Can discover content that is not tokens, but the features need labels afterwards.' },
      { k: 'probe', label: 'Supervised probe', tex: R`\hat c = \operatorname{softmax}\!\big(W \operatorname{vec} h_t\big)`, note: 'Finds only what you go looking for.' },
    ],
  },
  {
    id: 'direction', title: 'Directionality', viz: 'direction',
    q: 'Does S-space look forward, backward, or both?',
    context: R`A state direction can be written by past inputs (controllability), read by future outputs (observability), or both. The cumulative transition is $\Phi_{u,s} = \prod_{\tau=s+1}^{u}\bar A_\tau$.`,
    options: [
      { k: 'forward', label: 'Forward · observability', tex: R`W_o = \sum_{u\ge t}\Phi_{u,t}^{\top} C_u C_u^{\top} \Phi_{u,t}`, note: 'What the state will go on to affect.' },
      { k: 'backward', label: 'Backward · controllability', tex: R`W_c = \sum_{s\le t}\Phi_{t,s} B_s B_s^{\top} \Phi_{t,s}^{\top}`, note: 'What wrote into the state.' },
      { k: 'balanced', label: 'Balanced (both)', rec: true, tex: R`T W_c T^{\top} = T^{-\top} W_o T^{-1} = \operatorname{diag}(\sigma_1,\dots,\sigma_n)`, note: 'Hankel singular values σᵢ rank the directions that are both written and read. Directions that are only written or only read are not workspace.' },
    ],
  },
  {
    id: 'shape', title: 'Shape', viz: 'shape',
    q: 'Fixed subspace, per-prompt subspace, or curved manifold?',
    context: R`J-space uses fixed directions with sparse, non-negative activation. S-space could copy that structure or depart from it.`,
    options: [
      { k: 'global', label: 'Global basis, sparse occupancy', rec: true, tex: R`\begin{gathered}\operatorname{vec}h_t \approx U z_t \\ U \text{ fixed per layer},\ \ z_t \ge 0 \text{ sparse}\end{gathered}`, note: 'Mirrors J-space, so the two can be compared head-to-head.' },
      { k: 'perprompt', label: 'Per-prompt subspace', tex: R`\operatorname{vec}h_t \approx U(p)\,z_t`, note: 'More expressive, but hard to compare across prompts.' },
      { k: 'manifold', label: 'Input-dependent manifold', tex: R`\operatorname{vec}h_t \in \mathcal M = \{\varphi(z) : z\in\mathbb{R}^{r}\}`, note: 'Most faithful if the state really is curved. Hardest to steer.' },
    ],
  },
  {
    id: 'granularity', title: 'Granularity', viz: 'granularity',
    q: 'What is the unit of an S-space?',
    context: R`In Mamba-2, every channel of a head shares one decay $a_t$. In Mamba-1, each channel decays at its own rate.`,
    options: [
      { k: 'channel', label: 'Per channel', tex: R`\mathcal S_{\ell,h,p} \subseteq \mathbb{R}^{N}\quad(\text{one row of } h)`, note: 'The finest grain. Most natural for Mamba-1.' },
      { k: 'head', label: 'Per head', rec: true, tex: R`\mathcal S_{\ell,h} \subseteq \mathbb{R}^{P\times N}`, note: 'One shared decay makes the head a natural memory slot.' },
      { k: 'layer', label: 'Per layer', tex: R`\mathcal S_{\ell} \subseteq \textstyle\bigoplus_h \mathbb{R}^{P\times N}`, note: 'Matches the per-layer J-lens.' },
      { k: 'model', label: 'Whole model', tex: R`\mathcal S \subseteq \textstyle\bigoplus_{\ell,h} \mathbb{R}^{P\times N}`, note: 'A single global workspace. Very large.' },
    ],
  },
  {
    id: 'timescale', title: 'Timescale', viz: 'timescale',
    q: 'How is time scale handled?',
    context: R`Each head has an effective time constant $\tau = -1/\log \bar a = 1/(\Delta|A|)$ and half-life $t_{1/2} = \tau\ln 2$. Because $\Delta_t$ is selective, $\tau$ depends on the input. Use corpus statistics such as the median.`,
    options: [
      { k: 'single', label: 'One undivided space', tex: R`\mathcal S`, note: 'Simplest. Throws away the decay structure.' },
      { k: 'bands', label: 'Half-life bands', rec: true, tex: R`\begin{gathered}\mathcal S = \mathcal S_{\text{fast}} \oplus \mathcal S_{\text{mid}} \oplus \mathcal S_{\text{slow}} \\ \mathcal S_b = \operatorname{span}\{\text{heads } h : \tau_h \in I_b\}\end{gathered}`, note: 'Turns tenet 1 into a test: if the bands hold the same content, the tenet is false.' },
      { k: 'continuous', label: 'τ as a coordinate', tex: R`\mathcal S \subseteq \mathbb{R}^{PN}\times\mathbb{R}_{>0},\qquad (h,\ \tau)`, note: 'No arbitrary band edges, but harder to report.' },
    ],
  },
  {
    id: 'content', title: 'Unit of content', viz: 'content',
    q: 'What counts as one item in S-space?',
    context: R`J-space content is indexed by vocabulary tokens $w \in V$. S-space might hold things that are not words, such as counters, bracket depth, or a sense of “who is speaking”.`,
    options: [
      { k: 'tokens', label: 'Vocabulary tokens', tex: R`e_w,\quad w\in V`, note: 'A direct, like-for-like comparison with J-space.' },
      { k: 'concepts', label: 'Labeled concepts', tex: R`c \in \mathcal C\quad(\text{curated set})`, note: 'Interpretable, but limited to the labels you chose.' },
      { k: 'sae', label: 'SAE features', tex: R`f_i,\quad i \le m`, note: 'Can find content that is not words. Labeled afterwards.' },
      { k: 'modes', label: 'Unlabeled spectral modes', tex: R`\sigma_i\,u_i`, note: 'Pure geometry, with no meaning attached.' },
      { k: 'staged', label: 'Tokens, then SAE features', rec: true, tex: R`V \;\longrightarrow\; \{f_i\}_{i\le m}`, note: 'Start comparable to J-space, then widen to content that is not words.' },
    ],
  },
  {
    id: 'arch', title: 'Scope', viz: 'arch',
    q: 'Which architectures must the definition cover?',
    context: R`The shape of $\bar A_t$ sets what an eigen-analysis can see. Diagonal and scalar transitions have trivial eigenvectors. Rotations add phase, and delta-rule transitions are dense.`,
    options: [
      { k: 'm2', label: 'Mamba-2 only', tex: R`\bar A_t = a_t\, I`, note: 'The cleanest case.' },
      { k: 'mamba', label: 'Mamba-1 / 2 / 3', tex: R`\operatorname{diag}(e^{\Delta_t A}),\ \ a_tI,\ \ e^{-\Delta_t\lambda}R(\theta_t)`, note: 'The whole Mamba family. Mamba-3 adds phase.' },
      { k: 'linear', label: 'All linear RNNs', tex: R`\begin{gathered}\text{Mamba-1/2/3},\ \ I-\beta_t k_t k_t^{\top}\ (\text{DeltaNet}) \\ \operatorname{diag}(w_t)\ (\text{RWKV}),\ \ldots\end{gathered}`, note: 'The most general version, and the weakest commitments.' },
      { k: 'hybrid', label: 'Mamba-2 in hybrids', tex: R`\text{M}^{k}\,\text{A}\,\text{M}^{k}\,\text{A}\cdots`, note: 'Where S-space and J-space actually meet.' },
      { k: 'staged', label: 'Staged', rec: true, tex: R`\begin{gathered}\text{Mamba-2} \to \text{hybrids} \to \text{linear RNNs} \\ \text{Mamba-3 as phase extension}\end{gathered}`, note: 'Define it on the clean case, then test the hand-off, then generalize.' },
    ],
  },
  {
    id: 'falsify', title: 'Falsifier', viz: 'falsify',
    q: 'What result would prove S-space wrong?',
    context: R`A definition that cannot fail is only a name. $R(\cdot)$ is the rate of recovering concepts. The steering test uses $\Pi_{J^\perp}$, the projection that removes the J-space component.`,
    options: [
      { k: 'weak', label: 'Weak: redundancy', tex: R`\begin{gathered}\Delta R = R(J_{\text{res}}\oplus J_{\text{state}}) - R(J_{\text{res}}) \\ \Delta R \le \varepsilon \;\Rightarrow\; \text{redundant}\end{gathered}`, note: 'The state lens adds nothing beyond the residual J-lens.' },
      { k: 'strong', label: 'Strong: no independent effect', tex: R`\begin{gathered}\mathbb{E}\big|\Delta\log p\big(y\mid h + \alpha\,\Pi_{J^{\perp}} v\big)\big| \le \delta \\ \Rightarrow\ \text{no S-only causality}\end{gathered}`, note: 'Steering S-space has no effect beyond what steering J-space does.' },
      { k: 'both', label: 'Both, stated in advance', rec: true, tex: R`\neg(\Delta R \le \varepsilon) \;\wedge\; \neg(\mathbb{E}|\Delta\log p| \le \delta)`, note: 'Commit to both thresholds before running the experiments.' },
      { k: 'none', label: 'No falsifier (exploratory)', tex: R`\varnothing`, note: 'Then S-space is a renamed state lens.' },
    ],
  },
];

/* ───────────────────────── visualizations ───────────────────────── */
const VIZ = {};

VIZ.locus = (host, sel) => {
  const hl = { raw: ['state'], readout: ['read'], gates: ['gates'], joint: ['state', 'resid'] }[sel] || [];
  const on = g => hl.includes(g);
  const st = g => on(g) ? C.amber : '#4a5363';
  const sw = g => on(g) ? 2.2 : 1.2;
  const glow = g => on(g) ? 'filter="url(#glow)"' : '';
  let cells = '';
  for (let i = 0; i < 4; i++) for (let j = 0; j < 6; j++) {
    const v = 0.2 + 0.7 * (((i * 7 + j * 3) % 5) / 5);
    cells += `<rect x="${546 + j * 15}" y="${118 + i * 15}" width="12" height="12" rx="2" fill="${on('state') ? mix('#1a1f29', C.amber, v) : mix('#1a1f29', '#55607a', v * 0.6)}"/>`;
  }
  const path = (d, g, label, lx, ly) => `<path d="${d}" fill="none" stroke="${on(g) ? C.amber : '#4a5363'}" stroke-width="${on(g) ? 1.8 : 1.2}" class="flow" marker-end="url(#ah)"/>` + (label ? T(lx, ly, label, { c: on(g) ? C.amber : C.mute, s: 12, i: true }) : '');
  host.innerHTML = svg(860, 300, `
    ${path('M110,150 L440,150', 'write', 'xₜ Bₜᵀ', 300, 140)}
    ${path('M76,128 C76,70 120,64 168,64', 'gates')}
    ${path('M250,90 C250,160 330,190 352,214', 'gates', 'Δₜ → āₜ', 296, 176)}
    ${path('M310,90 C330,120 400,128 446,138', 'gates', 'Bₜ', 380, 112)}
    ${path('M340,58 C520,30 650,60 690,128', 'read', 'Cₜ', 560, 40)}
    ${path('M282,234 L340,234', 'state')}
    ${path('M378,230 C420,220 446,190 456,168', 'state')}
    ${path('M478,150 L536,150', 'state')}
    ${path('M642,150 L670,150', 'read')}
    ${path('M708,150 L776,150', 'read', 'yₜ', 742, 138)}
    ${path('M594,182 C594,262 300,280 236,256', 'state', 'next step', 420, 284)}
    <g ${glow('x')}><rect x="40" y="128" width="70" height="44" rx="8" fill="#1a1f29" stroke="#4a5363"/>${T(75, 150, 'xₜ', { s: 16, i: true })}</g>
    <g ${glow('gates')}><rect x="168" y="40" width="174" height="50" rx="10" fill="#1a1f29" stroke="${st('gates')}" stroke-width="${sw('gates')}"/>${T(255, 59, 'Δₜ, Bₜ, Cₜ = f(xₜ)', { s: 14, i: true })}${T(255, 78, 'selective gates', { s: 10.5, c: C.mute, f: 'ui-monospace, monospace' })}</g>
    <g><rect x="190" y="214" width="92" height="40" rx="8" fill="#1a1f29" stroke="#4a5363"/>${T(236, 234, 'hₜ₋₁', { s: 15, i: true })}</g>
    <g><circle cx="360" cy="232" r="19" fill="#1a1f29" stroke="#4a5363"/>${T(360, 232, 'āₜ·', { s: 14, i: true })}</g>
    <g><circle cx="460" cy="150" r="18" fill="#1a1f29" stroke="#4a5363"/>${T(460, 150, '+', { s: 18 })}</g>
    <g ${glow('state')}><rect x="538" y="110" width="104" height="72" rx="8" fill="#141922" stroke="${st('state')}" stroke-width="${sw('state')}"/>${cells}${T(590, 98, 'hₜ ∈ ℝᴾˣᴺ', { s: 13, i: true, c: on('state') ? C.amber : C.ink })}</g>
    <g ${glow('read')}><circle cx="690" cy="150" r="19" fill="#1a1f29" stroke="${st('read')}" stroke-width="${sw('read')}"/>${T(690, 150, '·Cₜ', { s: 14, i: true })}</g>
    <g ${glow('resid')}><rect x="780" y="30" width="30" height="240" rx="8" fill="${on('resid') ? 'rgba(227,168,87,.12)' : '#161b24'}" stroke="${st('resid')}" stroke-width="${sw('resid')}"/>
      <text x="795" y="150" fill="${on('resid') ? C.amber : C.mute}" font-size="12" text-anchor="middle" transform="rotate(-90 795 150)" font-family="ui-monospace, monospace">residual rₜ · J-space</text></g>
  `);
};

VIZ.criterion = (host, sel) => {
  const W = 860, H = 370, cx = 430, cy = 196;
  const S = {
    readable: { c: C.teal, x: -95, y: 25, r: -40, name: 'readable' },
    persistent: { c: C.amber, x: -30, y: -20, r: -40, name: 'persistent' },
    causal: { c: C.rose, x: 30, y: -20, r: 40, name: 'causal' },
    verbalizable: { c: C.violet, x: 95, y: 25, r: 40, name: 'verbalizable (J-space-like)' },
  };
  const pick = { readable: ['readable'], persistent: ['persistent'], causal: ['causal'], verbalizable: ['verbalizable'], pc: ['persistent', 'causal'], all: Object.keys(S) }[sel] || [];
  const ell = (k, attrs) => `<ellipse cx="0" cy="0" rx="180" ry="92" transform="translate(${cx + S[k].x} ${cy + S[k].y}) rotate(${S[k].r})" ${attrs}/>`;
  let defs = '<defs>' + Object.keys(S).map(k => `<clipPath id="cp-${k}">${ell(k, '')}</clipPath>`).join('') + '</defs>';
  let fill = '';
  if (pick.length) {
    let inner = `<rect x="0" y="0" width="${W}" height="${H}" fill="url(#rg)" opacity=".85"/>`;
    for (const k of pick) inner = `<g clip-path="url(#cp-${k})">${inner}</g>`;
    fill = `<radialGradient id="rg" cx="50%" cy="45%" r="60%"><stop offset="0" stop-color="${C.amber}" stop-opacity=".75"/><stop offset="1" stop-color="${C.amber}" stop-opacity=".25"/></radialGradient>` + inner;
  }
  const outlines = Object.keys(S).map(k => ell(k, `fill="${rgba(S[k].c, 0.05)}" stroke="${S[k].c}" stroke-width="${pick.includes(k) ? 2.4 : 1.1}" stroke-opacity="${pick.includes(k) || !pick.length ? 1 : 0.45}"`)).join('');
  const legend = Object.keys(S).map((k, i) => `<g transform="translate(${40 + i * 200} 22)"><rect width="12" height="12" rx="3" fill="${S[k].c}" opacity="${pick.includes(k) ? 1 : 0.5}" y="-6"/>${T(20, 0, S[k].name, { a: 'start', s: 13, c: pick.includes(k) ? C.ink : C.mute })}</g>`).join('');
  const lab = pick.length ? T(cx, pick.length === 4 ? cy - 58 : pick.length === 2 ? cy + 6 : cy + 150, '𝒮', { s: 26, c: '#fff8ea', i: true }) : '';
  host.innerHTML = svg(W, H, defs + fill + outlines + lab + legend);
};

VIZ.instrument = (host, sel) => {
  const x = mkCanvas(host, 860, 300);
  const r = rng(7);
  const spectrumX = 620, spectrumW = 210;
  if (sel === 'jacobian' || !sel) {
    const rows = 18, cols = 46, cw = 10, ch = 13, ox = 110, oy = 30;
    const U = [0, 1, 2].map(() => Array.from({ length: rows }, () => gauss(r)));
    const V = [0, 1, 2].map(() => Array.from({ length: cols }, () => gauss(r)));
    const s = [1.0, 0.6, 0.35];
    const toks = ['Paris', 'rugby', 'France', '(', ')', 'she', 'cat', 'seven', 'list', 'item', 'red', 'goal', 'tea', 'ball', 'he', 'blue', ',', 'end'];
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      let v = 0; for (let k = 0; k < 3; k++) v += s[k] * U[k][i] * V[k][j];
      v = v / 2.2 + gauss(r) * 0.06;
      x.fillStyle = diverge(v); x.fillRect(ox + j * cw, oy + i * ch, cw - 1, ch - 1);
    }
    toks.forEach((t, i) => txt(x, t, ox - 8, oy + i * ch + ch / 2, i < 3 ? C.ink : C.mute, 'right', '11px ui-monospace, monospace'));
    txt(x, 'state coordinates  vec hₜ  →', ox, oy + rows * ch + 14, C.mute);
    txt(x, '∂ logitsₜ₊ₖ / ∂ hₜ', ox + cols * cw / 2, 14, C.ink, 'center', 'italic 14px Iowan Old Style, Georgia, serif');
    const sv = [3.1, 1.9, 1.15, 0.32, 0.27, 0.23, 0.2, 0.18, 0.16, 0.14, 0.13, 0.12];
    bars(x, spectrumX, 50, spectrumW, 190, sv, (i) => i < 3 ? C.amber : C.dim, { label: 'singular values σᵢ(J)' });
    txt(x, 'low rank ⇒ a few directions carry the readout', spectrumX, 262, C.mute);
  } else if (sel === 'gramian') {
    const ox = 280, oy = 150;
    const E = (rx, ry, a, col, dash) => { x.save(); x.translate(ox, oy); x.rotate(a); x.setLineDash(dash || []); x.strokeStyle = col; x.lineWidth = 2; x.beginPath(); x.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2); x.stroke(); x.fillStyle = rgba(col.startsWith('#') ? col : C.amber, 0.06); x.fill(); x.restore(); };
    E(150, 42, -0.35, C.teal); E(130, 50, 0.75, C.violet);
    E(92, 34, 0.12, C.amber, [5, 4]);
    arrow(x, ox, oy, ox + 92 * Math.cos(0.12), oy + 92 * Math.sin(0.12), C.amber, 2);
    arrow(x, ox, oy, ox - 34 * Math.sin(0.12), oy + 34 * Math.cos(0.12), C.amber, 1.4);
    txt(x, 'Wc — written (controllability)', 70, 40, C.teal, 'left', 'italic 13px Iowan Old Style, Georgia, serif');
    txt(x, 'Wo — read (observability)', 70, 60, C.violet, 'left', 'italic 13px Iowan Old Style, Georgia, serif');
    txt(x, 'balanced axes', 70, 80, C.amber, 'left', 'italic 13px Iowan Old Style, Georgia, serif');
    bars(x, spectrumX, 50, spectrumW, 190, [2.6, 1.7, 0.9, 0.45, 0.22, 0.1, 0.06, 0.04, 0.03, 0.02], (i) => i < 3 ? C.amber : C.dim, { label: 'Hankel singular values σᵢ' });
    txt(x, 'truncate where σ drops ⇒ dim 𝒮', spectrumX, 262, C.mute);
  } else if (sel === 'covariance') {
    const ox = 280, oy = 150, a = -0.45;
    for (let i = 0; i < 650; i++) {
      const u = gauss(r) * 2.3, v = gauss(r) * 0.65;
      const px = ox + 40 * (u * Math.cos(a) - v * Math.sin(a)), py = oy + 40 * (u * Math.sin(a) + v * Math.cos(a));
      x.fillStyle = rgba(C.teal, 0.35); x.beginPath(); x.arc(px, py, 1.6, 0, 7); x.fill();
    }
    arrow(x, ox, oy, ox + 120 * Math.cos(a), oy + 120 * Math.sin(a), C.amber, 2.2);
    arrow(x, ox, oy, ox - 38 * Math.sin(a), oy + 38 * Math.cos(a), C.amber, 1.4);
    txt(x, 'u₁', ox + 132 * Math.cos(a), oy + 132 * Math.sin(a), C.amber, 'center', 'italic 14px Iowan Old Style, Georgia, serif');
    bars(x, spectrumX, 50, spectrumW, 190, [5.3, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33, 0.3, 0.28, 0.26], (i) => i < 1 ? C.amber : C.dim, { label: 'eigenvalues λᵢ(Σ)' });
    txt(x, 'caution: big variance ≠ big influence', spectrumX, 262, C.rose);
  } else if (sel === 'sae') {
    const h = Array.from({ length: 30 }, () => gauss(r) * 0.6);
    const f = Array.from({ length: 18 }, (_, i) => [3, 9, 14].includes(i) ? 0.6 + r() * 0.4 : 0);
    bars(x, 40, 70, 200, 150, h.map(Math.abs), (i) => h[i] > 0 ? C.amber : C.teal, { gap: 2, label: 'vec hₜ' });
    txt(x, '≈', 268, 145, C.ink, 'center', '28px Iowan Old Style, Georgia, serif');
    for (let i = 0; i < 30; i++) for (let j = 0; j < 18; j++) { x.fillStyle = diverge(gauss(r) * (f[j] ? 0.8 : 0.35)); x.fillRect(300 + j * 11, 40 + i * 7, 10, 6); }
    [3, 9, 14].forEach(j => { x.strokeStyle = C.amber; x.lineWidth = 1.5; x.strokeRect(299 + j * 11, 39, 12, 212); });
    txt(x, 'D (dictionary)', 300, 24, C.mute);
    txt(x, '·', 520, 145, C.ink, 'center', '28px Iowan Old Style, Georgia, serif');
    bars(x, spectrumX - 80, 70, 280, 150, f, (i) => f[i] ? C.amber : C.dim, { gap: 3, max: 1, label: 'fₜ  ≥ 0, sparse' });
    txt(x, 'only 3 of 18 features active', spectrumX - 80, 262, C.mute);
  } else if (sel === 'probe') {
    const ox = 290, oy = 150;
    for (let i = 0; i < 360; i++) {
      const c = i % 2, u = gauss(r) * 1.1 + (c ? 1.2 : -1.2), v = gauss(r) * 1.1;
      x.fillStyle = rgba(c ? C.amber : C.teal, 0.55); x.beginPath(); x.arc(ox + u * 45, oy + v * 45 + u * 10, 2.2, 0, 7); x.fill();
    }
    x.strokeStyle = C.ink; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(ox - 25, oy - 130); x.lineTo(ox + 25, oy + 130); x.stroke(); x.setLineDash([]);
    arrow(x, ox, oy, ox + 90, oy - 17, C.ink, 2);
    txt(x, 'w', ox + 100, oy - 22, C.ink, 'left', 'italic 15px Iowan Old Style, Georgia, serif');
    txt(x, 'class c = "sport"', 70, 40, C.amber); txt(x, 'class c = "not sport"', 70, 60, C.teal);
    txt(x, 'one direction per label you', spectrumX, 120, C.mute); txt(x, 'thought to ask about', spectrumX, 138, C.mute);
  }
};

VIZ.direction = (host, sel) => {
  const show = { forward: ['read'], backward: ['write'], balanced: ['read', 'write', 'bal'] }[sel] || [];
  const on = k => show.includes(k);
  let s = '';
  const xs = i => 50 + i * 62, hx = xs(4), hy = 165;
  for (let i = 0; i < 9; i++) {
    const lab = i === 4 ? 't' : i < 4 ? `t−${4 - i}` : `t+${i - 4}`;
    if (i <= 4) {
      const w = Math.pow(0.72, 4 - i);
      s += `<line x1="${xs(i)}" y1="66" x2="${hx}" y2="${hy - 28}" stroke="${on('write') ? C.teal : '#3a4250'}" stroke-opacity="${on('write') ? 0.25 + 0.75 * w : 0.5}" stroke-width="${on('write') ? 1 + 3 * w : 1}" class="${on('write') ? 'flow' : ''}"/>`;
      s += `<rect x="${xs(i) - 20}" y="40" width="40" height="26" rx="6" fill="#1a1f29" stroke="#3a4250"/>` + T(xs(i), 53, `x<tspan font-size="9" dy="4">${lab}</tspan>`, { s: 13, i: true });
    }
    if (i >= 4) {
      const w = Math.pow(0.72, i - 4);
      s += `<line x1="${hx}" y1="${hy + 28}" x2="${xs(i)}" y2="252" stroke="${on('read') ? C.violet : '#3a4250'}" stroke-opacity="${on('read') ? 0.25 + 0.75 * w : 0.5}" stroke-width="${on('read') ? 1 + 3 * w : 1}" class="${on('read') ? 'flow' : ''}"/>`;
      s += `<rect x="${xs(i) - 20}" y="252" width="40" height="26" rx="6" fill="#1a1f29" stroke="#3a4250"/>` + T(xs(i), 265, `y<tspan font-size="9" dy="4">${lab}</tspan>`, { s: 13, i: true });
    }
  }
  s += `<circle cx="${hx}" cy="${hy}" r="28" fill="#141922" stroke="${on('bal') ? C.amber : '#5a6475'}" stroke-width="2" ${on('bal') ? 'filter="url(#glow)"' : ''}/>` + T(hx, hy, 'hₜ', { s: 18, i: true });
  s += T(150, 105, on('write') ? 'written: Φₜ,ₛ Bₛ' : '', { c: C.teal, s: 13, i: true });
  s += T(440, 215, on('read') ? 'read: Cᵤ Φᵤ,ₜ' : '', { c: C.violet, s: 13, i: true });
  const ex = 720, ey = 160;
  s += `<line x1="590" y1="20" x2="590" y2="290" stroke="${C.line}"/>`;
  s += `<ellipse cx="${ex}" cy="${ey}" rx="118" ry="38" transform="rotate(-22 ${ex} ${ey})" fill="${rgba(C.teal, on('write') ? 0.12 : 0.03)}" stroke="${C.teal}" stroke-opacity="${on('write') ? 1 : 0.3}" stroke-width="2"/>`;
  s += `<ellipse cx="${ex}" cy="${ey}" rx="112" ry="44" transform="rotate(38 ${ex} ${ey})" fill="${rgba(C.violet, on('read') ? 0.12 : 0.03)}" stroke="${C.violet}" stroke-opacity="${on('read') ? 1 : 0.3}" stroke-width="2"/>`;
  if (on('bal')) {
    s += `<ellipse cx="${ex}" cy="${ey}" rx="80" ry="34" transform="rotate(8 ${ex} ${ey})" fill="${rgba(C.amber, 0.15)}" stroke="${C.amber}" stroke-dasharray="5 4" stroke-width="2"/>`;
    s += `<line x1="${ex}" y1="${ey}" x2="${ex + 79}" y2="${ey + 11}" stroke="${C.amber}" stroke-width="2.4" marker-end="url(#ah)"/>`;
    s += `<line x1="${ex}" y1="${ey}" x2="${ex - 5}" y2="${ey + 33}" stroke="${C.amber}" stroke-width="1.6" marker-end="url(#ah)"/>`;
    s += T(ex + 96, ey + 22, 'σ₁', { c: C.amber, s: 14, i: true });
  }
  s += T(ex, 30, 'W<tspan font-size="10" dy="4">c</tspan>', { c: C.teal, s: 15, i: true, a: 'end' }) + T(ex + 10, 30, '·', { c: C.mute }) + T(ex + 20, 30, 'W<tspan font-size="10" dy="4">o</tspan>', { c: C.violet, s: 15, i: true, a: 'start' });
  host.innerHTML = svg(860, 300, s);
};

VIZ.shape = (host, sel) => {
  const W = 860, H = 340;
  const x = mkCanvas(host, W, H);
  const r = rng(11);
  const planes = [];
  const add = (n, e1, e2, col, f) => planes.push({ n, e1, e2, col, f });
  const nz = (v) => { const l = Math.hypot(...v); return v.map(c => c / l); };
  if (sel === 'perprompt') {
    add([0, 0, 0], nz([1, 0.1, 0.3]), nz([0, 1, -0.4]), C.amber);
    add([0, 0, 0], nz([1, 0.5, -0.6]), nz([-0.2, 0.6, 0.9]), C.teal);
    add([0, 0, 0], nz([0.3, -0.2, 1]), nz([0.8, 0.6, 0]), C.violet);
  } else if (sel === 'manifold') {
    add([0, 0, 0], [1, 0, 0], [0, 0, 1], C.amber, (u, v) => 0.55 * Math.sin(1.7 * u) * Math.cos(1.4 * v));
  } else {
    add([0, 0, 0], nz([1, 0.15, 0.25]), nz([0, 0.25, 1]), C.amber);
  }
  const P = (pl, u, v) => {
    const h = pl.f ? pl.f(u, v) : 0;
    return [pl.e1[0] * u + pl.e2[0] * v, pl.e1[1] * u + pl.e2[1] * v + h, pl.e1[2] * u + pl.e2[2] * v];
  };
  const trajs = planes.map((pl, pi) => Array.from({ length: 140 }, (_, k) => {
    const u = 1.25 * Math.sin(0.07 * k + pi * 2) * Math.cos(0.013 * k), v = 1.1 * Math.sin(0.045 * k + pi);
    const p = P(pl, u, v);
    return p.map(c => c + gauss(r) * 0.02);
  }));
  const occ = Array.from({ length: 140 }, (_, k) => [Math.max(0, Math.sin(0.07 * k)), Math.max(0, Math.cos(0.045 * k))]);
  loop(t => {
    x.clearRect(0, 0, W, H);
    const th = 0.6 + t * 0.25, ph = 0.95;
    const proj = ([a, b, c]) => {
      const X = a * Math.cos(th) + c * Math.sin(th), Z = -a * Math.sin(th) + c * Math.cos(th);
      const Y = b * Math.cos(ph) - Z * Math.sin(ph), Z2 = b * Math.sin(ph) + Z * Math.cos(ph);
      const f = 3.6 / (3.6 + Z2);
      return [W / 2 - 60 + X * 150 * f, H / 2 - Y * 150 * f, Z2];
    };
    planes.forEach(pl => {
      x.lineWidth = 1;
      for (let i = -6; i <= 6; i++) {
        for (const dir of [0, 1]) {
          x.beginPath();
          for (let j = -6; j <= 6; j++) {
            const u = (dir ? i : j) * 0.25, v = (dir ? j : i) * 0.25;
            const [sx, sy] = proj(P(pl, u, v));
            j === -6 ? x.moveTo(sx, sy) : x.lineTo(sx, sy);
          }
          x.strokeStyle = rgba(pl.col, 0.16); x.stroke();
        }
      }
    });
    if (sel !== 'perprompt' && sel !== 'manifold') {
      const pl = planes[0], o = proj([0, 0, 0]);
      [[pl.e1, 'u₁'], [pl.e2, 'u₂']].forEach(([e, l]) => { const p = proj(e.map(c => c * 1.4)); arrow(x, o[0], o[1], p[0], p[1], C.ink, 1.8); txt(x, l, p[0] + 8, p[1] - 8, C.ink, 'left', 'italic 14px Iowan Old Style, Georgia, serif'); });
    }
    trajs.forEach((tr, pi) => {
      const col = planes[pi].col, head = Math.floor((t * 28) % tr.length);
      for (let k = 1; k < tr.length; k++) {
        const a = proj(tr[k - 1]), b = proj(tr[k]);
        const age = (head - k + tr.length) % tr.length, alpha = Math.max(0.06, 1 - age / 70);
        x.strokeStyle = rgba(col, alpha * 0.9); x.lineWidth = 1.6;
        x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke();
      }
      const p = proj(tr[head]);
      x.fillStyle = '#fff4dc'; x.beginPath(); x.arc(p[0], p[1], 4.5, 0, 7); x.fill();
      x.strokeStyle = col; x.lineWidth = 2; x.beginPath(); x.arc(p[0], p[1], 9, 0, 7); x.stroke();
      if (sel !== 'perprompt' && sel !== 'manifold') {
        const z = occ[head];
        bars(x, W - 170, 70, 120, 90, z, (i) => [C.amber, C.teal][i], { gap: 18, max: 1, label: 'occupancy zₜ ≥ 0' });
        txt(x, 'z₁', W - 140, 175, C.mute); txt(x, 'z₂', W - 72, 175, C.mute);
      }
    });
    const lbl = { global: 'one fixed basis U · state moves inside it', perprompt: 'each prompt p gets its own U(p)', manifold: 'state rides a curved surface 𝓜 = φ(z)' }[sel] || 'one fixed basis U · state moves inside it';
    txt(x, lbl, 24, H - 18, C.mute, 'left', '12px ui-monospace, monospace');
  });
};

VIZ.granularity = (host, sel) => {
  const L = 3, Hh = 4, P = 4, N = 8, cw = 16, ch = 14;
  const taus = [[0.2, 0.55, 0.8, 0.35], [0.9, 0.15, 0.6, 0.45], [0.7, 0.3, 0.95, 0.1]];
  let s = '';
  const hx = (h) => 90 + h * 186, ly = (l) => 34 + l * 88;
  for (let l = 0; l < L; l++) {
    s += T(40, ly(l) + 30, `ℓ${l + 1}`, { c: C.mute, s: 14, i: true });
    for (let h = 0; h < Hh; h++) {
      const col = ramp(taus[l][h]);
      for (let p = 0; p < P; p++) for (let n = 0; n < N; n++) {
        s += `<rect x="${hx(h) + n * cw}" y="${ly(l) + p * ch}" width="${cw - 2}" height="${ch - 2}" rx="2" fill="${col}" fill-opacity="${0.18 + 0.5 * ((p * 3 + n * 5 + h) % 7) / 7}"/>`;
      }
      if (sel === 'channel') for (let p = 0; p < P; p++) s += `<rect x="${hx(h) - 2}" y="${ly(l) + p * ch - 2}" width="${N * cw + 2}" height="${ch + 2}" rx="3" fill="none" stroke="${C.amber}" stroke-width="1.1"/>`;
      if (sel === 'head') s += `<rect x="${hx(h) - 5}" y="${ly(l) - 5}" width="${N * cw + 8}" height="${P * ch + 8}" rx="6" fill="none" stroke="${C.amber}" stroke-width="2" filter="url(#glow)"/>`;
      if (l === 0) s += T(hx(h) + N * cw / 2, 18, `head ${h + 1}`, { c: C.mute, s: 11, f: 'ui-monospace, monospace' });
    }
    if (sel === 'layer') s += `<rect x="78" y="${ly(l) - 9}" width="${Hh * 186 - 30}" height="${P * ch + 16}" rx="8" fill="none" stroke="${C.amber}" stroke-width="2" filter="url(#glow)"/>`;
  }
  if (sel === 'model') s += `<rect x="70" y="20" width="${Hh * 186 - 14}" height="${L * 88 + 2}" rx="10" fill="none" stroke="${C.amber}" stroke-width="2.4" filter="url(#glow)"/>`;
  s += T(840, 296, 'colour = head time constant τ (Mamba-2: one decay per head)', { c: C.mute, s: 11, a: 'end', f: 'ui-monospace, monospace' });
  host.innerHTML = svg(860, 305, s);
};

VIZ.timescale = (host, sel) => {
  const W = 860, H = 300, x = mkCanvas(host, W, H);
  const ox = 60, oy = 26, pw = 540, ph = 230, K = 300;
  const taus = Array.from({ length: 14 }, (_, i) => 2 * Math.pow(250, i / 13));
  const band = tau => tau < 12 ? 0 : tau < 90 ? 1 : 2;
  const bandCol = [C.rose, C.amber, C.teal];
  const lt = tau => Math.log(tau / 2) / Math.log(250);
  x.strokeStyle = C.line; x.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const yy = oy + ph * i / 4; x.beginPath(); x.moveTo(ox, yy); x.lineTo(ox + pw, yy); x.stroke(); }
  txt(x, '1', ox - 10, oy, C.mute, 'right'); txt(x, '½', ox - 10, oy + ph / 2, C.mute, 'right'); txt(x, '0', ox - 10, oy + ph, C.mute, 'right');
  txt(x, 'lag k (tokens) →', ox + pw, oy + ph + 18, C.mute, 'right');
  txt(x, 'āᵏ = exp(−k/τ)', ox + 6, oy + ph + 18, C.mute, 'left', 'italic 13px Iowan Old Style, Georgia, serif');
  x.setLineDash([3, 4]); x.strokeStyle = '#3a4250'; x.beginPath(); x.moveTo(ox, oy + ph / 2); x.lineTo(ox + pw, oy + ph / 2); x.stroke(); x.setLineDash([]);
  taus.forEach(tau => {
    const col = sel === 'bands' ? bandCol[band(tau)] : sel === 'continuous' ? ramp(lt(tau)) : C.amber;
    x.strokeStyle = col; x.lineWidth = sel === 'single' || !sel ? 1.2 : 1.8; x.globalAlpha = sel === 'single' || !sel ? 0.55 : 0.95;
    x.beginPath();
    for (let k = 0; k <= K; k += 2) { const px = ox + pw * k / K, py = oy + ph * (1 - Math.exp(-k / tau)); k ? x.lineTo(px, py) : x.moveTo(px, py); }
    x.stroke(); x.globalAlpha = 1;
    const kh = tau * Math.LN2;
    if (kh < K) { x.fillStyle = col; x.beginPath(); x.arc(ox + pw * kh / K, oy + ph / 2, 3, 0, 7); x.fill(); }
  });
  const lx = 650;
  if (sel === 'bands') {
    [['fast', 'τ < 12', 'syntax, local', 0], ['mid', '12 ≤ τ < 90', 'clause, entity', 1], ['slow', 'τ ≥ 90', 'task, persona, topic', 2]].forEach(([n, rng_, d, i], j) => {
      const yy = 50 + j * 62; x.fillStyle = bandCol[i]; x.fillRect(lx, yy - 8, 14, 14);
      txt(x, `𝒮 ${n}`, lx + 24, yy, C.ink, 'left', 'italic 15px Iowan Old Style, Georgia, serif');
      txt(x, rng_, lx + 24, yy + 18, C.mute, 'left', '11px ui-monospace, monospace');
      txt(x, d, lx + 24, yy + 33, C.mute);
    });
  } else if (sel === 'continuous') {
    for (let i = 0; i < 180; i++) { x.fillStyle = ramp(i / 179); x.fillRect(lx + 20, 40 + i, 16, 1.2); }
    txt(x, 'τ = 2', lx + 44, 42, C.mute); txt(x, 'τ = 500', lx + 44, 218, C.mute);
    txt(x, 'each state direction', lx, 246, C.mute); txt(x, 'carries its own τ', lx, 262, C.mute);
  } else {
    txt(x, 'one 𝒮 — decay ignored', lx, 120, C.mute, 'left', 'italic 14px Iowan Old Style, Georgia, serif');
  }
  txt(x, '● = half-life  (āᵏ = ½)', lx, H - 18, C.mute);
};

VIZ.content = (host, sel) => {
  const cols = [
    { k: 'tokens', t: 'tokens w ∈ V', items: ['Paris', 'rugby', 'seven', '(', 'she', 'blue'] },
    { k: 'concepts', t: 'concepts c ∈ 𝒞', items: ['capital-of', 'sport', 'count', 'speaker', 'colour', 'negation'] },
    { k: 'sae', t: 'SAE features fᵢ', items: ['f#0412 bracket depth', 'f#1877 list index', 'f#2203 speaker turn', 'f#0039 sport', 'f#3110 ?', 'f#0927 ?'] },
    { k: 'modes', t: 'modes σᵢuᵢ', items: null },
  ];
  const act = { tokens: ['tokens'], concepts: ['concepts'], sae: ['sae'], modes: ['modes'], staged: ['tokens', 'sae'] }[sel] || [];
  let s = '';
  cols.forEach((c, i) => {
    const x0 = 22 + i * 210, on = act.includes(c.k);
    s += `<rect x="${x0}" y="14" width="194" height="272" rx="12" fill="${on ? 'rgba(227,168,87,.07)' : '#141922'}" stroke="${on ? C.amber : C.line}" stroke-width="${on ? 2 : 1}" ${on ? 'filter="url(#glow)"' : ''}/>`;
    s += T(x0 + 97, 38, c.t, { s: 15, i: true, c: on ? C.amber : C.ink });
    if (c.items) c.items.forEach((it, j) => {
      const unk = it.endsWith('?');
      s += `<rect x="${x0 + 16}" y="${60 + j * 36}" width="162" height="26" rx="13" fill="#1a1f29" stroke="${unk ? C.rose : '#3a4250'}" stroke-dasharray="${unk ? '3 3' : ''}"/>` + T(x0 + 97, 73 + j * 36, it, { s: 12, f: 'ui-monospace, monospace', c: on ? C.ink : C.mute });
    });
    else {
      for (let m = 0; m < 5; m++) {
        let d = '';
        for (let k = 0; k <= 40; k++) { const px = x0 + 20 + k * 4, py = 80 + m * 42 - 12 * Math.exp(-m * 0.35) * Math.sin(k * (0.3 + m * 0.22)); d += (k ? 'L' : 'M') + px + ',' + py; }
        s += `<path d="${d}" fill="none" stroke="${on ? C.amber : '#4a5363'}" stroke-opacity="${1 - m * 0.15}" stroke-width="1.5"/>`;
      }
    }
  });
  if (sel === 'staged') s += `<path d="M119,290 C160,330 480,330 523,292" fill="none" stroke="${C.amber}" stroke-width="2" class="flow" marker-end="url(#ah)"/>` + T(322, 322, 'then widen', { c: C.amber, s: 12, i: true });
  host.innerHTML = svg(860, 334, s);
};

VIZ.arch = (host, sel) => {
  const W = 860, H = 290, x = mkCanvas(host, W, H);
  const r = rng(3), n = 12, cs = 10;
  const mats = {
    m1: (i, j) => i === j ? 0.25 + 0.7 * ((i * 7) % 11) / 10 : 0,
    m2: (i, j) => i === j ? 0.82 : 0,
    m3: (() => { const th = Array.from({ length: n / 2 }, (_, b) => 0.3 + b * 0.35); return (i, j) => { const b = Math.floor(i / 2); if (Math.floor(j / 2) !== b) return 0; const c = Math.cos(th[b]) * 0.85, s_ = Math.sin(th[b]) * 0.85; return (i % 2 === 0) ? (j % 2 === 0 ? c : -s_) : (j % 2 === 0 ? s_ : c); }; })(),
    delta: (() => { const k = Array.from({ length: n }, () => gauss(r)); const l = Math.hypot(...k); const kk = k.map(v => v / l); return (i, j) => (i === j ? 1 : 0) - 0.85 * kk[i] * kk[j]; })(),
  };
  const panels = [
    { k: 'm1', t: 'Mamba-1', f: 'diag(exp(ΔA))' },
    { k: 'm2', t: 'Mamba-2', f: 'aₜ I' },
    { k: 'm3', t: 'Mamba-3', f: 'exp(−Δλ)·R(θ)' },
    { k: 'delta', t: 'DeltaNet', f: 'I − β k kᵀ' },
    { k: 'hybrid', t: 'Hybrid', f: 'M M M A M M M A' },
  ];
  const inc = { m2: ['m2'], mamba: ['m1', 'm2', 'm3'], linear: ['m1', 'm2', 'm3', 'delta'], hybrid: ['m2', 'hybrid'], staged: ['m2', 'hybrid', 'delta', 'm3', 'm1'] }[sel] || [];
  const order = { m2: '1', hybrid: '2', delta: '3', m1: '3', m3: 'ext' };
  panels.forEach((p, pi) => {
    const x0 = 24 + pi * 168, y0 = 50, on = inc.includes(p.k);
    x.strokeStyle = on ? C.amber : C.line; x.lineWidth = on ? 2 : 1;
    x.beginPath(); x.roundRect(x0 - 10, 14, 150, 262, 10); x.stroke();
    txt(x, p.t, x0 + 65, 32, on ? C.amber : C.ink, 'center', '15px Iowan Old Style, Georgia, serif');
    if (p.k === 'hybrid') {
      'MMMAMMMAMM'.split('').forEach((c, i) => {
        const yy = y0 + i * 18; x.fillStyle = c === 'A' ? rgba(C.violet, 0.8) : rgba(C.teal, 0.55);
        x.beginPath(); x.roundRect(x0 + 15, yy, 100, 14, 4); x.fill();
        txt(x, c === 'A' ? 'attention → J' : 'Mamba-2 → 𝒮', x0 + 65, yy + 7, '#0c0f14', 'center', '10px ui-monospace, monospace');
      });
    } else {
      const f = mats[p.k];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { x.fillStyle = diverge(f(i, j)); x.fillRect(x0 + 5 + j * cs, y0 + i * cs, cs - 1, cs - 1); }
      const ev = { m1: 'eigvecs = axes', m2: 'trivial: any vector', m3: '2-D rotating blocks', delta: 'dense, non-trivial' }[p.k];
      txt(x, ev, x0 + 65, y0 + n * cs + 18, p.k === 'delta' || p.k === 'm3' ? C.green : C.rose, 'center', '11px ui-monospace, monospace');
    }
    txt(x, p.f, x0 + 65, 252, C.mute, 'center', 'italic 13px Iowan Old Style, Georgia, serif');
    if (sel === 'staged' && order[p.k]) {
      x.fillStyle = C.amber; x.beginPath(); x.arc(x0 + 128, 22, 11, 0, 7); x.fill();
      txt(x, order[p.k], x0 + 128, 22, '#1a1408', 'center', 'bold 10px ui-monospace, monospace');
    }
  });
};

VIZ.falsify = (host, sel) => {
  const W = 860, H = 300, x = mkCanvas(host, W, H);
  const showW = sel === 'weak' || sel === 'both', showS = sel === 'strong' || sel === 'both', none = sel === 'none';
  const chart = (x0, title, labels, vals, thrIdx, thrLabel, active) => {
    const y0 = 50, h = 180, bw = 56, gap = 26;
    txt(x, title, x0, 24, active ? C.ink : C.mute, 'left', '15px Iowan Old Style, Georgia, serif');
    x.strokeStyle = C.line; x.beginPath(); x.moveTo(x0, y0 + h); x.lineTo(x0 + labels.length * (bw + gap), y0 + h); x.stroke();
    vals.forEach((v, i) => {
      const bh = v * h, px = x0 + 8 + i * (bw + gap);
      x.fillStyle = active ? (i === vals.length - 1 ? C.amber : i === vals.length - 2 ? C.teal : '#3a4250') : '#262d38';
      x.fillRect(px, y0 + h - bh, bw, bh);
      txt(x, labels[i], px + bw / 2, y0 + h + 16, C.mute, 'center', '11px ui-monospace, monospace');
    });
    if (thrIdx !== null && active) {
      const ty = y0 + h - (vals[thrIdx] + 0.06) * h, x1 = x0 + 8 + thrIdx * (bw + gap), x2 = x0 + labels.length * (bw + gap);
      x.setLineDash([5, 4]); x.strokeStyle = C.rose; x.beginPath(); x.moveTo(x1, ty); x.lineTo(x2, ty); x.stroke(); x.setLineDash([]);
      txt(x, thrLabel, x2 + 4, ty, C.rose, 'left', 'italic 13px Iowan Old Style, Georgia, serif');
    }
  };
  chart(40, 'concept recovery R', ['logit', 'J_res', 'J_state', 'joint'], [0.41, 0.58, 0.66, 0.71], 1, '+ε', showW && !none);
  chart(470, 'steering effect |Δ log p|', ['J-dir', 'S-dir', 'Π⊥J S'], [0.8, 0.7, 0.36], null, '', showS && !none);
  if (showS && !none) {
    const ty = 50 + 180 - 0.22 * 180; x.setLineDash([5, 4]); x.strokeStyle = C.rose; x.beginPath(); x.moveTo(478, ty); x.lineTo(720, ty); x.stroke(); x.setLineDash([]);
    txt(x, 'δ', 726, ty, C.rose, 'left', 'italic 14px Iowan Old Style, Georgia, serif');
  }
  if (none) txt(x, 'no test can fail ⇒ nothing is claimed', W / 2, 140, C.rose, 'center', 'italic 18px Iowan Old Style, Georgia, serif');
  txt(x, 'schematic values · not data', W - 16, H - 10, C.mute, 'right', '11px ui-monospace, monospace');
};

/* ───────────────────────── state & persistence ───────────────────────── */
let state = { answers: {}, definition: '', updated: null };
let cur = 0;
const SUMMARY = CHOICES.length;

function ans(id) { return (state.answers[id] ||= { choice: null, synthesis: '', commentary: '' }); }
function setStatus(s) { document.getElementById('status').textContent = s; }

async function load() {
  try {
    const r = await fetch('/api/answers', { cache: 'no-store' });
    if (r.ok) { const j = await r.json(); if (j && j.answers) state = j; }
  } catch { /* offline */ }
  if (!Object.keys(state.answers).length) {
    const ls = localStorage.getItem('sspace-answers');
    if (ls) try { state = JSON.parse(ls); } catch { }
  }
  state.answers ||= {}; state.definition ||= '';
}

let saveT;
function save() {
  state.updated = new Date().toISOString();
  localStorage.setItem('sspace-answers', JSON.stringify(state));
  setStatus('saving…');
  clearTimeout(saveT);
  saveT = setTimeout(async () => {
    try {
      const r = await fetch('/api/answers', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...state, markdown: toMarkdown() }) });
      setStatus(r.ok ? `saved · explorer/answers.json · ${new Date().toLocaleTimeString()}` : 'server error · saved in browser');
    } catch { setStatus('server unreachable · saved in browser'); }
  }, 450);
  updateNav();
}

function toMarkdown() {
  const lines = ['# S-space · design decisions', '', `_Updated ${new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC_`, ''];
  CHOICES.forEach((c, i) => {
    const a = state.answers[c.id] || {};
    const o = c.options.find(o => o.k === a.choice);
    lines.push(`## ${i + 1}. ${c.title} — ${c.q}`, '');
    lines.push(`**Choice:** ${o ? `${o.label}${o.rec ? ' _(recommended)_' : ''}` : '_undecided_'}`, '');
    if (o) lines.push('```tex', o.tex, '```', '');
    if (a.synthesis?.trim()) lines.push('**Synthesis:**', '', a.synthesis.trim(), '');
    if (a.commentary?.trim()) lines.push('**Commentary:**', '', a.commentary.trim(), '');
  });
  lines.push('## Working definition', '', state.definition?.trim() || '_not yet written_', '');
  return lines.join('\n');
}

/* ───────────────────────── rendering ───────────────────────── */
function math(el) {
  if (window.renderMathInElement) renderMathInElement(el, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false });
}
function tex(el, s) { try { katex.render(s, el, { displayMode: true, throwOnError: false }); } catch { el.textContent = s; } }

function progress(id) { const a = state.answers[id]; if (!a) return 0; return a.choice ? 2 : (a.synthesis || a.commentary) ? 1 : 0; }
function updateNav() {
  const nav = document.getElementById('nav');
  nav.innerHTML = CHOICES.map((c, i) => {
    const p = progress(c.id);
    return `<button data-i="${i}" class="${i === cur ? 'cur' : ''}"><span class="num">${String(i + 1).padStart(2, '0')}</span><span>${esc(c.title)}</span><span class="dot ${p === 2 ? 'done' : p === 1 ? 'part' : ''}"></span></button>`;
  }).join('') + `<div class="sep"></div><button data-i="${SUMMARY}" class="${cur === SUMMARY ? 'cur' : ''}"><span class="num">∑</span><span>Synthesis &amp; definition</span><span></span></button>`;
  nav.querySelectorAll('button').forEach(b => b.onclick = () => go(+b.dataset.i));
  const done = CHOICES.filter(c => progress(c.id) === 2).length;
  document.getElementById('pbar').style.width = `${(done / CHOICES.length) * 100}%`;
  document.getElementById('ptext').textContent = `${done} of ${CHOICES.length} decided`;
}

function go(i) { cur = Math.max(0, Math.min(SUMMARY, i)); location.hash = cur === SUMMARY ? 'summary' : CHOICES[cur].id; render(); window.scrollTo({ top: 0 }); }

function renderViz(c) {
  stopAnim();
  const host = document.getElementById('vizhost');
  host.innerHTML = '';
  VIZ[c.viz](host, ans(c.id).choice || c.options.find(o => o.rec)?.k);
  const capL = document.getElementById('capL');
  const o = c.options.find(o => o.k === ans(c.id).choice);
  capL.textContent = o ? `showing: ${o.label}` : 'showing: recommended (nothing selected yet)';
}

function render() {
  updateNav();
  const main = document.getElementById('main');
  if (cur === SUMMARY) return renderSummary(main);
  const c = CHOICES[cur], a = ans(c.id);
  main.innerHTML = `
    <div class="eyebrow">Decision ${cur + 1} / ${CHOICES.length} · ${esc(c.title)}</div>
    <h1>${esc(c.q)}</h1>
    <p class="context" id="ctx">${c.context}</p>
    <div class="viz"><div id="vizhost"></div><div class="viz-cap"><span id="capL"></span><span>schematic</span></div></div>
    <h2>Options</h2>
    <div class="opts" role="radiogroup">
      ${c.options.map((o, i) => `<button class="opt ${a.choice === o.k ? 'sel' : ''}" role="radio" aria-checked="${a.choice === o.k}" data-k="${o.k}">
        <div class="opt-head"><span class="radio"></span><span class="opt-label">${String.fromCharCode(65 + i)}. ${esc(o.label)}</span>${o.rec ? '<span class="rec">recommended</span>' : ''}</div>
        <div class="opt-tex" data-tex="${i}"></div>
        <div class="opt-note">${esc(o.note)}</div>
      </button>`).join('')}
    </div>
    <div class="fields">
      <div class="field">
        <label for="syn">Synthesis <span class="hint">— combine options, or write your own; $…$ renders</span></label>
        <textarea id="syn" placeholder="e.g. raw state, but report the readout alongside…">${esc(a.synthesis)}</textarea>
        <div class="preview" id="synp"></div>
      </div>
      <div class="field">
        <label for="com">Commentary <span class="hint">— doubts, reasons, experiments to run</span></label>
        <textarea id="com" placeholder="why, what would change your mind…">${esc(a.commentary)}</textarea>
        <div class="preview" id="comp"></div>
      </div>
    </div>
    <div class="navrow">
      <button class="btn" id="prev" ${cur === 0 ? 'disabled' : ''}>← Previous</button>
      <button class="btn primary" id="next">${cur === CHOICES.length - 1 ? 'Synthesis →' : 'Next →'}<span class="kbd"></span></button>
    </div>`;
  math(document.getElementById('ctx'));
  main.querySelectorAll('.opt-tex').forEach(el => tex(el, c.options[+el.dataset.tex].tex));
  main.querySelectorAll('.opt').forEach(b => b.onclick = () => {
    a.choice = a.choice === b.dataset.k ? null : b.dataset.k;
    main.querySelectorAll('.opt').forEach(o => { const s = o.dataset.k === a.choice; o.classList.toggle('sel', s); o.setAttribute('aria-checked', s); });
    renderViz(c); save();
  });
  const bind = (id, key, pid) => {
    const el = document.getElementById(id), pv = document.getElementById(pid);
    const upd = () => { pv.textContent = /\$/.test(el.value) ? el.value : ''; math(pv); };
    el.oninput = () => { a[key] = el.value; upd(); save(); };
    upd();
  };
  bind('syn', 'synthesis', 'synp'); bind('com', 'commentary', 'comp');
  document.getElementById('prev').onclick = () => go(cur - 1);
  document.getElementById('next').onclick = () => go(cur + 1);
  renderViz(c);
}

function renderSummary(main) {
  stopAnim();
  const draft = R`\mathcal S_{\ell,h,b} = \operatorname{span}\big\{u_i : \sigma_i(\text{balanced } J_{\text{state}}) > \epsilon,\ \tau_h \in I_b\big\},\quad u_i \text{ persistent} \wedge \text{causal}`;
  main.innerHTML = `
    <div class="eyebrow">Synthesis</div>
    <h1>Your S-space, as decided</h1>
    <p class="lede">Click any row to revisit it. Everything autosaves to <code>explorer/answers.json</code> and <code>explorer/answers.md</code>.</p>
    <div class="sum">
      ${CHOICES.map((c, i) => {
        const a = state.answers[c.id] || {}, o = c.options.find(o => o.k === a.choice);
        return `<div class="sum-row" data-i="${i}"><div class="sum-n">${String(i + 1).padStart(2, '0')}</div><div class="sum-t">${esc(c.title)}</div>
          <div class="sum-c">${o ? `<span class="pick">${esc(o.label)}</span>` : '<span class="none">undecided</span>'}
          ${a.synthesis?.trim() ? `<div class="syn">${esc(a.synthesis.trim())}</div>` : ''}</div></div>`;
      }).join('')}
    </div>
    <h2>Draft definition from the recommended picks</h2>
    <div class="def-tex" id="draft"></div>
    <h2>Your working definition</h2>
    <div class="field"><textarea id="def" style="min-height:180px" placeholder="S-space is …  ($…$ renders)">${esc(state.definition)}</textarea><div class="preview" id="defp"></div></div>
    <div class="actions">
      <button class="btn primary" id="dl">Download answers.md</button>
      <button class="btn" id="cp">Copy markdown</button>
      <button class="btn" id="back">← Back to decision 10</button>
    </div>`;
  tex(document.getElementById('draft'), draft);
  main.querySelectorAll('.sum-row').forEach(r => r.onclick = () => go(+r.dataset.i));
  const el = document.getElementById('def'), pv = document.getElementById('defp');
  const upd = () => { pv.textContent = /\$/.test(el.value) ? el.value : ''; math(pv); };
  el.oninput = () => { state.definition = el.value; upd(); save(); }; upd();
  document.getElementById('dl').onclick = () => {
    const b = new Blob([toMarkdown()], { type: 'text/markdown' });
    const u = URL.createObjectURL(b), aEl = document.createElement('a'); aEl.href = u; aEl.download = 'answers.md'; aEl.click(); URL.revokeObjectURL(u);
  };
  document.getElementById('cp').onclick = async () => { await navigator.clipboard.writeText(toMarkdown()); setStatus('markdown copied'); };
  document.getElementById('back').onclick = () => go(SUMMARY - 1);
}

document.addEventListener('keydown', e => {
  if (/TEXTAREA|INPUT/.test(document.activeElement?.tagName)) return;
  if (e.key === 'ArrowRight' || e.key === 'j') go(cur + 1);
  if (e.key === 'ArrowLeft' || e.key === 'k') go(cur - 1);
  if (cur < SUMMARY && /^[a-f]$/i.test(e.key)) {
    const b = document.querySelectorAll('.opt')[e.key.toLowerCase().charCodeAt(0) - 97]; if (b) b.click();
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  await load();
  const h = location.hash.slice(1);
  const idx = CHOICES.findIndex(c => c.id === h);
  cur = h === 'summary' ? SUMMARY : idx >= 0 ? idx : 0;
  render();
  setStatus('ready · autosave on');
});
