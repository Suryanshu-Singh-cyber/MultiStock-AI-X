import React, { Component, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ToyScene from './ToyScene'

const pct = (x, d = 1) => (x == null || !isFinite(x) ? '—' : `${(x * 100).toFixed(d)}%`)
const sgn = (x, d = 2) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`)

/* ---------- WebGL fallback ---------- */
class GlBoundary extends Component {
  state = { err: false }
  static getDerivedStateFromError() { return { err: true } }
  render() {
    if (this.state.err) return (
      <div className="gl-fallback">
        <div className="gf-tower">
          <div className="gf-b" style={{ background: '#2563eb' }} />
          <div className="gf-b" style={{ background: '#eab308' }} />
          <div className="gf-b" style={{ background: '#10b981' }} />
        </div>
        <p>This device can't run the 3D view — the forecast data below is still fully live.</p>
      </div>)
    return this.props.children
  }
}

/* ---------- helpers ---------- */
const PART_META = {
  market: { name: 'Market Signals', color: '#60a5fa',
    what: 'How the whole Indian stock market (NIFTY 50) is likely to move tomorrow.',
    why: 'POWERGRID doesn\'t trade in a vacuum — when the ocean rises, every boat floats.' },
  sector: { name: 'Sector Signals', color: '#facc15',
    what: 'How the power-sector team (NTPC, Tata Power, Adani Power…) is likely to move.',
    why: 'Stocks in the same industry move together — this block is the team\'s huddle.' },
  alpha: { name: 'Alpha Signals', color: '#34d399',
    what: 'What is unique to POWERGRID itself — its own momentum, volume, news and patterns.',
    why: 'This is the AI\'s real edge: the part no index or sector can give you.' },
}

function friendlyClue(k) {
  const s = k.toLowerCase()
  if (s.includes('lag1') || s.includes('ret_lag')) return 'Yesterday\'s move'
  if (s.includes('macd')) return 'Trend speedometer'
  if (s.includes('vol')) return 'Wiggle size'
  if (s.includes('corr_sector')) return 'Sector mood'
  if (s.includes('vs_peers') || s.includes('peer')) return 'vs rival stocks'
  if (s.includes('mom')) return 'Momentum'
  if (s.includes('52w') || s.includes('dist')) return 'Distance from top'
  if (s.includes('news') || s.includes('sent')) return 'News mood'
  if (s.includes('vix')) return 'Fear gauge'
  if (s.includes('crude') || s.includes('oil') || s.includes('cl_f')) return 'Oil prices'
  if (s.includes('usdinr') || s.includes('usd')) return 'Dollar rate'
  if (s.includes('volume') || s.includes('volu')) return 'Trading crowd'
  return k.replace(/^tgt_/, '').replace(/_/g, ' ')
}

function weatherOf(d) {
  const v = d?.risk?.vol_forecast?.vol_next_annual ?? d?.forecast?.sigma
  const sigma = d?.forecast?.sigma ?? 0.02
  const score = (v ?? sigma * Math.sqrt(252))
  const pUp = d?.forecast?.p_up ?? 0.5
  // blend vol + direction into a 0..1 "weather" score
  const w = Math.max(0, Math.min(1, 0.5 + (pUp - 0.5) * 2 - (score - 0.2) * 1.5))
  if (w > 0.62) return { cls: 'sun', icon: 'sun', label: 'Sunny — signals lean favorable' }
  if (w > 0.45) return { cls: 'cloud', icon: 'cloud', label: 'Cloudy — mixed signals' }
  return { cls: 'storm', icon: 'storm', label: 'Stormy — uncertain / unfavorable' }
}
const WX_SVG = {
  sun: <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="5" fill="#facc15"/><g stroke="#facc15" strokeWidth="2" strokeLinecap="round">{[0,45,90,135,180,225,270,315].map(a=><line key={a} x1={12+8*Math.cos(a*Math.PI/180)} y1={12+8*Math.sin(a*Math.PI/180)} x2={12+11*Math.cos(a*Math.PI/180)} y2={12+11*Math.sin(a*Math.PI/180)}/>)}</g></svg>,
  cloud: <svg viewBox="0 0 24 24"><path d="M7 18a4.5 4.5 0 0 1-.4-9A6 6 0 0 1 18 8.5 4 4 0 0 1 17.5 18Z" fill="#94a3b8"/></svg>,
  storm: <svg viewBox="0 0 24 24"><path d="M7 15a4.5 4.5 0 0 1-.4-9A6 6 0 0 1 18 5.5 4 4 0 0 1 17.5 15Z" fill="#64748b"/><path d="M12 15 L9 20 h3 L10 24 l5-6 h-3 l3-3 Z" fill="#facc15"/></svg>,
}

export default function ToyBox({ d, score }) {
  const [exploded, setExploded] = useState(false)
  const [sel, setSel] = useState(null)
  const [pro, setPro] = useState(false)
  const f = d?.forecast || {}
  const decomp = d?.decomposition || {}
  const drivers = useMemo(() => Object.entries(d?.drivers || {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3), [d])
  const w = weatherOf(d)
  const meta = sel ? PART_META[sel] : null
  const total = Math.abs(decomp.market || 0) + Math.abs(decomp.sector || 0) + Math.abs(decomp.alpha || 0) || 1e-9
  const share = k => Math.abs(decomp[k] || 0) / total
  const blockClues = useMemo(() => {
    if (!sel) return []
    const map = { market: ['vix', 'nsei', 'market'], sector: ['sector', 'peer', 'cnx'], alpha: ['tgt_', 'mom', 'vol'] }
    return drivers.filter(([k]) => map[sel].some(t => k.toLowerCase().includes(t)))
  }, [sel, drivers])

  return (
    <section id="toybox" className="toybox-v2">
      <div className="sec-head">
        <span className="sec-kicker">INTERACTIVE 3D</span>
        <h2>The forecast, exploded</h2>
        <p>Tomorrow's guess is a tower of three signal blocks. Pull it apart and see what's inside.</p>
      </div>

      <div className="toy-grid">
        <div className="toy-stage">
          <GlBoundary>
            <ToyScene sigma={f.sigma} pUp={f.p_up} noTrade={d?.sizing?.no_trade}
              exploded={exploded} selected={sel}
              onPick={k => { setSel(k); setExploded(true) }} />
          </GlBoundary>
          <div className="toy-controls">
            <button className="pull-btn" onClick={() => { setExploded(e => !e); if (!exploded) setSel(null) }}>
              {exploded ? '⇓ Snap together' : '⇑ Pull it apart'}
            </button>
            <div className="toy-legend">
              {Object.entries(PART_META).map(([k, m]) => (
                <button key={k} className={`legend ${sel === k ? 'on' : ''}`}
                  style={{ ['--c']: m.color }}
                  onClick={() => { setSel(k); setExploded(true) }}>
                  <i />{m.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="toy-side">
          <AnimatePresence mode="wait">
            {meta ? (
              <motion.div key={sel} className="toy-card"
                initial={{ opacity: 0, y: 24, rotate: -1 }}
                animate={{ opacity: 1, y: 0, rotate: 0 }}
                exit={{ opacity: 0, y: -16 }} transition={{ duration: 0.35 }}>
                <div className="toy-card-head" style={{ borderColor: meta.color }}>
                  <h3 style={{ color: meta.color }}>{meta.name}</h3>
                  <span className="toy-share">{(share(sel) * 100).toFixed(0)}% of guess</span>
                </div>
                <p><b>What it is:</b> {meta.what}</p>
                <p><b>Why it matters:</b> {meta.why}</p>
                <div className="toy-share-bar"><div style={{ width: `${share(sel) * 100}%`, background: meta.color }} /></div>
                {blockClues.length > 0 && (
                  <div className="toy-clues">
                    <span className="tc-label">Top clues inside:</span>
                    {blockClues.map(([k, x]) => (
                      <span key={k} className={`tc-chip ${x >= 0 ? 'pos' : 'neg'}`}>
                        {friendlyClue(k)} {x >= 0 ? '▲' : '▼'}
                      </span>
                    ))}
                  </div>
                )}
                {pro && <div className="pro-line">contribution ≈ {sgn(decomp[sel] || 0, 3)} log-return</div>}
              </motion.div>
            ) : (
              <motion.div key="hint" className="toy-card hint"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h3>Pick a block</h3>
                <p>Tap any layer of the tower — or press "Pull it apart" — to learn what drives that part of the forecast.</p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="toy-meta-row">
            <div className={`wx-badge ${w.cls}`} title="Market-weather metaphor — based on forecast volatility & direction, not actual weather">
              <span className="wx-icon">{WX_SVG[w.icon]}</span>
              <div><b>{w.label.split('—')[0].trim()}</b><small>{w.label.split('—')[1]?.trim() || ''}</small></div>
            </div>
            {d?.sizing?.no_trade ? (
              <div className="wx-badge shield">
                <span className="wx-icon"><svg viewBox="0 0 24 24"><path d="M12 3 L20 6 V12 C20 17 16.5 20.5 12 22 C7.5 20.5 4 17 4 12 V6 Z" fill="#fb7185"/><path d="M9 12 L11 14 L15 10" stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round"/></svg></span>
                <div><b>Rest Day</b><small>edge &lt; costs — coins safe</small></div>
              </div>
            ) : (
              <div className="wx-badge go">
                <span className="wx-icon"><svg viewBox="0 0 24 24"><path d="M4 18 L10 12 L14 15 L20 7" stroke="#34d399" strokeWidth="2.5" fill="none" strokeLinecap="round"/><path d="M15 7 H20 V12" stroke="#34d399" strokeWidth="2.5" fill="none" strokeLinecap="round"/></svg></span>
                <div><b>Edge found</b><small>small position OK</small></div>
              </div>
            )}
          </div>

          <button className={`pro-toggle ${pro ? 'on' : ''}`} onClick={() => setPro(p => !p)}>
            <span className="pt-track"><span className="pt-thumb" /></span>
            Pro Quant Mode
          </button>

          {pro && (
            <motion.div className="pro-panel" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
              <div className="pro-grid">
                <div><span>μ log-ret</span><b>{f.mu_log_return?.toExponential(3)}</b></div>
                <div><span>σ</span><b>{f.sigma?.toFixed(5)}</b></div>
                <div><span>P(up)</span><b>{f.p_up?.toFixed(4)}</b></div>
                <div><span>disagreement</span><b>{f.disagreement?.toFixed(5)}</b></div>
                <div><span>live dir-acc</span><b>{score?.dir_acc != null ? pct(score.dir_acc) : '—'}</b></div>
                <div><span>live Brier</span><b>{score?.brier_p_up?.toFixed(3) ?? '—'}</b></div>
                <div><span>SPA p-value*</span><b>0.649</b></div>
                <div><span>LSTM dir-acc*</span><b>0.526</b></div>
              </div>
              <p className="pro-note">* research findings, 525 OOS days — nothing significantly beats naive yet. Live scorecard accumulates daily.</p>
            </motion.div>
          )}
        </div>
      </div>
    </section>
  )
}
