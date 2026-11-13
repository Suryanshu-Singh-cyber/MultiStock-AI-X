import React, { useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, Cell, ReferenceLine,
} from 'recharts'

const pct = (x, d = 1) => (x == null || !isFinite(x) ? '—' : `${(x * 100).toFixed(d)}%`)
const sgn = (x, d = 2) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`)

const tooltipStyle = { background: '#0d0d1f', border: '1px solid #2a2a45', borderRadius: 12, fontSize: 12, color: '#eef0ff' }

function verdictOf(d) {
  if (!d || !d.forecast) return { cls: 'warn', title: 'INITIALIZING', plain: 'Fetching the latest forecast…' }
  const f = d.forecast, s = d.sizing || {}
  const up = (f.p_up || 0.5) - 0.5
  if (d.status === 'DEGRADED' || d.status === 'ERROR')
    return { cls: 'warn', title: 'SYSTEM PAUSED', plain: 'Data looks stale or model health is poor — the system refuses to guess. Check back after market close.' }
  if (s.no_trade)
    return { cls: 'nt', title: 'NO TRADE', plain: `Best estimate is ${sgn(f.mu_log_return)}, but trading costs ~${pct(s.round_trip_cost)} — bigger than the expected edge. Sitting out is the disciplined move.` }
  if (up > 0.08)
    return { cls: 'long', title: 'SMALL LONG BIASED', plain: `Model leans UP (${pct(f.p_up)}). Edge beats costs — keep size small; this is a probability, not a promise.` }
  if (up < -0.08)
    return { cls: 'short', title: 'STAY AWAY', plain: `Model leans DOWN (only ${pct(f.p_up)} chance of rising). Do not initiate longs.` }
  return { cls: 'nt', title: 'NO TRADE', plain: `Signal is noise-level (${pct(f.p_up)} up-chance). Costs would eat any edge.` }
}

function Panel({ title, sub, children, delay = 0 }) {
  return (
    <motion.div className="cp" initial={{ opacity: 0, y: 22 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-30px' }} transition={{ duration: 0.5, delay }}>
      <div className="cp-head"><h3>{title}</h3>{sub && <span className="cp-sub">{sub}</span>}</div>
      {children}
    </motion.div>
  )
}
const Row = ({ k, v, cls }) => <div className="cp-row"><span>{k}</span><b className={cls}>{v}</b></div>

export default function CommandCenter({ d, prices, score, equity, tape, live, refresh, doRefresh }) {
  const v = useMemo(() => verdictOf(d), [d])
  const f = d?.forecast || {}
  const risk = d?.risk || {}
  const mc = d?.monte_carlo_21d || {}
  const sizing = d?.sizing || {}

  const priceData = useMemo(() => {
    const arr = (prices || []).map(p => ({ date: p.date.slice(5), close: p.close }))
    // 20-day simple moving average
    return arr.map((r, i) => ({
      ...r,
      ma20: i >= 19 ? +(arr.slice(i - 19, i + 1).reduce((a, b) => a + b.close, 0) / 20).toFixed(2) : null,
    }))
  }, [prices])

  const drivers = useMemo(() => {
    const e = Object.entries(d?.drivers || {})
    const mx = Math.max(...e.map(([, x]) => Math.abs(x)), 1e-9)
    return e.slice(0, 10).map(([k, x]) => ({
      k: k.replace(/^tgt_/, '').replace(/_/g, ' '), x, w: Math.abs(x) / mx * 50,
    }))
  }, [d])

  const hist = mc.final_hist
  const histData = hist ? hist.counts.map((c, i) => ({
    x: hist.edges[i], h: c,
  })) : []

  const eqData = equity?.dates ? equity.dates.map((dt, i) => ({
    date: dt.slice(2),
    strategy: +((equity.strategy[i] - 1) * 100).toFixed(2),
    buyhold: +((equity.buy_hold[i] - 1) * 100).toFixed(2),
  })) : []

  const models = Object.entries(f.model_preds || {})
  const spread = f.disagreement ?? 0

  return (
    <section id="command" className="command-v2">
      {/* A. command header */}
      <div className="cmd-head">
        <div>
          <span className="sec-kicker">LIVE · POWERGRID.NS · NSE INDIA</span>
          <h2>Command Center</h2>
          <div className="cmd-meta">
            <span>Updated {d?.generated_at?.slice(0, 19).replace('T', ' ') || '—'}</span>
            <span className={`stale ${d && d.data_staleness_days > 2 ? 'bad' : ''}`}>
              data through {d?.latest_data_date || '—'}
              {d && d.data_staleness_days > 2 && ' ⚠ STALE'}
            </span>
            {live?.type === 'price_tick' && <span className="tick-live">● tick {live.symbol} ₹{Number(live.price).toFixed(2)}</span>}
          </div>
        </div>
        <div className="cmd-actions">
          {refresh.msg && <span className="refresh-msg">{refresh.msg}</span>}
          <button className="refresh-btn" onClick={doRefresh} disabled={refresh.running}>
            {refresh.running ? 'Updating…' : 'Update Now'}
          </button>
        </div>
      </div>
      {tape.length > 0 && <div className="ticker"><div className="ticker-inner">{tape.join('  ·  ')}&nbsp;&nbsp;{tape.join('  ·  ')}</div></div>}

      {/* B. verdict */}
      <motion.div className={`verdict-panel ${v.cls}`}
        initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}>
        <div className="vp-left">
          <div className="vp-kicker">TOMORROW'S CALL · {d?.forecast_date || ''}</div>
          <div className="vp-title">{v.title}</div>
          <p className="vp-plain">{v.plain}</p>
        </div>
        <div className="vp-right">
          <div className="vp-chip"><span>Expected</span><b className={f.mu_log_return >= 0 ? 'up' : 'down'}>{sgn(f.mu_log_return)}</b></div>
          <div className="vp-chip"><span>P(up)</span><b>{pct(f.p_up)}</b></div>
          <div className="vp-chip"><span>σ daily</span><b>{f.sigma != null ? `${(f.sigma * 100).toFixed(2)}%` : '—'}</b></div>
          <div className="vp-chip"><span>Uncertainty</span><b>{f.disagreement < 0.003 ? 'low' : 'elevated'}</b></div>
        </div>
      </motion.div>

      <div className="cp-grid">
        {/* C. price history */}
        <Panel title="Price — last 90 days" sub="close + 20-day average" delay={0.02}>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={priceData} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                <defs>
                  <linearGradient id="pg2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148,163,184,.07)" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: '#7c86ad', fontSize: 10 }} interval={9} />
                <YAxis domain={['auto', 'auto']} tick={{ fill: '#7c86ad', fontSize: 10 }} width={52} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="close" stroke="#60a5fa" strokeWidth={2} fill="url(#pg2)" dot={false} name="Close" />
                <Line type="monotone" dataKey="ma20" stroke="#a78bfa" strokeWidth={1.4} dot={false} name="20-day avg" strokeDasharray="5 4" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        {/* E. gauge */}
        <Panel title="Chance it rises tomorrow" sub="probability, not certainty" delay={0.05}>
          <div className="gauge-big">
            <Gauge value={f.p_up} />
            <div className="gb-read">
              <div className={`gb-num ${(f.p_up || 0) >= 0.5 ? 'up' : 'down'}`}>{pct(f.p_up)}</div>
              <div className="gb-note">
                {(f.p_up || 0) >= 0.56 ? 'Leans bullish — but still a probability.' :
                 (f.p_up || 0) <= 0.44 ? 'Leans bearish — treat rallies with suspicion.' :
                 'Essentially a coin flip — the honest answer.'}
              </div>
            </div>
          </div>
        </Panel>

        {/* F. range */}
        <Panel title="Prediction range" sub="P10 → P50 → P90 (80% band)" delay={0.08}>
          {(() => {
            const p10 = f.p10 ?? -0.02, p90 = f.p90 ?? 0.02
            const lo = Math.min(p10, 0), hi = Math.max(p90, 0), span = hi - lo || 1e-6
            const pos = ((0 - lo) / span) * 100, w = ((hi - lo) / span) * 100
            const mid = (((f.p50 ?? 0) - lo) / span) * 100
            return (
              <div className="range-v2">
                <div className="rv-track">
                  <div className="rv-band" style={{ left: `${pos}%`, width: `${w}%` }} />
                  <div className="rv-needle" style={{ left: `${mid}%` }} />
                </div>
                <div className="rv-ends">
                  <span>Bad case<br /><b className="down">{pct(p10)}</b></span>
                  <span>Best guess<br /><b>{sgn(f.p50)}</b></span>
                  <span>Good case<br /><b className="up">{pct(p90)}</b></span>
                </div>
                <p className="rv-note">The wide band is the honest part — tomorrow is genuinely uncertain.</p>
              </div>
            )
          })()}
        </Panel>

        {/* G. horizons */}
        <Panel title="Beyond tomorrow" sub="separate models per horizon" delay={0.11}>
          <div className="horizon-v2">
            {[['Tomorrow (1d)', f.mu_log_return], ['5 days', f.horizons?.['5d']], ['21 days', f.horizons?.['21d']]].map(([k, val], i) => (
              <div key={k} className="hz">
                <span className="hz-k">{k}</span>
                <span className={`hz-v ${(val || 0) > 0.003 ? 'up' : (val || 0) < -0.003 ? 'down' : 'flat'}`}>
                  {val != null ? sgn(val) : '—'}
                </span>
                <div className="hz-bar"><i style={{ width: `${Math.min(100, Math.abs(val || 0) * 4000)}%` }} /></div>
              </div>
            ))}
            <p className="rv-note">Cumulative expected return if held that long. Longer horizons historically forecast better (59% direction at 21d in research).</p>
          </div>
        </Panel>
      </div>

      {/* H. risk */}
      <Panel title="Risk intelligence" sub="what a bad day could look like" delay={0.02}>
        <div className="risk-grid">
          {[
            ['VaR 95%', pct(risk.var95_gauss), '1-in-20 bad day loss'],
            ['ES 95%', pct(risk.es95_gauss), 'when bad, usually ≈ this'],
            ['VaR 99%', pct(risk.var99_gauss), '1-in-100 bad day'],
            ['ES 99%', pct(risk.es99_gauss), 'tail-risk average'],
            ['Vol (annual)', pct(risk.vol_annual), 'yearly swing size'],
            ['Max drawdown', pct(risk.max_drawdown_window), 'worst peak→trough'],
            ['GARCH vol tmrw', risk.vol_forecast ? pct(risk.vol_forecast.vol_next_annual) : '—', 'model-based forecast'],
            ['Persistence', risk.vol_forecast ? risk.vol_forecast.persistence?.toFixed(3) : '—', 'vol clustering strength'],
          ].map(([k, v, s]) => (
            <div key={k} className="risk-cell">
              <span className="rk-k">{k}</span>
              <b className="rk-v">{v}</b>
              <small>{s}</small>
            </div>
          ))}
        </div>
      </Panel>

      {/* I. drivers */}
      <Panel title="Why the AI thinks this" sub="SHAP contributions — what pushed the guess" delay={0.02}>
        <div className="drivers-v2">
          {drivers.length === 0 && <p className="muted">Drivers appear after the next forecast run.</p>}
          {drivers.map(r => (
            <div key={r.k} className="drv-row">
              <span className="drv-k" title={r.k}>{r.k}</span>
              <div className="drv-track">
                <div className="drv-center" />
                <div className={`drv-fill ${r.x >= 0 ? 'pos' : 'neg'}`}
                  style={{ width: `${r.w}%`, [r.x >= 0 ? 'left' : 'right']: '50%' }} />
              </div>
              <b className={`drv-v ${r.x >= 0 ? 'up' : 'down'}`}>{r.x >= 0 ? '+' : ''}{(r.x * 100).toFixed(2)}%</b>
            </div>
          ))}
        </div>
      </Panel>

      {/* J. Monte Carlo */}
      <Panel title="10,000 simulated futures" sub={`Student-t paths, ${mc.horizon_days || 21}-day horizon — real simulation`} delay={0.02}>
        <div className="mc-grid">
          <div className="mc-stats">
            <Row k="Bad outcome (5%)" v={pct(mc.final_ret_p05)} cls="down" />
            <Row k="Median (50%)" v={pct(mc.final_ret_p50)} cls="" />
            <Row k="Good outcome (95%)" v={pct(mc.final_ret_p95)} cls="up" />
            <Row k="Chance of losing" v={pct(mc.prob_loss)} cls="" />
            <Row k="Typical dip" v={pct(mc.maxdd_p50)} cls="down" />
            <Row k="Bad-luck dip" v={pct(mc.maxdd_p95)} cls="down" />
          </div>
          {histData.length > 0 && (
            <div className="mc-hist">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={histData} margin={{ top: 4, right: 4, left: -26, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(148,163,184,.07)" vertical={false} />
                  <XAxis dataKey="x" tick={{ fill: '#7c86ad', fontSize: 9 }} interval={4}
                    tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <YAxis tick={{ fill: '#7c86ad', fontSize: 9 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <ReferenceLine x={0} stroke="#f87171" strokeDasharray="3 3" />
                  <Bar dataKey="h" radius={[3, 3, 0, 0]}>
                    {histData.map((r, i) => <Cell key={i} fill={r.x < 0 ? '#fb7185' : '#34d399'} opacity={0.75} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <small className="muted">Distribution of 21-day outcomes across 10,000 fat-tailed paths.</small>
            </div>
          )}
        </div>
      </Panel>

      {/* K. voting room */}
      <Panel title="Model voting room" sub="each model guesses independently — equal weight average" delay={0.02}>
        <div className="vote-grid">
          {models.map(([name, val]) => {
            const dir = val > 0.001 ? 'up' : val < -0.001 ? 'down' : 'flat'
            return (
              <div key={name} className="vote-card">
                <span className="vote-name">{name.toUpperCase()}</span>
                <b className={`vote-val ${dir}`}>{sgn(val)}</b>
                <span className={`vote-dir ${dir}`}>{dir === 'up' ? '▲ bullish' : dir === 'down' ? '▼ bearish' : '◆ neutral'}</span>
              </div>
            )
          })}
          <div className="vote-card ensemble">
            <span className="vote-name">ENSEMBLE μ</span>
            <b className={f.mu_log_return >= 0 ? 'up' : 'down'}>{sgn(f.mu_log_return)}</b>
            <span className="vote-dir">spread {(spread * 100).toFixed(2)}% {spread < 0.003 ? '· agree ✓' : '· nervous ⚠'}</span>
          </div>
        </div>
        <p className="rv-note">Higher disagreement widens the uncertainty band and makes the no-trade filter stricter.</p>
      </Panel>

      {/* L. should I buy */}
      <Panel title="Position decision" sub="from the risk engine — never overridden by the UI" delay={0.02}>
        {sizing.no_trade ? (
          <div className="decision nt">
            <div className="dec-big">₹0 — DO NOTHING</div>
            <p>{sizing.reasons?.[0] || 'Expected edge does not exceed transaction costs.'}</p>
            <p className="muted">No-trade is a valid outcome. Most professional desks sit out more often than they trade.</p>
          </div>
        ) : (
          <div className="decision go">
            <div className="dec-big">{sizing.kelly_pct?.toFixed(1)}% of capital</div>
            <p>¼-Kelly allocation — the cautious fraction of the theoretically optimal bet.</p>
          </div>
        )}
      </Panel>
    </section>
  )
}

/* gauge reused */
function Gauge({ value }) {
  const pctv = Math.max(0, Math.min(100, (value || 0) * 100))
  const ang = Math.PI * (1 - pctv / 100)
  const x = 60 + 46 * Math.cos(ang), y = 58 - 46 * Math.sin(ang)
  const large = pctv > 50 ? 1 : 0
  return (
    <svg width="180" height="105" viewBox="0 0 120 66">
      <path d="M10 58 A50 50 0 0 1 110 58" fill="none" stroke="rgba(148,163,184,.15)" strokeWidth="11" strokeLinecap="round" />
      <motion.path d={`M10 58 A50 50 0 ${large} 1 ${x} ${y}`} fill="none"
        stroke={pctv >= 50 ? '#34d399' : '#fb7185'} strokeWidth="11" strokeLinecap="round"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }}
        transition={{ duration: 1.2, ease: 'easeOut' }}
        style={{ filter: `drop-shadow(0 0 8px ${pctv >= 50 ? 'rgba(52,211,153,.7)' : 'rgba(251,113,133,.7)'})` }} />
      <text x="60" y="56" textAnchor="middle" fontSize="22" fontWeight="800" fill="#f4f1ea">{pctv.toFixed(0)}%</text>
    </svg>
  )
}
