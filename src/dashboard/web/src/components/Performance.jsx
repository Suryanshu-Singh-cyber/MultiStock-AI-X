import React from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

const pct = (x, d = 1) => (x == null || !isFinite(x) ? '—' : `${(x * 100).toFixed(d)}%`)
const tooltipStyle = { background: '#0d0d1f', border: '1px solid #2a2a45', borderRadius: 12, fontSize: 12, color: '#eef0ff' }

/* honest research summary — static findings, clearly labeled as research not live */
const RESEARCH = [
  ['LSTM (deep)', '52.6%', '25-fold walk-forward, ~4y window'],
  ['Best baseline', '52.4%', 'Ridge / ARIMA family'],
  ['SPA test p-value', '0.649', 'NOT significant — nothing beats naive yet'],
  ['21-day horizon', '59.1%', 'longer horizons predict better'],
  ['OOS days tested', '525', 'pooled walk-forward predictions'],
]

export default function Performance({ equity, score }) {
  const eqData = equity?.dates ? equity.dates.map((dt, i) => ({
    date: dt.slice(2),
    strategy: +((equity.strategy[i] - 1) * 100).toFixed(2),
    buyhold: +((equity.buy_hold[i] - 1) * 100).toFixed(2),
  })) : []

  return (
    <section id="performance" className="perf-v2">
      <div className="sec-head">
        <span className="sec-kicker">EVIDENCE</span>
        <h2>Performance — proven, not promised</h2>
        <p>Backtest results are labeled as such. Live accuracy is earned one forecast at a time.</p>
      </div>

      <div className="perf-grid">
        <motion.div className="perf-card wide" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <div className="pc-head"><h3>Model vs buy-and-hold</h3><span className="pc-sub">net of 0.26% round-trip costs · ARIMA strategy · 525 OOS days</span></div>
          {eqData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={eqData} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(148,163,184,.07)" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: '#7c86ad', fontSize: 10 }} interval={40} />
                  <YAxis tick={{ fill: '#7c86ad', fontSize: 10 }} tickFormatter={x => `${x}%`} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="strategy" stroke="#34d399" strokeWidth={2.2} dot={false} name="Model (net)" />
                  <Line type="monotone" dataKey="buyhold" stroke="#64748b" strokeWidth={1.5} dot={false} strokeDasharray="5 4" name="Buy & hold" />
                </LineChart>
              </ResponsiveContainer>
              <div className="perf-final">
                <span>Model final: <b className={equity.final_strategy >= 0 ? 'up' : 'down'}>{pct(equity.final_strategy)}</b></span>
                <span>Buy & hold: <b>{pct(equity.final_buyhold)}</b></span>
                <span className="muted">past performance ≠ future results</span>
              </div>
            </>
          ) : <p className="muted">Loading…</p>}
        </motion.div>

        <motion.div className="perf-card" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.08 }}>
          <div className="pc-head"><h3>Live report card</h3><span className="pc-sub">real forecasts, scored daily</span></div>
          {score && score.total_scored > 0 ? (
            <div className="score-grid">
              <div className="sg"><span>Scored</span><b>{score.total_scored}</b></div>
              <div className="sg"><span>Dir-accuracy</span><b className={score.dir_acc >= 0.52 ? 'up' : score.dir_acc <= 0.48 ? 'down' : ''}>{pct(score.dir_acc)}</b></div>
              <div className="sg"><span>Last 20d</span><b>{pct(score.dir_acc_20d)}</b></div>
              <div className="sg"><span>Brier</span><b>{score.brier_p_up?.toFixed(3)}</b></div>
              <div className="sg"><span>No-trade rate</span><b>{pct(score.no_trade_rate)}</b></div>
            </div>
          ) : (
            <p className="muted" style={{ lineHeight: 1.6 }}>
              Your live report card is being built — every forecast is logged tonight and scored against tomorrow's real close. Check back in a few weeks for true out-of-sample numbers.
            </p>
          )}
        </motion.div>

        <motion.div className="perf-card" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.14 }}>
          <div className="pc-head"><h3>Research findings</h3><span className="pc-sub">honest backtest summary</span></div>
          <table className="research-table">
            <tbody>
              {RESEARCH.map(([k, v, s]) => (
                <tr key={k}><td>{k}</td><td><b>{v}</b></td><td className="muted">{s}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ fontSize: 11, marginTop: 10 }}>
            SPA (Superior Predictive Ability) test p=0.649 → we do NOT claim proven alpha yet. The system's edge today is honesty + risk control.
          </p>
        </motion.div>
      </div>
    </section>
  )
}
