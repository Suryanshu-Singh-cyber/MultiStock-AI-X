import React from 'react'
import { motion } from 'framer-motion'

/* varied mini-illustrations built from CSS/SVG — not emoji, not icon packs */
const VIS = {
  lock: <div className="vis-lock"><i/><i/><i/></div>,
  chart: <svg viewBox="0 0 60 40"><polyline points="4,32 14,26 24,29 34,18 44,22 56,10" fill="none" stroke="#34d399" strokeWidth="3" strokeLinecap="round"/></svg>,
  shield: <svg viewBox="0 0 60 40"><path d="M30 6 L50 14 V26 C50 33 42 37 30 39 C18 37 10 33 10 26 V14 Z" fill="none" stroke="#f472b6" strokeWidth="3"/><path d="M22 21 L28 27 L39 15" stroke="#f472b6" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  vote: <svg viewBox="0 0 60 40"><rect x="6" y="18" width="12" height="16" rx="3" fill="#34d399"/><rect x="24" y="10" width="12" height="24" rx="3" fill="#facc15"/><rect x="42" y="22" width="12" height="12" rx="3" fill="#3b82f6"/></svg>,
  horizon: <svg viewBox="0 0 60 40"><path d="M6 32 Q20 30 30 22 T54 12" fill="none" stroke="#a78bfa" strokeWidth="3" strokeDasharray="4 4"/><circle cx="54" cy="12" r="4" fill="#a78bfa"/></svg>,
  radar: <svg viewBox="0 0 60 40"><circle cx="30" cy="20" r="14" fill="none" stroke="#60a5fa" strokeWidth="2"/><circle cx="30" cy="20" r="8" fill="none" stroke="#60a5fa" strokeWidth="2" opacity=".5"/><line x1="30" y1="20" x2="40" y2="10" stroke="#60a5fa" strokeWidth="2.5" strokeLinecap="round"/></svg>,
  clock: <svg viewBox="0 0 60 40"><circle cx="30" cy="20" r="15" fill="none" stroke="#facc15" strokeWidth="3"/><path d="M30 11 V20 L38 24" stroke="#facc15" strokeWidth="3" fill="none" strokeLinecap="round"/></svg>,
  report: <svg viewBox="0 0 60 40"><rect x="14" y="6" width="32" height="28" rx="4" fill="none" stroke="#34d399" strokeWidth="3"/><path d="M20 26 L26 20 L32 23 L40 14" stroke="#34d399" strokeWidth="2.5" fill="none" strokeLinecap="round"/></svg>,
}

const FEATURES = [
  { vis: 'lock', title: 'Leak-proof', text: 'Every input is timestamped. A test suite proves no future data can ever sneak into a forecast.', span: 'wide' },
  { vis: 'chart', title: 'Honest stats', text: 'Walk-forward testing and real trading costs — no fake 98% accuracy claims.' },
  { vis: 'shield', title: 'Says NO', text: 'When expected gain < trading fee, it literally recommends doing nothing. That\'s the edge.', span: 'wide' },
  { vis: 'vote', title: 'Model voting', text: 'Four independent models vote; disagreement makes the system more cautious.' },
  { vis: 'horizon', title: 'Multi-horizon', text: 'Tomorrow, 5-day and 21-day forecasts. Longer horizons are more predictable.' },
  { vis: 'radar', title: 'Risk engine', text: 'VaR, expected shortfall, GARCH volatility and 10,000-path Monte Carlo.' },
  { vis: 'clock', title: 'Daily updates', text: 'Runs itself at 3:45 PM IST every trading day — forecast ready before tomorrow\'s open.' },
  { vis: 'report', title: 'Report card', text: 'Every forecast is logged and scored against reality. Trust is earned, not claimed.' },
]

export default function Features() {
  return (
    <section id="features" className="features-v2">
      <div className="sec-head">
        <span className="sec-kicker">CAPABILITIES</span>
        <h2>What can it do?</h2>
        <p>Eight guarantees, each one testable. No marketing fluff.</p>
      </div>
      <div className="feat-grid-v2">
        {FEATURES.map((f, i) => (
          <motion.div key={f.title} className={`feat-v2 ${f.span || ''}`}
            initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.5, delay: (i % 4) * 0.07 }}>
            <div className="feat-vis">{VIS[f.vis]}</div>
            <h4>{f.title}</h4>
            <p>{f.text}</p>
          </motion.div>
        ))}
      </div>
    </section>
  )
}
