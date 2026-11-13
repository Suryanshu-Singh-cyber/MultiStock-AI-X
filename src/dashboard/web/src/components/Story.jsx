import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/* tiny inline SVG illustrations — no emoji, no icon packs */
const ICONS = {
  collect: (
    <svg viewBox="0 0 64 64" fill="none">
      <rect x="8" y="40" width="10" height="16" rx="2" fill="#3b82f6"/>
      <rect x="22" y="30" width="10" height="26" rx="2" fill="#60a5fa"/>
      <rect x="36" y="20" width="10" height="36" rx="2" fill="#93c5fd"/>
      <rect x="50" y="34" width="10" height="22" rx="2" fill="#3b82f6" opacity=".5"/>
      <path d="M6 56 H58" stroke="#8b90b8" strokeWidth="2" strokeLinecap="round"/>
    </svg>),
  think: (
    <svg viewBox="0 0 64 64" fill="none">
      <circle cx="32" cy="32" r="22" stroke="#a78bfa" strokeWidth="3"/>
      <circle cx="24" cy="28" r="4" fill="#a78bfa"/>
      <circle cx="40" cy="28" r="4" fill="#a78bfa"/>
      <path d="M24 40 Q32 46 40 40" stroke="#a78bfa" strokeWidth="3" strokeLinecap="round"/>
      {[...Array(6)].map((_, i) => {
        const a = i * Math.PI / 3
        return <circle key={i} cx={32 + Math.cos(a) * 30} cy={32 + Math.sin(a) * 30} r="2.5" fill="#8b90b8"/>
      })}
    </svg>),
  vote: (
    <svg viewBox="0 0 64 64" fill="none">
      <rect x="10" y="18" width="14" height="30" rx="4" fill="#34d399"/>
      <rect x="26" y="10" width="14" height="44" rx="4" fill="#facc15"/>
      <rect x="42" y="24" width="14" height="24" rx="4" fill="#3b82f6"/>
    </svg>),
  decide: (
    <svg viewBox="0 0 64 64" fill="none">
      <path d="M32 8 L52 16 V32 C52 44 44 52 32 56 C20 52 12 44 12 32 V16 Z"
        stroke="#f472b6" strokeWidth="3" fill="rgba(244,114,182,.12)"/>
      <path d="M24 32 L30 38 L41 25" stroke="#f472b6" strokeWidth="4"
        strokeLinecap="round" strokeLinejoin="round"/>
    </svg>),
}

const STAGES = [
  { key: 'collect', title: 'COLLECT',
    text: 'Every evening it gathers 19 years of POWERGRID prices — plus 9 rival stocks, NIFTY 50, the power-sector index, the fear gauge (VIX), dollar-rupee and crude oil. Like giving the AI a giant memory.',
    stat: '15 symbols · ~4,700 trading days' },
  { key: 'think', title: 'THINK',
    text: '184 detective clues hunt for patterns — momentum, volatility, volume, what rivals did. Then a strict no-cheating filter keeps only the 53 that truly help.',
    stat: '184 clues in → 53 survive' },
  { key: 'vote', title: 'VOTE',
    text: 'Four models guess independently: a math formula, a random forest, gradient boosting, and a deep-learning LSTM that reads the last 60 days like a story.',
    stat: '4 models · ensemble average' },
  { key: 'decide', title: 'DECIDE',
    text: 'The risk engine compares expected gain vs real trading cost (~0.26%). If the edge is smaller than the fee — which is most days — it says DO NOTHING.',
    stat: 'edge > cost? → trade : rest' },
]

export default function Story() {
  const [i, setI] = useState(0)
  const [paused, setPaused] = useState(false)
  useEffect(() => {
    if (paused) return
    const t = setInterval(() => setI(x => (x + 1) % STAGES.length), 5000)
    return () => clearInterval(t)
  }, [paused])
  const s = STAGES[i]

  return (
    <section id="brain" className="story-v2">
      <div className="sec-head">
        <span className="sec-kicker">UNDER THE HOOD</span>
        <h2>How does the robot brain work?</h2>
        <p>Four steps, every single night. No black box — just a pipeline you can audit.</p>
      </div>

      <div className="stage-track">
        {STAGES.map((st, idx) => (
          <button key={st.key}
            className={`stage-tab ${idx === i ? 'on' : ''} ${idx < i ? 'done' : ''}`}
            onClick={() => setI(idx)}>
            <span className="stage-num">{idx + 1}</span>
            <span className="stage-name">{st.title}</span>
          </button>
        ))}
        <button className="stage-pause" onClick={() => setPaused(p => !p)}
          title={paused ? 'Resume' : 'Pause'}>
          {paused ? '▶' : '❚❚'}
        </button>
      </div>

      <div className="stage-progress"><div style={{ width: `${(i + 1) * 25}%` }} /></div>

      <AnimatePresence mode="wait">
        <motion.div key={i} className="stage-panel"
          initial={{ opacity: 0, x: 60 }} animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -60 }} transition={{ duration: 0.45, ease: 'easeOut' }}>
          <div className="stage-illus">{ICONS[s.key]}</div>
          <div className="stage-body">
            <div className="stage-stat">{s.stat}</div>
            <p>{s.text}</p>
          </div>
        </motion.div>
      </AnimatePresence>
    </section>
  )
}
