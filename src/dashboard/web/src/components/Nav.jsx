import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

const LINKS = [
  ['overview', 'Overview'], ['brain', 'AI Brain'], ['features', 'Features'],
  ['toybox', 'Toy Box'], ['command', 'Command Center'], ['performance', 'Performance'],
]

export default function Nav({ live }) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [])
  return (
    <motion.nav className={`nav ${scrolled ? 'scrolled' : ''}`}
      initial={{ y: -60, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}>
      <a href="#overview" className="nav-brand">
        <span className="nav-logo">M</span>MultiStock<span className="nav-ai">-AI X</span>
      </a>
      <div className="nav-links">
        {LINKS.map(([id, label]) => (
          <a key={id} href={`#${id}`}>{label}</a>
        ))}
      </div>
      <div className="nav-status">
        <span className={`live-dot ${live ? 'on' : ''}`} />
        <span>{live ? 'LIVE' : 'CONNECTING'}</span>
      </div>
    </motion.nav>
  )
}
