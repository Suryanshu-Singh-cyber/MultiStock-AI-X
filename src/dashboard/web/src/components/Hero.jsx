import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

/* lightweight animated candlestick/line canvas background */
function MarketCanvas() {
  const ref = useRef(null)
  useEffect(() => {
    const cv = ref.current, ctx = cv.getContext('2d')
    let W, H, raf, t = 0
    const resize = () => { W = cv.width = cv.offsetWidth; H = cv.height = cv.offsetHeight }
    resize(); window.addEventListener('resize', resize)
    const candles = Array.from({ length: 40 }, (_, i) => ({
      x: i / 40, o: 0.5 + Math.sin(i * 0.7) * 0.15, drift: Math.random() * 0.4 - 0.2 }))
    const draw = () => {
      t += 0.008
      ctx.clearRect(0, 0, W, H)
      // faint grid
      ctx.strokeStyle = 'rgba(99,102,241,0.05)'
      for (let y = 0; y < H; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }
      // flowing price line
      ctx.beginPath()
      for (let x = 0; x <= W; x += 4) {
        const u = x / W
        const y = H * 0.55 + Math.sin(u * 9 + t * 2) * H * 0.08
               + Math.sin(u * 23 - t * 3.4) * H * 0.035
               + Math.sin(u * 4 + t) * H * 0.05
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.strokeStyle = 'rgba(52,211,153,0.35)'; ctx.lineWidth = 2; ctx.stroke()
      // candlesticks
      candles.forEach((c, i) => {
        const cx = (c.x + Math.sin(t * 0.3 + i) * 0.002) * W
        const ch = 30 + Math.sin(t * 1.5 + i * 1.3) * 14 + c.drift * 20
        const up = Math.sin(i * 2.1 + t) > 0
        ctx.strokeStyle = up ? 'rgba(52,211,153,0.20)' : 'rgba(248,113,113,0.20)'
        ctx.fillStyle = up ? 'rgba(52,211,153,0.16)' : 'rgba(248,113,113,0.16)'
        ctx.fillRect(cx - 5, H * 0.62 - ch / 2, 10, ch)
        ctx.beginPath(); ctx.moveTo(cx, H * 0.62 - ch / 2 - 12); ctx.lineTo(cx, H * 0.62 + ch / 2 + 12); ctx.stroke()
      })
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])
  return <canvas ref={ref} className="hero-canvas" />
}

function TiltCard({ children }) {
  const ref = useRef(null)
  const onMove = e => {
    const r = ref.current.getBoundingClientRect()
    const px = (e.clientX - r.left) / r.width - 0.5
    const py = (e.clientY - r.top) / r.height - 0.5
    ref.current.style.transform =
      `perspective(1000px) rotateX(${-py * 8}deg) rotateY(${px * 10}deg)`
  }
  const onLeave = () => { ref.current.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)' }
  return (
    <div ref={ref} className="tilt-card" onMouseMove={onMove} onMouseLeave={onLeave}>
      {children}
    </div>
  )
}

export default function Hero({ d }) {
  const f = d?.forecast || {}
  const sizing = d?.sizing || {}
  const pUp = f.p_up ?? null
  return (
    <section id="overview" className="hero-v2">
      <MarketCanvas />
      <div className="hero-vignette" />
      <div className="hero-inner">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}>
          <div className="hero-badge-v2">AI STOCK FORECASTER&nbsp;&nbsp;|&nbsp;&nbsp;NSE INDIA</div>
          <h1 className="hero-h1">Not every market day<br /><span className="grad">deserves a trade.</span></h1>
          <p className="hero-p">
            An honest AI that predicts <b>POWERGRID.NS</b> tomorrow —
            and tells you when <b>NOT</b> to trade. Probabilities, not promises.
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}>
          <TiltCard>
            <div className="verdict-card-v2">
              <div className="vc-top">
                <div>
                  <div className="vc-label">FORECAST FOR</div>
                  <div className="vc-date">{d?.forecast_date || '—'}</div>
                </div>
                <div className="vc-verdict-wrap">
                  <span className={`vc-verdict ${sizing.no_trade ? 'nt' : 'trade'}`}>
                    {sizing.no_trade ? 'HANDS OFF' : 'TRADE WINDOW'}
                  </span>
                </div>
              </div>
              <div className="vc-grid">
                <div className="vc-cell">
                  <span className="vc-k">Expected move</span>
                  <span className={`vc-v ${f.mu_log_return >= 0 ? 'up' : 'down'}`}>
                    {f.mu_log_return != null ? `${(f.mu_log_return * 100).toFixed(2)}%` : '—'}
                  </span>
                </div>
                <div className="vc-cell">
                  <span className="vc-k">Chance of rising</span>
                  <span className="vc-v">{pUp != null ? `${(pUp * 100).toFixed(1)}%` : '—'}</span>
                </div>
                <div className="vc-cell">
                  <span className="vc-k">Daily wobble σ</span>
                  <span className="vc-v">{f.sigma != null ? `${(f.sigma * 100).toFixed(2)}%` : '—'}</span>
                </div>
                <div className="vc-cell">
                  <span className="vc-k">Data through</span>
                  <span className="vc-v sm">{d?.latest_data_date || '—'}</span>
                </div>
              </div>
              <div className="vc-foot">
                {sizing.no_trade
                  ? 'Expected edge is smaller than trading costs — sitting out is the disciplined move.'
                  : 'Edge exceeds estimated costs — a small position may be justified.'}
              </div>
            </div>
          </TiltCard>
        </motion.div>
      </div>
      <div className="hero-scroll-hint">SCROLL TO EXPLORE ↓</div>
    </section>
  )
}
