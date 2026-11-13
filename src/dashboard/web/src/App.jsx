import React, { useEffect, useMemo, useRef, useState } from 'react'
import Nav from './components/Nav'
import Hero from './components/Hero'
import Story from './components/Story'
import Features from './components/Features'
import ToyBox from './components/ToyBox'
import CommandCenter from './components/CommandCenter'
import Performance from './components/Performance'
import './styles.css'

const API = '/api'

function useData() {
  const [data, setData] = useState(null)
  const [prices, setPrices] = useState([])
  const [score, setScore] = useState(null)
  const [equity, setEquity] = useState(null)
  const [tape, setTape] = useState([])
  const [live, setLive] = useState(null)
  const ws = useRef(null)

  const load = useMemo(() => () => {
    fetch(`${API}/forecast/latest`).then(r => r.json()).then(setData).catch(() => {})
    fetch(`${API}/price/history?days=90`).then(r => r.json()).then(setPrices).catch(() => {})
    fetch(`${API}/scorecard`).then(r => r.json()).then(setScore).catch(() => {})
    fetch(`${API}/backtest/equity?model=arima`).then(r => r.json()).then(setEquity).catch(() => {})
  }, [])

  useEffect(() => {
    load()
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws.current = new WebSocket(`${proto}://${location.host}/ws/stream`)
    ws.current.onmessage = e => {
      try {
        const m = JSON.parse(e.data)
        setLive(m)
        if (m.type === 'price_tick') {
          setTape(t => [...t.slice(-30), `${m.symbol} ₹${Number(m.price).toFixed(2)}`])
        } else if (m.forecast_date) { setData(m); load() }
      } catch {}
    }
    const t = setInterval(load, 30000)
    return () => { ws.current?.close(); clearInterval(t) }
  }, [load])

  return { data, prices, score, equity, tape, live, load }
}

function useRefresh(load) {
  const [st, setSt] = useState({ running: false, msg: '' })
  const start = async () => {
    setSt({ running: true, msg: 'Starting…' })
    try {
      const r = await fetch(`${API}/forecast/refresh`, { method: 'POST' })
      const j = await r.json()
      if (j.status === 'already_running') { setSt({ running: true, msg: 'Already running…' }); return }
      const poll = setInterval(async () => {
        const s = await (await fetch(`${API}/forecast/refresh/status`)).json()
        if (!s.running) {
          clearInterval(poll)
          setSt({ running: false, msg: s.error ? `Failed: ${s.error}` : 'Done — new forecast live ✅' })
          setTimeout(() => setSt({ running: false, msg: '' }), 6000)
          load?.()
        } else setSt({ running: true, msg: 'AI is thinking… (30–90s)' })
      }, 5000)
    } catch (e) { setSt({ running: false, msg: `Error: ${e}` }) }
  }
  return { st, start }
}

export default function App() {
  const { data, prices, score, equity, tape, live, load } = useData()
  const { st: refresh, start: doRefresh } = useRefresh(load)
  return (
    <div className="app-v2">
      <Nav live={!!live} />
      <Hero d={data} />
      <Story />
      <Features />
      <ToyBox d={data} score={score} />
      <CommandCenter d={data} prices={prices} score={score} equity={equity}
        tape={tape} live={live} refresh={refresh} doRefresh={doRefresh} />
      <Performance equity={equity} score={score} />
      <footer className="footer-v2">
        <div>
          <b>MultiStock-AI X</b> · probabilities, not guarantees · not financial advice
        </div>
        <div className="muted">
          {data?.generated_at ? `forecast generated ${data.generated_at.slice(0, 19).replace('T', ' ')}` : ''}
          {' '}· research tool — verify everything before trading
        </div>
      </footer>
    </div>
  )
}
