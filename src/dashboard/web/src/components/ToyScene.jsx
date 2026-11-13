import React, { useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float, ContactShadows, Html, RoundedBox } from '@react-three/drei'
import * as THREE from 'three'

/* ToyFight palette: cobalt / electric yellow / mint / coral */
const PARTS = [
  { key: 'market', label: 'MARKET SIGNALS', color: '#2563eb', y: -1.05 },
  { key: 'sector', label: 'SECTOR SIGNALS', color: '#eab308', y: 0 },
  { key: 'alpha',  label: 'ALPHA SIGNALS',  color: '#10b981', y: 1.05 },
]

function damp(cur, target, lambda, dt) {
  return THREE.MathUtils.damp(cur, target, lambda, dt)
}

function Block({ part, index, exploded, selected, onPick }) {
  const ref = useRef()
  const targetY = exploded ? part.y * 2.4 : part.y
  useFrame((_, dt) => {
    ref.current.position.y = damp(ref.current.position.y, targetY, 5.5, dt)
    const s = selected === part.key ? 1.18 : 1
    const cs = ref.current.scale.x
    ref.current.scale.setScalar(damp(cs, s, 8, dt))
    ref.current.rotation.y = damp(ref.current.rotation.y,
      selected === part.key ? 0.4 : 0, 6, dt)
  })
  return (
    <group ref={ref} position={[0, part.y, 0]}>
      <Float speed={1.6 + index * 0.35} rotationIntensity={0.25}
        floatIntensity={0.55}>
        <RoundedBox
          args={[3.0, 0.85, 3.0]} radius={0.14} smoothness={6}
          onClick={e => { e.stopPropagation(); onPick(part.key) }}
          onPointerOver={e => { e.stopPropagation(); document.body.style.cursor = 'pointer' }}
          onPointerOut={() => { document.body.style.cursor = 'default' }}>
          <meshStandardMaterial
            color={part.color} roughness={0.28} metalness={0.12}
            emissive={part.color}
            emissiveIntensity={selected === part.key ? 0.4 : 0.1} />
        </RoundedBox>
        {/* toy stud */}
        <mesh position={[0, 0.52, 0]}>
          <cylinderGeometry args={[0.42, 0.46, 0.2, 24]} />
          <meshStandardMaterial color={part.color} roughness={0.3} />
        </mesh>
      </Float>
      <Html position={[0, -0.85, 0]} center distanceFactor={7} zIndexRange={[10, 0]}>
        <div className={`toy-tag ${selected === part.key ? 'on' : ''}`}
          style={{ ['--c']: part.color }}>
          {part.label}
        </div>
      </Html>
    </group>
  )
}

function ConfidenceSphere({ sigma, pUp }) {
  const ref = useRef()
  const size = THREE.MathUtils.clamp(1.9 + sigma * 55, 2.1, 3.6)
  const wob = THREE.MathUtils.clamp(sigma * 200, 0.1, 0.7)
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    ref.current.scale.setScalar(size * (1 + Math.sin(t * 1.8) * wob * 0.07))
    ref.current.rotation.y = t * 0.25
  })
  const sure = Math.abs((pUp ?? 0.5) - 0.5) > 0.08
  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[1, 3]} />
      <meshPhysicalMaterial
        color={sure ? '#34d399' : '#facc15'}
        transparent opacity={0.13} roughness={0.15}
        transmission={0.4} thickness={0.6} ior={1.3} />
    </mesh>
  )
}

function Shield({ show }) {
  const ref = useRef()
  useFrame((_, dt) => {
    const target = show ? 1 : 0.001
    const s = damp(ref.current.scale.x, target, 6, dt)
    ref.current.scale.setScalar(Math.max(s, 0.001))
    ref.current.visible = s > 0.02
  })
  return (
    <group ref={ref} position={[2.6, 1.4, 1.4]} visible={false}>
      <Float speed={2.5} floatIntensity={1.1} rotationIntensity={0.3}>
        <mesh scale={[1, 1.15, 0.45]}>
          <sphereGeometry args={[0.75, 32, 32]} />
          <meshStandardMaterial color="#fb7185" roughness={0.18}
            metalness={0.1} emissive="#fb7185" emissiveIntensity={0.35} />
        </mesh>
        <mesh position={[0, 0.1, 0.42]} scale={[1, 1.15, 0.4]}>
          <torusGeometry args={[0.3, 0.08, 12, 32]} />
          <meshStandardMaterial color="#fff1f2" roughness={0.25} />
        </mesh>
      </Float>
      <Html position={[0, -1.15, 0]} center distanceFactor={7}>
        <div className="toy-tag on" style={{ ['--c']: '#fb7185' }}>REST DAY</div>
      </Html>
    </group>
  )
}

function Rig({ children }) {
  const g = useRef()
  useFrame((state, dt) => {
    g.current.rotation.y = Math.sin(state.clock.getElapsedTime() * 0.18) * 0.28
    g.current.position.y = Math.sin(state.clock.getElapsedTime() * 0.5) * 0.06
  })
  return <group ref={g}>{children}</group>
}

export default function ToyScene({ sigma, pUp, noTrade, exploded, selected, onPick }) {
  return (
    <Canvas camera={{ position: [0, 0.4, 8.2], fov: 40 }}
      dpr={[1, 1.75]} gl={{ antialias: true, alpha: true }}>
      <ambientLight intensity={0.55} />
      <directionalLight position={[5, 8, 5]} intensity={1.4} />
      <directionalLight position={[-6, 3, -4]} intensity={0.5} color="#818cf8" />
      <spotLight position={[0, -4, 4]} intensity={0.6} color="#34d399"
        angle={0.7} penumbra={0.8} />
      <Rig>
        {/* pedestal */}
        <mesh position={[0, -2.35, 0]} receiveShadow>
          <cylinderGeometry args={[3.4, 3.8, 0.35, 64]} />
          <meshStandardMaterial color="#14142a" roughness={0.35} metalness={0.55} />
        </mesh>
        <mesh position={[0, -2.1, 0]}>
          <cylinderGeometry args={[3.0, 3.3, 0.18, 64]} />
          <meshStandardMaterial color="#1e1e3a" roughness={0.4} metalness={0.5} />
        </mesh>
        <ContactShadows position={[0, -1.95, 0]} opacity={0.6}
          scale={11} blur={2.8} far={5} color="#02020a" />

        <ConfidenceSphere sigma={sigma} pUp={pUp} />
        {PARTS.map((p, i) => (
          <Block key={p.key} part={p} index={i}
            exploded={exploded} selected={selected} onPick={onPick} />
        ))}
        <Shield show={noTrade} />
      </Rig>
    </Canvas>
  )
}
