"use client"

import { useCallback, useEffect, useRef, useState } from "react"

/**
 * Voice narration for the autoresearch loop — zero backend, zero API.
 * Uses the browser-native Web Speech API (speechSynthesis). Drop-in for any React frontend.
 *
 * Usage:
 *   const voice = useVoiceNarration()
 *   voice.speak("Qwen-AgentWorld predicts accuracy 0.93 to 0.95")   // narrates a loop step
 *   <button onClick={voice.toggle}>{voice.enabled ? "🔊 Voice on" : "🔇 Voice off"}</button>
 *
 * Speaking only happens while enabled, so audio never auto-blasts on load.
 */
export function useVoiceNarration(opts: { rate?: number; pitch?: number } = {}) {
  const [supported, setSupported] = useState(false)
  const [enabled, setEnabled] = useState(false)
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null)

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return
    setSupported(true)
    const pick = () => {
      const voices = window.speechSynthesis.getVoices()
      // Prefer a crisp English voice; fall back to the first available.
      voiceRef.current =
        voices.find((v) => /en[-_]US/i.test(v.lang) && /samantha|google|natural|aria/i.test(v.name)) ||
        voices.find((v) => /^en/i.test(v.lang)) ||
        voices[0] ||
        null
    }
    pick()
    window.speechSynthesis.onvoiceschanged = pick
    return () => {
      window.speechSynthesis.onvoiceschanged = null
      window.speechSynthesis.cancel()
    }
  }, [])

  const speak = useCallback(
    (text: string) => {
      if (!enabled || !supported || typeof window === "undefined" || !text) return
      const u = new SpeechSynthesisUtterance(text)
      if (voiceRef.current) u.voice = voiceRef.current
      u.rate = opts.rate ?? 1.05
      u.pitch = opts.pitch ?? 1
      window.speechSynthesis.speak(u) // queues after any in-flight utterance
    },
    [enabled, supported, opts.rate, opts.pitch],
  )

  const cancel = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) window.speechSynthesis.cancel()
  }, [])

  const toggle = useCallback(() => {
    setEnabled((prev) => {
      const next = !prev
      if (!next) cancel()
      else {
        // Speaking once on the click satisfies browser autoplay policies for later utterances.
        const u = new SpeechSynthesisUtterance("Voice narration on.")
        if (voiceRef.current) u.voice = voiceRef.current
        u.rate = opts.rate ?? 1.05
        window.speechSynthesis.speak(u)
      }
      return next
    })
  }, [cancel, opts.rate])

  return { supported, enabled, toggle, speak, cancel }
}

/**
 * Optional voice INPUT — say a command, get the matched keyword back.
 * Chrome-only (webkitSpeechRecognition). Returns null-safe handle.
 *
 *   const listen = useVoiceCommand((cmd) => { if (cmd.includes("run")) runLoop() })
 *   <button onClick={listen.start}>🎙️ Speak a command</button>
 */
export function useVoiceCommand(onResult: (transcript: string) => void) {
  const [supported, setSupported] = useState(false)
  const [listening, setListening] = useState(false)
  const recRef = useRef<any>(null)

  useEffect(() => {
    if (typeof window === "undefined") return
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) return
    setSupported(true)
    const rec = new SR()
    rec.lang = "en-US"
    rec.interimResults = false
    rec.maxAlternatives = 1
    rec.onresult = (e: any) => onResult((e.results[0][0].transcript || "").toLowerCase())
    rec.onend = () => setListening(false)
    rec.onerror = () => setListening(false)
    recRef.current = rec
    return () => {
      try {
        rec.abort()
      } catch {
        /* noop */
      }
    }
  }, [onResult])

  const start = useCallback(() => {
    if (!recRef.current || listening) return
    try {
      recRef.current.start()
      setListening(true)
    } catch {
      setListening(false)
    }
  }, [listening])

  return { supported, listening, start }
}
