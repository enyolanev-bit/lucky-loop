/**
 * Lucky Loop — voice narration (drop-in, zero backend, zero API key).
 * Browser-native Web Speech API (speechSynthesis). Works in any frontend.
 *
 * This is an ES module. Two ways to load it:
 *   1. import { makeVoice, LOOP_NARRATION } from "./voice.js"   (bundler / React / Next)
 *   2. <script type="module" src="voice.js"></script>          → also sets window.LuckyVoice
 *   (a plain <script src> WITHOUT type="module" will NOT work — `export` needs a module.)
 *
 * Wiring (≈5 lines) ───────────────────────────────────────────────
 *   const voice = makeVoice();
 *
 *   // a toggle button — toggle() returns the new on/off state
 *   voiceBtn.onclick = () => {
 *     const on = voice.toggle();
 *     voiceBtn.textContent = on ? "🔊 Voice on" : "🔇 Voice off";
 *   };
 *
 *   // narrate a loop step as it reveals (queues, reads in order)
 *   voice.speak("Qwen-AgentWorld predicts accuracy 0.93 to 0.95, before any compute.");
 *
 * Notes:
 *   - speak() only emits while toggled ON → no audio auto-blasts on load.
 *   - The first utterance must follow a user click (autoplay policy). toggle() handles it.
 *   - For the live backend: feed speak() the real fields (predicted range, actual metric,
 *     verifier verdict) instead of the canned lines below — same calls, real data.
 */

export function makeVoice({ rate = 1.05, pitch = 1 } = {}) {
  const synth = typeof window !== "undefined" ? window.speechSynthesis : null;
  const supported = !!synth;
  let enabled = false;
  let voice = null;

  function pick() {
    const vs = synth ? synth.getVoices() : [];
    // Prefer a crisp English voice; fall back to the first available.
    voice =
      vs.find((v) => /en[-_]US/i.test(v.lang) && /samantha|google|natural|aria/i.test(v.name)) ||
      vs.find((v) => /^en/i.test(v.lang)) ||
      vs[0] ||
      null;
  }
  if (supported) {
    pick();
    synth.onvoiceschanged = pick; // voices often load async
  }

  return {
    supported,
    isOn: () => enabled,

    /** Flip narration on/off. Returns the new state. Speaking once on the
     *  click satisfies the browser autoplay policy for later utterances. */
    toggle() {
      enabled = !enabled;
      if (!enabled) synth && synth.cancel();
      else this.speak("Voice narration on. Let's do some science.");
      return enabled;
    },

    /** Queue a line. No-op while OFF, unsupported, or empty. */
    speak(text) {
      if (!enabled || !supported || !text) return;
      const u = new SpeechSynthesisUtterance(text);
      if (voice) u.voice = voice;
      u.rate = rate;
      u.pitch = pitch;
      synth.speak(u);
    },

    /** Stop and clear any queued speech. */
    cancel() {
      if (supported) synth.cancel();
    },
  };
}

/**
 * Fun narration script for the autoresearch loop, in order.
 * Map each one to the matching step in the RUN animation, or replace `.say`
 * with the real backend fields for the live version.
 */
export const LOOP_NARRATION = [
  { step: "question", say: "New research question incoming. Let's see if the machine can actually do science." },
  { step: "propose",  say: "Agent proposes logistic regression. Bold, classic, let's roll." },
  { step: "predict",  say: "Qwen-AgentWorld predicts accuracy zero point nine three to zero point nine five, before any compute. Spooky." },
  { step: "decision", say: "Forecast says it's cheap and informative. Verdict: run it." },
  { step: "run",      say: "Real run: accuracy zero point nine five. The world model nailed the call." },
  { step: "verify",   say: "Verifier says: single split, observation only. No robust claim. We log it honestly, not proudly." },
  { step: "next",     say: "Next up: add cross-validation before bragging. And the loop goes round again." },
];

// Optional voice INPUT (Chrome only). Say a command, get the lowercased transcript.
//   const listen = makeVoiceCommand((cmd) => { if (cmd.includes("run")) runLoop(); });
//   micBtn.onclick = listen.start;
export function makeVoiceCommand(onResult) {
  const SR =
    typeof window !== "undefined" && (window.SpeechRecognition || window.webkitSpeechRecognition);
  if (!SR) return { supported: false, start() {} };
  const rec = new SR();
  rec.lang = "en-US";
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  rec.onresult = (e) => onResult((e.results[0][0].transcript || "").toLowerCase());
  return {
    supported: true,
    start() {
      try {
        rec.start();
      } catch {
        /* already listening */
      }
    },
  };
}

// Also expose on window for plain <script> usage (no bundler needed).
if (typeof window !== "undefined") {
  window.LuckyVoice = { makeVoice, makeVoiceCommand, LOOP_NARRATION };
}
