# Voice narration — drop-in for the autoresearch demo

Browser-native Web Speech API. **Zero backend, zero API key, free.** Scores the rubric's
"Make it fun / voice-first" axis. Works on any frontend (the Next demo OR Hicham's new site).

## React / Next (this repo)

Hook: `components/lucky-loop/use-voice-narration.ts` (already in the repo).

Wire it into the loop runner in ~5 lines:

```tsx
import { useVoiceNarration } from "./use-voice-narration"

const voice = useVoiceNarration()

// a voice toggle button (place it near the RUN button)
{voice.supported && (
  <button onClick={voice.toggle} className="...">
    {voice.enabled ? "🔊 Voice on" : "🔇 Voice off"}
  </button>
)}

// then, as each loop step reveals, narrate it:
voice.speak("Agent proposes logistic regression.")
voice.speak("Qwen-AgentWorld predicts accuracy 0.93 to 0.95, before any compute.")
voice.speak("Real run: accuracy 0.95. Prediction miss, logged not hidden.")
voice.speak("Verifier: single split, observation only. No robust claim.")
```

Optional voice INPUT (Chrome): `useVoiceCommand((cmd) => { if (cmd.includes("run")) runLoop() })`.

## Vanilla JS (any plain HTML frontend)

Self-contained — paste into a `<script>` or a `.js` file:

```js
function makeVoice({ rate = 1.05 } = {}) {
  let enabled = false, voice = null;
  const synth = window.speechSynthesis;
  const supported = !!synth;
  function pick() {
    const vs = synth ? synth.getVoices() : [];
    voice = vs.find(v => /^en/i.test(v.lang)) || vs[0] || null;
  }
  if (supported) { pick(); synth.onvoiceschanged = pick; }
  return {
    supported,
    isOn: () => enabled,
    toggle() {
      enabled = !enabled;
      if (!enabled) synth.cancel();
      else this.speak("Voice narration on."); // unlocks autoplay
      return enabled;
    },
    speak(text) {
      if (!enabled || !supported || !text) return;
      const u = new SpeechSynthesisUtterance(text);
      if (voice) u.voice = voice;
      u.rate = rate;
      synth.speak(u);
    },
  };
}

// usage:
const voice = makeVoice();
document.querySelector("#voiceBtn").onclick = () => {
  const on = voice.toggle();
  document.querySelector("#voiceBtn").textContent = on ? "🔊 Voice on" : "🔇 Voice off";
};
// when a loop step appears:
voice.speak("Qwen-AgentWorld predicts accuracy 0.93 to 0.95, before any compute.");
```

## Notes
- Speaking only happens while toggled ON → no audio auto-blasts on load (judges hate that).
- The first utterance must follow a user click (browser autoplay policy) — the toggle handles it.
- `speak()` queues, so narrating each step in sequence reads them in order.
- For the live-backend version: narrate the real fields returned by the backend (predicted range,
  actual metric, verifier verdict) — same `speak()` calls, real data.
