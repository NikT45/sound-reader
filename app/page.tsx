"use client";

import { useState, useRef } from "react";

interface Segment {
  type: "narration" | "dialogue";
  text: string;
  speaker?: string;
}

interface VoiceInfo {
  name: string;
  description: string;
}

interface SegmentTiming { segmentIndex: number; startTime: number; endTime: number; }
interface SfxItem { segmentIndex: number; prompt: string; startTime: number; audioBase64: string; }
interface DialogueTone { segmentIndex: number; stability: number; style: number; emotion: string; }

interface GenerateResponse {
  segments: Segment[];
  voiceMap: Record<string, VoiceInfo>;
  segmentsAudio: string[];
  segmentTimings: SegmentTiming[];
  sfxList: SfxItem[];
  dialogueTones: DialogueTone[];
  error?: string;
}

// Ordered speaker colors for dialogue characters (skipping narrator)
const SPEAKER_COLORS = [
  "text-rose-400 border-rose-800 bg-rose-950/40",
  "text-sky-400 border-sky-800 bg-sky-950/40",
  "text-emerald-400 border-emerald-800 bg-emerald-950/40",
  "text-violet-400 border-violet-800 bg-violet-950/40",
  "text-orange-400 border-orange-800 bg-orange-950/40",
  "text-pink-400 border-pink-800 bg-pink-950/40",
];

const SAMPLE_TEXT = `The old man sat by the fire, his hands trembling slightly as he poured the tea.

"You shouldn't have come here," he said without looking up. "It isn't safe."

Margaret stepped inside, closing the door softly behind her. The room smelled of woodsmoke and old paper.

"I had no choice," she replied. "They know about the letters."

He finally raised his eyes to meet hers — pale, watery, but still sharp. "All of them?"

"Every last one."

He set down the teapot and stared into the fire for a long moment. "Then we have perhaps two days," he murmured, more to himself than to her. "Perhaps less."`;

// WAV encoder helpers
function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function audioBufferToWav(buf: AudioBuffer): ArrayBuffer {
  const numCh = buf.numberOfChannels, sr = buf.sampleRate, len = buf.length;
  const dataSize = len * numCh * 2;
  const ab = new ArrayBuffer(44 + dataSize); const v = new DataView(ab);
  writeString(v, 0, "RIFF"); v.setUint32(4, 36 + dataSize, true); writeString(v, 8, "WAVE");
  writeString(v, 12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, numCh, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * numCh * 2, true); v.setUint16(32, numCh * 2, true); v.setUint16(34, 16, true);
  writeString(v, 36, "data"); v.setUint32(40, dataSize, true);
  let off = 44;
  for (let f = 0; f < len; f++)
    for (let ch = 0; ch < numCh; ch++) {
      const s = Math.max(-1, Math.min(1, buf.getChannelData(ch)[f]));
      v.setInt16(off, s < 0 ? s * 32768 : s * 32767, true); off += 2;
    }
  return ab;
}

// Gaps inserted between segments for natural cadence
const GAP_NAR_TO_DLG = 0.22;  // pause before a character speaks
const GAP_DLG_TO_NAR = 0.18;  // pause before narration resumes
const GAP_DLG_TO_DLG = 0.12;  // brief beat between different speakers

async function decodeB64Audio(b64: string): Promise<AudioBuffer> {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(bytes.buffer.slice(0));
  await ctx.close();
  return buf;
}

async function mixAudio(segmentsAudio: string[], segments: Segment[], sfxList: SfxItem[]): Promise<string> {
  // Decode all speech segments
  const segBuffers: AudioBuffer[] = [];
  for (const b64 of segmentsAudio) {
    try { segBuffers.push(await decodeB64Audio(b64)); }
    catch { segBuffers.push(new AudioContext().createBuffer(1, 1, 44100)); }
  }

  // Calculate absolute start times with cadence gaps
  const startTimes: number[] = [];
  let cursor = 0;
  for (let i = 0; i < segments.length; i++) {
    if (i > 0) {
      const prev = segments[i - 1];
      const curr = segments[i];
      if (prev.type === "narration" && curr.type === "dialogue") cursor += GAP_NAR_TO_DLG;
      else if (prev.type === "dialogue" && curr.type === "narration") cursor += GAP_DLG_TO_NAR;
      else if (prev.type === "dialogue" && curr.type === "dialogue") cursor += GAP_DLG_TO_DLG;
    }
    startTimes.push(cursor);
    cursor += segBuffers[i]?.duration ?? 0;
  }

  // Decode SFX — position by looking up the narration segment's calculated start time
  const sfxDecoded: { buffer: AudioBuffer; startTime: number }[] = [];
  for (const sfx of sfxList) {
    try {
      const buf = await decodeB64Audio(sfx.audioBase64);
      const t = startTimes[sfx.segmentIndex] ?? sfx.startTime;
      sfxDecoded.push({ buffer: buf, startTime: t });
    } catch { /* skip */ }
  }

  const sampleRate = segBuffers[0]?.sampleRate ?? 44100;
  const numChannels = segBuffers[0]?.numberOfChannels ?? 1;
  const totalDuration = Math.max(
    cursor,
    sfxDecoded.reduce((m, { buffer, startTime }) => Math.max(m, startTime + buffer.duration), 0)
  );

  const offCtx = new OfflineAudioContext(numChannels, Math.ceil(totalDuration * sampleRate), sampleRate);

  for (let i = 0; i < segBuffers.length; i++) {
    const src = offCtx.createBufferSource();
    src.buffer = segBuffers[i];
    src.connect(offCtx.destination);
    src.start(startTimes[i]);
  }

  const TARGET_PEAK = 0.6;
  for (const { buffer, startTime } of sfxDecoded) {
    // Find peak amplitude across all channels
    let peak = 0;
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const data = buffer.getChannelData(ch);
      for (let s = 0; s < data.length; s++) {
        const abs = Math.abs(data[s]);
        if (abs > peak) peak = abs;
      }
    }
    const src = offCtx.createBufferSource(); src.buffer = buffer;
    const gain = offCtx.createGain();
    gain.gain.value = peak > 0 ? TARGET_PEAK / peak : TARGET_PEAK;
    src.connect(gain); gain.connect(offCtx.destination); src.start(startTime);
  }

  const rendered = await offCtx.startRendering();
  return URL.createObjectURL(new Blob([audioBufferToWav(rendered)], { type: "audio/wav" }));
}

export default function Home() {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const prevAudioUrl = useRef<string | null>(null);

  async function handleGenerate() {
    if (!text.trim()) return;
    setLoading(true);
    setStatus("Parsing text with Gemini...");
    setResult(null);
    if (prevAudioUrl.current) {
      URL.revokeObjectURL(prevAudioUrl.current);
      prevAudioUrl.current = null;
    }
    setAudioUrl(null);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      setStatus("Generating audio with ElevenLabs...");
      const data: GenerateResponse = await res.json();

      if (!res.ok || data.error) {
        setStatus(`Error: ${data.error ?? "Unknown error"}`);
        setLoading(false);
        return;
      }

      setStatus("Mixing audio with sound effects...");
      const url = await mixAudio(data.segmentsAudio, data.segments, data.sfxList ?? []);
      prevAudioUrl.current = url;
      setAudioUrl(url);
      setResult(data);
      setStatus("");
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }

  // Build a stable speaker → color index map from the response voiceMap
  function getSpeakerColorMap(voiceMap: Record<string, VoiceInfo>): Record<string, string> {
    const colorMap: Record<string, string> = {};
    let idx = 0;
    for (const speaker of Object.keys(voiceMap)) {
      if (speaker === "narrator") continue;
      colorMap[speaker] = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
      idx++;
    }
    return colorMap;
  }

  const speakerColorMap = result ? getSpeakerColorMap(result.voiceMap) : {};
  const sfxMap = result ? Object.fromEntries((result.sfxList ?? []).map(s => [s.segmentIndex, s.prompt])) : {};
  const toneMap = result ? Object.fromEntries((result.dialogueTones ?? []).map(t => [t.segmentIndex, t])) : {} as Record<number, DialogueTone>;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-serif">
      <div className="max-w-3xl mx-auto px-6 py-16">
        {/* Header */}
        <header className="mb-12 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-amber-400 mb-2">
            Sound Reader
          </h1>
          <p className="text-zinc-500 text-sm tracking-widest uppercase">
            Immersive Book Audio Generator
          </p>
        </header>

        {/* Input */}
        <section className="mb-8">
          <label className="block text-xs font-sans font-semibold text-zinc-400 tracking-widest uppercase mb-3">
            Book Excerpt
          </label>
          <textarea
            className="w-full h-56 bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 text-zinc-200 text-sm leading-relaxed resize-none focus:outline-none focus:border-amber-600 placeholder:text-zinc-600 font-sans"
            placeholder="Paste a passage of text with dialogue and narration..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={loading}
          />
        </section>

        {/* Generate Button */}
        <div className="flex items-center gap-4 mb-10">
          <button
            onClick={handleGenerate}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-6 py-2.5 bg-amber-500 hover:bg-amber-400 disabled:bg-zinc-700 disabled:text-zinc-500 text-zinc-950 font-sans font-semibold text-sm rounded-full transition-colors"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Processing…
              </>
            ) : (
              "Generate Audio"
            )}
          </button>
          {status && (
            <span className="text-xs font-sans text-zinc-400 italic">{status}</span>
          )}
        </div>

        {/* Audio Player */}
        {audioUrl && (
          <section className="mb-10">
            <label className="block text-xs font-sans font-semibold text-zinc-400 tracking-widest uppercase mb-3">
              Audio
            </label>
            <audio
              controls
              src={audioUrl}
              className="w-full rounded-lg"
            />
          </section>
        )}

        {/* Voice Map Legend */}
        {result && Object.keys(result.voiceMap).length > 0 && (
          <section className="mb-8">
            <label className="block text-xs font-sans font-semibold text-zinc-400 tracking-widest uppercase mb-3">
              Voice Cast
            </label>
            <div className="flex flex-col gap-2">
              {Object.entries(result.voiceMap).map(([speaker, voice]) => {
                const colorClass = speaker === "narrator"
                  ? "text-zinc-300 border-zinc-700 bg-zinc-800/60"
                  : speakerColorMap[speaker] ?? SPEAKER_COLORS[0];
                const textColorClass = colorClass.split(" ")[0];
                return (
                  <div
                    key={speaker}
                    className={`flex items-baseline gap-2 px-3 py-2 rounded-lg border text-xs font-sans ${colorClass}`}
                  >
                    <span className={`font-bold shrink-0 ${textColorClass}`}>{speaker}</span>
                    <span className="text-zinc-500">·</span>
                    <span className="font-medium text-zinc-300 shrink-0">{voice.name}</span>
                    <span className="text-zinc-500 shrink-0">—</span>
                    <span className="text-zinc-500 italic">{voice.description}</span>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* Debug Panel */}
        {result && (
          <section className="mb-8">
            <button
              onClick={() => setDebugOpen(o => !o)}
              className="flex items-center gap-2 text-xs font-sans font-semibold text-zinc-500 tracking-widest uppercase hover:text-zinc-300 transition-colors"
            >
              <svg className={`h-3 w-3 transition-transform ${debugOpen ? "rotate-90" : ""}`} viewBox="0 0 6 10" fill="currentColor">
                <path d="M1 1l4 4-4 4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              ElevenLabs API Calls ({result.segments.length + (result.sfxList?.length ?? 0)})
            </button>

            {debugOpen && (
              <div className="mt-3 rounded-lg border border-zinc-800 overflow-hidden font-mono text-xs">
                {/* TTS calls */}
                <div className="px-3 py-2 bg-zinc-900 border-b border-zinc-800 text-zinc-500 uppercase tracking-widest text-[10px] font-sans font-semibold">
                  Text-to-Speech · convertWithTimestamps · eleven_multilingual_v2
                </div>
                {result.segments.map((seg, i) => {
                  const speakerKey = seg.type === "narration" ? "narrator" : (seg.speaker ?? "unknown");
                  const voice = result.voiceMap[speakerKey] ?? result.voiceMap["narrator"];
                  const timing = result.segmentTimings?.find(t => t.segmentIndex === i);
                  const tone = result.dialogueTones?.find(t => t.segmentIndex === i);
                  const s = seg.type === "narration"
                    ? { stability: 0.65, style: 0.1 }
                    : { stability: tone?.stability ?? 0.35, style: tone?.style ?? 0.6 };
                  const settings = `stability: ${s.stability.toFixed(2)}  similarity: 0.80  style: ${s.style.toFixed(2)}`;
                  return (
                    <div key={i} className="flex gap-0 border-b border-zinc-800/60 last:border-0">
                      <div className="shrink-0 w-8 flex items-start justify-center pt-3 text-zinc-600">{i}</div>
                      <div className="flex-1 px-2 py-2.5 border-l border-zinc-800/60 space-y-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={seg.type === "narration" ? "text-zinc-400" : "text-sky-400"}>
                            {seg.type}
                          </span>
                          {seg.speaker && <span className="text-zinc-600">·</span>}
                          {seg.speaker && <span className="text-sky-300">{seg.speaker}</span>}
                          {tone && <span className="text-zinc-500 italic">{tone.emotion}</span>}
                          <span className="text-zinc-600">→</span>
                          <span className="text-amber-400">{voice?.name}</span>
                          <span className="text-zinc-600 italic">{voice?.description}</span>
                          {timing && (
                            <span className="text-zinc-600 ml-auto">
                              {timing.startTime.toFixed(2)}s – {timing.endTime.toFixed(2)}s
                            </span>
                          )}
                        </div>
                        <div className="text-zinc-600">{settings}</div>
                        <div className="text-zinc-300 leading-relaxed line-clamp-2">&ldquo;{seg.text}&rdquo;</div>
                      </div>
                    </div>
                  );
                })}

                {/* SFX calls */}
                {(result.sfxList?.length ?? 0) > 0 && (
                  <>
                    <div className="px-3 py-2 bg-zinc-900 border-t border-b border-zinc-800 text-zinc-500 uppercase tracking-widest text-[10px] font-sans font-semibold">
                      Sound Effects · textToSoundEffects.convert · duration: 3s · prompt_influence: 0.3
                    </div>
                    {result.sfxList.map((sfx, i) => {
                      const timing = result.segmentTimings?.find(t => t.segmentIndex === sfx.segmentIndex);
                      return (
                        <div key={i} className="flex gap-0 border-b border-zinc-800/60 last:border-0">
                          <div className="shrink-0 w-8 flex items-start justify-center pt-3 text-zinc-600">{sfx.segmentIndex}</div>
                          <div className="flex-1 px-2 py-2.5 border-l border-zinc-800/60 space-y-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-amber-400">🔊 {sfx.prompt}</span>
                              {timing && (
                                <span className="text-zinc-600 ml-auto">@ {sfx.startTime.toFixed(2)}s</span>
                              )}
                            </div>
                            <div className="text-zinc-600">text: &ldquo;{sfx.prompt}&rdquo;  duration_seconds: 3  prompt_influence: 0.3</div>
                          </div>
                        </div>
                      );
                    })}
                  </>
                )}
              </div>
            )}
          </section>
        )}

        {/* Segment List */}
        {result && result.segments.length > 0 && (
          <section>
            <label className="block text-xs font-sans font-semibold text-zinc-400 tracking-widest uppercase mb-3">
              Segments
            </label>
            <div className="space-y-2">
              {result.segments.map((seg, i) => {
                if (seg.type === "narration") {
                  return (
                    <div
                      key={i}
                      className="px-4 py-3 rounded-lg bg-zinc-800/50 border border-zinc-800 text-zinc-400 text-sm leading-relaxed italic font-serif"
                    >
                      {seg.text}
                      {sfxMap[i] && (
                        <span className="not-italic ml-2 inline-flex items-center gap-1 text-xs font-sans text-amber-400 border border-amber-800 bg-amber-950/40 rounded px-1.5 py-0.5">
                          🔊 {sfxMap[i]}
                        </span>
                      )}
                    </div>
                  );
                }

                const speaker = seg.speaker ?? "unknown";
                const colorClass = speakerColorMap[speaker] ?? SPEAKER_COLORS[0];
                const [textColor, borderColor, bgColor] = colorClass.split(" ");
                const tone = toneMap[i];

                return (
                  <div
                    key={i}
                    className={`px-4 py-3 rounded-lg border text-sm leading-relaxed font-serif ${borderColor} ${bgColor}`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs font-sans font-bold tracking-widest uppercase ${textColor}`}>
                        {speaker}
                      </span>
                      {tone && (
                        <span className="inline-flex items-center text-xs font-sans text-zinc-400 border border-zinc-700 bg-zinc-800/60 rounded px-1.5 py-0.5 italic">
                          {tone.emotion}
                        </span>
                      )}
                    </div>
                    <span className="text-zinc-200">{seg.text}</span>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
