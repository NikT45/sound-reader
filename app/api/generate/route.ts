import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { ElevenLabsClient } from "elevenlabs";
import WebSocket from "ws";

interface ScriptScene {
  index: number;
  heading: string;
  action: string;
}

interface ScriptDialogueLine {
  sceneIndex: number;
  lineIndex: number;
  character: string;
  parenthetical?: string;
  text: string;
}

interface StoryboardFrame {
  sceneIndex: number;
  imageBase64: string;
  mimeType: string;
}

interface SfxSuggestion { sceneIndex: number; prompt: string | null; }
interface SfxItem { sceneIndex: number; prompt: string; startTime: number; audioBase64: string; }
interface DialogueTone { lineIndex: number; stability: number; style: number; emotion: string; }
interface MusicSuggestion { sceneIndex: number; prompt: string | null; }
interface MusicItem { sceneIndex: number; prompt: string; startTime: number; audioBase64: string; }

interface VoiceAssignment {
  voiceId: string;
  voiceName: string;
  description: string;
}

// Run async tasks with bounded concurrency
async function runWithConcurrency<T>(
  tasks: Array<() => Promise<T | null>>,
  concurrency: number
): Promise<Array<T | null>> {
  const results: Array<T | null> = new Array(tasks.length).fill(null);
  let next = 0;
  async function worker() {
    while (next < tasks.length) {
      const i = next++;
      results[i] = await tasks[i]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, tasks.length) }, worker));
  return results;
}

async function streamToBuffer(stream: AsyncIterable<Buffer>): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}

// PCM-16 → WAV header (server-side, uses Node Buffer)
function pcmToWav(pcm: Buffer, channels: number, sampleRate: number): Buffer {
  const dataSize = pcm.length;
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + dataSize, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);          // PCM
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * channels * 2, 28);
  header.writeUInt16LE(channels * 2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(dataSize, 40);
  return Buffer.concat([header, pcm]);
}

// Collect `durationSeconds` of Lyria PCM audio and return as WAV base64.
// Bypasses the SDK's live.music.connect() which generates a double-slash URL (SDK bug).
async function generateLyriaMusic(
  apiKey: string,
  musicPrompt: string,
  durationSeconds: number
): Promise<string | null> {
  const chunks: Buffer[] = [];
  let totalBytes = 0;
  // PCM16 stereo 48 kHz → 4 bytes per frame
  const targetBytes = Math.ceil(Math.min(durationSeconds, 25) * 48000 * 4);

  const WS_URL =
    `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic?key=${apiKey}`;

  return new Promise<string | null>((resolve) => {
    let finished = false;

    const finish = () => {
      if (finished) return;
      finished = true;
      try { ws.close(); } catch {}
      if (!chunks.length) { resolve(null); return; }
      const pcm = Buffer.concat(chunks);
      resolve(pcmToWav(pcm, 2, 48000).toString("base64"));
    };

    const timer = setTimeout(() => {
      console.warn("[LYRIA] timeout — using collected audio");
      finish();
    }, 40_000);

    const ws = new WebSocket(WS_URL, {
      headers: {
        "content-type": "application/json",
        "user-agent": "google-genai-sdk/1.43.0 gl-node/v23.7.0",
      },
    });

    ws.on("open", () => {
      console.log("[LYRIA] connected, sending setup");
      ws.send(JSON.stringify({ setup: { model: "models/lyria-realtime-exp" } }));
    });

    ws.on("message", (raw: Buffer | string) => {
      const rawStr = raw.toString();
      console.log("[LYRIA] message:", rawStr.slice(0, 600));

      let msg: Record<string, unknown>;
      try { msg = JSON.parse(rawStr) as Record<string, unknown>; }
      catch (e) { console.warn("[LYRIA] non-JSON message, length:", rawStr.length); return; }

      if (msg.setupComplete !== undefined) {
        console.log("[LYRIA] setupComplete — sending prompt + PLAY");
        ws.send(JSON.stringify({ clientContent: { weightedPrompts: [{ text: musicPrompt, weight: 1.0 }] } }));
        ws.send(JSON.stringify({ playbackControl: "PLAY" }));
      }

      if (msg.filteredPrompt !== undefined) {
        console.warn("[LYRIA] prompt filtered:", JSON.stringify(msg.filteredPrompt));
      }

      const audioChunks = (msg as { serverContent?: { audioChunks?: { data?: string }[] } })
        .serverContent?.audioChunks ?? [];
      if (audioChunks.length) console.log("[LYRIA] audio chunks received:", audioChunks.length, "totalBytes so far:", totalBytes);
      for (const chunk of audioChunks) {
        if (chunk.data && !finished) {
          const buf = Buffer.from(chunk.data, "base64");
          chunks.push(buf);
          totalBytes += buf.length;
          if (totalBytes >= targetBytes) { clearTimeout(timer); finish(); }
        }
      }
    });

    ws.on("error", (e: Error) => { console.error("[LYRIA] ws error:", e.message); clearTimeout(timer); finish(); });
    ws.on("close", (code: number, reason: Buffer) => {
      console.log("[LYRIA] closed — code:", code, "reason:", reason.toString(), "chunks:", chunks.length, "bytes:", totalBytes);
      clearTimeout(timer); finish();
    });
  });
}

export async function POST(req: NextRequest) {
  try {
    const { text } = await req.json();
    if (!text || typeof text !== "string") {
      return NextResponse.json({ error: "text is required" }, { status: 400 });
    }

    const geminiKey = process.env.GEMINI_API_KEY;
    const elKey = process.env.ELEVENLABS_API_KEY;
    if (!geminiKey || !elKey) {
      return NextResponse.json(
        { error: "Missing API keys. Set GEMINI_API_KEY and ELEVENLABS_API_KEY in .env.local" },
        { status: 500 }
      );
    }

    const genai = new GoogleGenAI({ apiKey: geminiKey });
    const elClient = new ElevenLabsClient({ apiKey: elKey });

    // ── Step 1: Parse screenplay or prose + fetch ElevenLabs voices in parallel ──────
    const parsePrompt = `You are a script parser for cinematic audio. Your job is to convert ALL text into a spoken audio experience — nothing is skipped. Every word must appear somewhere in dialogueLines.

━━━ SCENES ━━━
Split into VISUAL SCENES for a storyboard. Aim for 4–8+ scenes.
- SCREENPLAY: each INT./EXT. heading is a boundary; also split within a location at major visual beats (camera directions like TIGHT ON, SMASH CUT, CUT TO, new character entrance, dramatic tonal shift).
- PROSE: create a new scene every 3–5 paragraphs or at each major visual beat. Give each a descriptive screenplay-style heading (e.g. "INT. SERVER ROOM - WIDE - ENGINEERS REACT").
- Each scene's "action" field: a SHORT summary of the setting/action for image generation only (1–2 sentences max). Do NOT put narration text here — it goes in dialogueLines.

━━━ DIALOGUE LINES ━━━
Include EVERY piece of text in dialogueLines, in the ORDER it appears in the source:
- Narration, description, action, attribution, stage directions → character: "narrator"
- Character speech → character: ALL-CAPS name (e.g. "HARRY", "SORTING ALGORITHM", "ENGINEER 1")

PROSE rules:
- Non-quoted text (including attribution like "said Harry", "he thought") → narrator line.
- Text inside quotation marks (" " or ' ') → spoken by the identified character.
- Split long narrator passages at natural sentence or paragraph breaks — each chunk is a separate narrator line (keep each under ~40 words for natural pacing).

SCREENPLAY rules:
- Action block text → narrator line(s) before that scene's character dialogue. Split long blocks into multiple narrator lines at paragraph breaks.
- Parenthetical in () between character name and speech → parenthetical field (optional).
- ALL-CAPS standalone line → character name for the following dialogue.

ORDERING: dialogueLines must be in strict top-to-bottom source order. narrator lines appear between character lines exactly where the text places them. lineIndex is globally sequential starting at 0.

Return ONLY valid JSON, no markdown, no explanation:
{"scenes":[{"index":0,"heading":"INT. SERVER ROOM - DAY","action":"Brief visual summary."}],"dialogueLines":[{"sceneIndex":0,"lineIndex":0,"character":"narrator","text":"As Harry stepped forward, the room went quiet."},{"sceneIndex":0,"lineIndex":1,"character":"ENGINEER 1","text":"Potter, did she say?"}]}

Text:
${text}`;

    const [parseResult, voicesResult] = await Promise.all([
      genai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [{ role: "user", parts: [{ text: parsePrompt }] }],
      }),
      elClient.voices.getAll(),
    ]);

    const rawJson = parseResult.text ?? "";
    const cleaned = rawJson.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();

    let scenes: ScriptScene[];
    let dialogueLines: ScriptDialogueLine[];
    try {
      const parsed = JSON.parse(cleaned);
      scenes = parsed.scenes;
      dialogueLines = parsed.dialogueLines;
    } catch {
      return NextResponse.json({ error: "Gemini returned invalid JSON", raw: rawJson }, { status: 500 });
    }

    // ── Step 2: Voice casting ─────────────────────────────────────────────────
    const allVoices = (voicesResult.voices ?? []).filter(v => v.category === "premade");
    const voiceById = new Map(allVoices.map(v => [v.voice_id, v]));

    const voiceCatalogue = allVoices.map(v => {
      const l = v.labels ?? {};
      const desc = [l["description"], l["age"], l["accent"], l["gender"], l["use case"]].filter(Boolean).join(", ");
      return `${v.name} (${v.voice_id}): ${desc || "no description"}`;
    }).join("\n");

    const characters = [...new Set(dialogueLines.map(l => l.character).filter(c => c !== "narrator"))];

    const castingPrompt = `You are a casting director for a cinematic audio experience. Assign the most fitting ElevenLabs voice to each role based on the character's implied age, gender, accent, and personality from the screenplay.

Roles to cast:
- narrator (off-screen guide / scene announcer — prefer gravitas, clarity, and range)
${characters.map(c => `- ${c}`).join("\n")}

Rules:
- Never assign the same voice to two roles
- Match gender, age, and accent to what the screenplay implies about each character
- For expressive characters prefer voices described as "expressive" or "warm" — avoid flat voices for leads
- Return ONLY a valid JSON object, no markdown: {"narrator":"<voiceId>","<character>":"<voiceId>",...}

Available voices:
${voiceCatalogue}

Screenplay excerpt (for character context):
${text.slice(0, 1500)}`;

    let castingRaw: Record<string, string> = {};
    try {
      const castResult = await genai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [{ role: "user", parts: [{ text: castingPrompt }] }],
      });
      const castJson = (castResult.text ?? "").replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
      castingRaw = JSON.parse(castJson);
      console.log("[CAST] Gemini voice assignments:", JSON.stringify(castingRaw, null, 2));
    } catch (e) {
      console.error("[CAST] Voice casting failed, falling back to defaults:", e);
    }

    const FALLBACK_NARRATOR = "JBFqnCBsd6RMkjVDRZzb"; // George
    const FALLBACK_VOICES = [
      "21m00Tcm4TlvDq8ikWAM", // Rachel
      "ErXwobaYiN019PkySvjV", // Antoni
      "XrExE9yKIg1WjnnlVkGX", // Matilda
      "TxGEqnHWrfWFTfGW9XjX", // Josh
      "VR6AewLTigWG4xSOukaG", // Arnold
    ];

    const usedIds = new Set<string>();
    function resolveVoice(id: string | undefined, fallbackId: string): VoiceAssignment {
      const resolved = id && voiceById.has(id) ? id : fallbackId;
      const v = voiceById.get(resolved);
      const l = v?.labels ?? {};
      const desc = [l["description"], l["age"], l["accent"], l["gender"]].filter(Boolean).join(", ");
      return { voiceId: resolved, voiceName: v?.name ?? resolved, description: desc || "ElevenLabs voice" };
    }

    const voiceMap: Record<string, VoiceAssignment> = {};
    voiceMap["narrator"] = resolveVoice(castingRaw["narrator"], FALLBACK_NARRATOR);
    usedIds.add(voiceMap["narrator"].voiceId);

    let fallbackIdx = 0;
    for (const character of characters) {
      let fallback = FALLBACK_VOICES[fallbackIdx % FALLBACK_VOICES.length];
      while (usedIds.has(fallback)) { fallbackIdx++; fallback = FALLBACK_VOICES[fallbackIdx % FALLBACK_VOICES.length]; }
      const assigned = resolveVoice(castingRaw[character], fallback);
      if (usedIds.has(assigned.voiceId)) {
        while (usedIds.has(fallback)) { fallbackIdx++; fallback = FALLBACK_VOICES[fallbackIdx % FALLBACK_VOICES.length]; }
        voiceMap[character] = resolveVoice(fallback, fallback);
      } else {
        voiceMap[character] = assigned;
      }
      usedIds.add(voiceMap[character].voiceId);
      fallbackIdx++;
    }

    // ── Step 3: Three parallel Gemini calls ──────────────────────────────────

    // a. Storyboard image generation — one per scene, concurrency-limited to avoid rate limits
    const IMAGE_MODEL = "gemini-3-pro-image-preview";
    const imageTasks = scenes.map((scene) => async (): Promise<StoryboardFrame | null> => {
      const prompt =
        `Film storyboard panel. Hand-drawn rough black marker sketch on white paper. ` +
        `Bold thick lines, gestural style, black and white only, no color, ` +
        `like a professional director's storyboard. ` +
        `NO TEXT, NO WORDS, NO LETTERS, NO LABELS, NO CAPTIONS anywhere in the image. ` +
        `Scene: ${scene.heading}. ${scene.action.slice(0, 200)}`;

      for (let attempt = 0; attempt <= 2; attempt++) {
        if (attempt > 0) {
          const delay = attempt * 6000;
          console.warn(`[IMAGE] Scene ${scene.index}: retrying in ${delay / 1000}s (attempt ${attempt + 1})`);
          await new Promise(r => setTimeout(r, delay));
        }
        try {
          const imageResult = await genai.models.generateContent({
            model: IMAGE_MODEL,
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            config: { responseModalities: ["TEXT", "IMAGE"], imageConfig: { aspectRatio: "16:9" } },
          });
          const candidates = imageResult.candidates ?? [];
          console.log(`[IMAGE] Scene ${scene.index}: ${candidates.length} candidate(s)`);
          for (const cand of candidates) {
            for (const part of cand.content?.parts ?? []) {
              if (part.inlineData?.data) {
                console.log(`[IMAGE] Scene ${scene.index} OK: ${part.inlineData.mimeType}, ${part.inlineData.data.length} chars`);
                return { sceneIndex: scene.index, imageBase64: part.inlineData.data, mimeType: part.inlineData.mimeType ?? "image/png" };
              }
            }
          }
          console.warn(`[IMAGE] Scene ${scene.index}: no inlineData in response`);
          return null; // no image returned — don't retry
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          const isRateLimit = msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED") || msg.includes("quota");
          if (isRateLimit && attempt < 2) continue;
          console.error(`[IMAGE] Scene ${scene.index} failed (attempt ${attempt + 1}):`, e);
          return null;
        }
      }
      return null;
    });

    // b. SFX suggestions per scene
    const sfxGeminiPrompt = `You are a sound designer writing prompts for an AI sound effects generator (ElevenLabs). Your prompts must describe the ACOUSTIC CHARACTER of the sound — its scale, texture, and environment — not just label the action.

GOOD prompts (evocative, acoustic):
- "thunderous crowd applause filling a large hall"
- "roaring fireplace with crackling wood pops"
- "heavy oak door slamming shut with echo"
- "torrential rain hammering a tin roof"
- "espresso machine steaming and hissing in a cafe"
- "footsteps crunching on wet gravel"

BAD — return null for:
- Voices, speech, or any human vocal sound
- Silence or absence of sound
- Music or melodic elements
- Abstract emotions with no physical sound source

For each scene, write a 4-10 word evocative ambient sound prompt, or null if not applicable.
Reply ONLY with a valid JSON array, no markdown: [{"sceneIndex":<number>,"prompt":<string|null>}]

Scenes:
${scenes.map(s => `${s.index}: "${s.heading} — ${s.action.slice(0, 150)}"`).join("\n")}`;

    // c. Dialogue tone analysis
    const fullScript = dialogueLines.map(l =>
      `${l.lineIndex} [${l.character}${l.parenthetical ? ` (${l.parenthetical})` : ""}]: "${l.text.slice(0, 150)}"`
    ).join("\n");

    const toneGeminiPrompt = `You are a voice director for a cinematic audio experience. Given the full script below, suggest delivery settings for each dialogue line — capturing the character's emotional state, urgency, and tone.

Parameters:
- stability (0.0–1.0): LOW (0.05–0.25) for fear/panic/anger/excitement/grief. MEDIUM (0.3–0.5) for tension/urgency/unease. HIGH (0.55–0.75) for calm/controlled/cold/authoritative.
- style (0.0–1.0): higher (0.5–0.85) for dramatic/emotional, lower (0.1–0.35) for understated/deadpan.
- emotion: 2-4 word label (e.g. "hushed urgency", "cold warning", "barely suppressed panic", "weary resignation")

Return ONLY a JSON array: [{"lineIndex":<n>,"stability":<f>,"style":<f>,"emotion":"<str>"}]

Script:
${fullScript}`;

    // d. Ambiance music suggestions (Lyria prompts)
    const ambianceGeminiPrompt = `You are a film music supervisor deciding where to add non-diegetic ambient background music.

Lyria is a music generation AI that accepts short descriptive text prompts like:
"tense low strings and piano", "warm orchestral swell", "ominous dark ambient pads", "gentle acoustic guitar daylight"

Rules — add music (prompt != null) when:
- Establishing/opening shots that set mood
- Tense silences, dramatic reveals, emotional peaks
- Quiet monologue or reflection moments
- Scene transitions that need atmosphere

Rules — skip music (prompt = null) when:
- Scenes already heavy with SFX (loud ambient sounds)
- Fast-paced dialogue with lots of back-and-forth
- Comic or mundane moments where music would feel forced

For scenes that need music: write a 5–15 word prompt describing mood + instrumentation + tempo feel.
Return ONLY a valid JSON array, no markdown:
[{"sceneIndex":<number>,"prompt":<string|null>}]

Scenes:
${scenes.map(s => `${s.index}: "${s.heading}" — ${s.action.slice(0, 180)}`).join("\n")}`;

    const [imageResults, sfxRawResult, toneRawResult, ambianceRawResult] = await Promise.allSettled([
      runWithConcurrency(imageTasks, 3),
      genai.models.generateContent({ model: "gemini-3-flash-preview", contents: [{ role: "user", parts: [{ text: sfxGeminiPrompt }] }] }),
      genai.models.generateContent({ model: "gemini-3-flash-preview", contents: [{ role: "user", parts: [{ text: toneGeminiPrompt }] }] }),
      genai.models.generateContent({ model: "gemini-3-flash-preview", contents: [{ role: "user", parts: [{ text: ambianceGeminiPrompt }] }] }),
    ]);

    const storyboardFrames: StoryboardFrame[] = imageResults.status === "fulfilled"
      ? imageResults.value.filter((f): f is StoryboardFrame => f !== null && f !== undefined)
      : [];

    let sfxSuggestions: SfxSuggestion[] = [];
    if (sfxRawResult.status === "fulfilled") {
      try {
        const raw = (sfxRawResult.value.text ?? "").replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
        sfxSuggestions = JSON.parse(raw);
        console.log("[SFX] Gemini suggestions:", JSON.stringify(sfxSuggestions, null, 2));
      } catch (e) { console.error("[SFX] Gemini SFX parse failed:", e); }
    }

    let dialogueTones: DialogueTone[] = [];
    if (toneRawResult.status === "fulfilled") {
      try {
        const raw = (toneRawResult.value.text ?? "").replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
        dialogueTones = JSON.parse(raw);
        console.log("[TONE] Gemini dialogue tones:", JSON.stringify(dialogueTones, null, 2));
      } catch (e) { console.error("[TONE] Gemini tone parse failed:", e); }
    }

    let musicSuggestions: MusicSuggestion[] = [];
    if (ambianceRawResult.status === "fulfilled") {
      try {
        const raw = (ambianceRawResult.value.text ?? "").replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
        musicSuggestions = JSON.parse(raw);
        console.log("[LYRIA] Gemini music suggestions:", JSON.stringify(musicSuggestions, null, 2));
      } catch (e) { console.error("[LYRIA] Gemini ambiance parse failed:", e); }
    }

    const toneMap: Record<number, DialogueTone> = {};
    for (const t of dialogueTones) toneMap[t.lineIndex] = t;

    // ── Step 4: TTS per dialogue line (sequential) ───────────────────────────
    const defaultNarratorSettings = { stability: 0.65, similarity_boost: 0.8, style: 0.1 };
    const defaultDialogueSettings = { stability: 0.35, similarity_boost: 0.8, style: 0.6 };
    const linesAudio: string[] = [];
    const lineDurations: number[] = [];

    for (let i = 0; i < dialogueLines.length; i++) {
      const line = dialogueLines[i];
      const isNarrator = line.character === "narrator";
      const assignment = voiceMap[line.character] ?? voiceMap["narrator"];
      const tone = toneMap[i];
      const settings = isNarrator
        ? defaultNarratorSettings
        : tone
          ? { stability: tone.stability, similarity_boost: 0.8, style: tone.style }
          : defaultDialogueSettings;

      const toneLabel = tone ? ` [${tone.emotion}]` : "";
      console.log(`[TTS] line ${i} (${line.character}${toneLabel}) → voice "${assignment.voiceName}" | stability: ${settings.stability} style: ${settings.style} | text: "${line.text.slice(0, 80)}${line.text.length > 80 ? "…" : ""}"`);

      const tsResponse = await elClient.textToSpeech.convertWithTimestamps(assignment.voiceId, {
        text: line.text,
        model_id: "eleven_multilingual_v2",
        voice_settings: settings,
      });

      linesAudio.push(tsResponse.audio_base64);
      const endTimes = tsResponse.alignment?.character_end_times_seconds ?? [];
      lineDurations.push(endTimes.length > 0 ? endTimes[endTimes.length - 1] : 0);
    }

    // ── Step 5: Calculate sceneStartTimes ────────────────────────────────────
    const GAP_DLG_TO_DLG = 0.12;
    const GAP_BETWEEN_SCENES = 0.5;

    const lineStartTimes: number[] = [];
    let cursor = 0;
    for (let i = 0; i < dialogueLines.length; i++) {
      if (i > 0) {
        cursor += dialogueLines[i - 1].sceneIndex !== dialogueLines[i].sceneIndex
          ? GAP_BETWEEN_SCENES
          : GAP_DLG_TO_DLG;
      }
      lineStartTimes.push(cursor);
      cursor += lineDurations[i] ?? 0;
    }

    // Two-pass sceneStartTimes: anchor on scenes with dialogue, then interpolate
    // between anchors so visual-only sub-scenes don't bunch up at the start.
    const rawTimes: (number | null)[] = scenes.map((scene) => {
      const firstLine = dialogueLines.find(l => l.sceneIndex === scene.index);
      return firstLine ? lineStartTimes[firstLine.lineIndex] : null;
    });

    const sceneStartTimes: number[] = rawTimes.map((t, i) => {
      if (t !== null) return t;
      // Find surrounding anchors
      let prevIdx = i - 1;
      while (prevIdx >= 0 && rawTimes[prevIdx] === null) prevIdx--;
      let nextIdx = i + 1;
      while (nextIdx < rawTimes.length && rawTimes[nextIdx] === null) nextIdx++;
      const prevTime = prevIdx >= 0 ? (rawTimes[prevIdx] as number) : 0;
      const nextTime = nextIdx < rawTimes.length ? (rawTimes[nextIdx] as number) : cursor;
      // Linearly interpolate between the two nearest anchors
      const span = nextIdx - prevIdx;
      return prevTime + ((i - prevIdx) / span) * (nextTime - prevTime);
    });

    // ── Step 6a: Parallel Lyria ambiance music generation ────────────────────
    const musicList: MusicItem[] = [];
    const nonNullMusic = musicSuggestions.filter((s): s is MusicSuggestion & { prompt: string } => s.prompt !== null);
    if (nonNullMusic.length > 0) {
      const musicResults = await Promise.allSettled(nonNullMusic.map(async ({ sceneIndex, prompt }) => {
        const startTime = sceneStartTimes[sceneIndex] ?? 0;
        const nextStart = sceneIndex + 1 < sceneStartTimes.length ? sceneStartTimes[sceneIndex + 1] : cursor;
        const duration = Math.max(6, Math.min(nextStart - startTime + 2, 25));
        console.log(`[LYRIA] scene ${sceneIndex} | prompt: "${prompt}" | duration: ${duration.toFixed(1)}s`);
        const audioBase64 = await generateLyriaMusic(geminiKey, prompt, duration);
        if (!audioBase64) throw new Error("no audio returned");
        return { sceneIndex, prompt, startTime, audioBase64 } satisfies MusicItem;
      }));
      for (const r of musicResults) {
        if (r.status === "fulfilled") musicList.push(r.value);
        else console.error("[LYRIA] generation failed:", r.reason);
      }
    }

    // ── Step 6b: Parallel SFX generation ─────────────────────────────────────
    const sfxList: SfxItem[] = [];
    const nonNullSfx = sfxSuggestions.filter((s): s is SfxSuggestion & { prompt: string } => s.prompt !== null);
    if (nonNullSfx.length > 0) {
      const results = await Promise.allSettled(nonNullSfx.map(async ({ sceneIndex, prompt }) => {
        const startTime = sceneStartTimes[sceneIndex] ?? 0;
        console.log(`[SFX] generating → scene ${sceneIndex} | prompt: "${prompt}" | startTime: ${startTime.toFixed(2)}s`);
        const sfxStream = await elClient.textToSoundEffects.convert({ text: prompt, duration_seconds: 5, prompt_influence: 0.5 });
        const sfxBuffer = await streamToBuffer(sfxStream as AsyncIterable<Buffer>);
        console.log(`[SFX] done → scene ${sceneIndex} | bytes: ${sfxBuffer.length}`);
        return { sceneIndex, prompt, startTime, audioBase64: sfxBuffer.toString("base64") } satisfies SfxItem;
      }));
      for (const r of results) {
        if (r.status === "fulfilled") sfxList.push(r.value);
        else console.error("[SFX] generation failed:", r.reason);
      }
    }

    const responseVoiceMap: Record<string, { name: string; description: string }> = {};
    for (const [speaker, assignment] of Object.entries(voiceMap)) {
      responseVoiceMap[speaker] = { name: assignment.voiceName, description: assignment.description };
    }

    return NextResponse.json({ scenes, dialogueLines, storyboardFrames, voiceMap: responseVoiceMap, linesAudio, sfxList, musicList, dialogueTones, sceneStartTimes });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
