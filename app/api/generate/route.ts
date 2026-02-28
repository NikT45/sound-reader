import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI, type LiveMusicSession } from "@google/genai";
import { ElevenLabsClient } from "elevenlabs";

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

// Collect `durationSeconds` of Lyria PCM audio and return as WAV base64
async function generateLyriaMusic(
  genai: GoogleGenAI,
  musicPrompt: string,
  durationSeconds: number
): Promise<string | null> {
  const chunks: Buffer[] = [];
  let totalBytes = 0;
  // PCM16 stereo 48 kHz → 4 bytes per frame
  const targetBytes = Math.ceil(Math.min(durationSeconds, 25) * 48000 * 4);

  return new Promise<string | null>((resolve) => {
    let session: LiveMusicSession;
    let finished = false;

    const finish = () => {
      if (finished) return;
      finished = true;
      try { session?.close(); } catch {}
      if (!chunks.length) { resolve(null); return; }
      const pcm = Buffer.concat(chunks);
      resolve(pcmToWav(pcm, 2, 48000).toString("base64"));
    };

    const timer = setTimeout(finish, 40_000);

    genai.live.music.connect({
      model: "models/lyria-realtime-exp",
      callbacks: {
        onmessage: async (msg) => {
          if (msg.setupComplete) {
            try {
              await session.setWeightedPrompts({
                weightedPrompts: [{ text: musicPrompt, weight: 1.0 }],
              });
              session.play();
            } catch (e) {
              console.error("[LYRIA] setWeightedPrompts failed:", e);
              clearTimeout(timer); finish();
            }
          }
          for (const chunk of msg.serverContent?.audioChunks ?? []) {
            if (chunk.data && !finished) {
              const buf = Buffer.from(chunk.data, "base64");
              chunks.push(buf);
              totalBytes += buf.length;
              if (totalBytes >= targetBytes) { clearTimeout(timer); finish(); }
            }
          }
        },
        onerror: (e) => { console.error("[LYRIA] ws error:", e); clearTimeout(timer); finish(); },
        onclose: () => { clearTimeout(timer); finish(); },
      },
    }).then((s) => { session = s; }).catch((e) => {
      console.error("[LYRIA] connect failed:", e);
      clearTimeout(timer); resolve(null);
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
    const parsePrompt = `You are a script and prose parser. Parse the following text into scenes and dialogue lines. The input may be a formatted screenplay OR prose narrative — handle both.

SCREENPLAY FORMAT (has INT./EXT. headings and ALL-CAPS character names):
- Lines starting with INT., EXT., I., E., or I/E. → scene headings
- ALL-CAPS standalone line (not a heading) → character name; following indented lines → dialogue
- Text between headings that is not a character name or dialogue → action text for that scene
- Parenthetical line in () between character name and dialogue → optional parenthetical field

PROSE FORMAT (no INT./EXT. headings — narrative paragraphs with quoted dialogue):
- Divide into scenes by logical setting changes, time jumps, or major paragraph breaks. If the whole passage is one location, it's one scene.
- Give each scene a SHORT descriptive heading in screenplay style (e.g. "INT. SERVER ROOM - DAY", "EXT. STREET - NIGHT"). Infer location and time from the text.
- Any text inside quotation marks (' ' or " ") spoken by a character is dialogue. Extract the quoted text only (no attribution phrase).
- Identify the speaker from the surrounding attribution text (e.g. "said Harry", "the voice whispered"). Use a SHORT character name in ALL-CAPS (e.g. HARRY, VOICE, SORTING HAT). Use "UNKNOWN" only if truly unidentifiable.
- All non-quoted text (description, attribution, action) is the scene's action text.

ALWAYS:
- Assign each scene a sequential index starting at 0.
- Assign each dialogue line a global lineIndex starting at 0 (incrementing across all scenes in order).
- Return ONLY a valid JSON object with no markdown fences, no explanation:
{"scenes":[{"index":0,"heading":"INT. SERVER ROOM - DAY","action":"Description text here."}],"dialogueLines":[{"sceneIndex":0,"lineIndex":0,"character":"HARRY","parenthetical":"quietly","text":"Not the Legacy Codebase."}]}

Text:
${text}`;

    const [parseResult, voicesResult] = await Promise.all([
      genai.models.generateContent({
        model: "gemini-2.5-flash",
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

    const characters = [...new Set(dialogueLines.map(l => l.character))];

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
        model: "gemini-2.5-flash",
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
    const IMAGE_MODEL = "gemini-2.5-flash-image";
    const imageTasks = scenes.map((scene) => async (): Promise<StoryboardFrame | null> => {
      const prompt =
        `Film storyboard panel. Hand-drawn rough black marker sketch on white paper. ` +
        `Bold thick lines, gestural style, black and white only, no color, ` +
        `like a professional director's storyboard. ` +
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
      genai.models.generateContent({ model: "gemini-2.5-flash", contents: [{ role: "user", parts: [{ text: sfxGeminiPrompt }] }] }),
      genai.models.generateContent({ model: "gemini-2.5-flash", contents: [{ role: "user", parts: [{ text: toneGeminiPrompt }] }] }),
      genai.models.generateContent({ model: "gemini-2.5-flash", contents: [{ role: "user", parts: [{ text: ambianceGeminiPrompt }] }] }),
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
    const defaultDialogueSettings = { stability: 0.35, similarity_boost: 0.8, style: 0.6 };
    const linesAudio: string[] = [];
    const lineDurations: number[] = [];

    for (let i = 0; i < dialogueLines.length; i++) {
      const line = dialogueLines[i];
      const assignment = voiceMap[line.character] ?? voiceMap["narrator"];
      const tone = toneMap[i];
      const settings = tone
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

    const sceneStartTimes: number[] = [];
    for (const scene of scenes) {
      const firstLine = dialogueLines.find(l => l.sceneIndex === scene.index);
      if (firstLine) {
        sceneStartTimes.push(lineStartTimes[firstLine.lineIndex]);
      } else {
        const prevSceneLastLine = [...dialogueLines].reverse().find(l => l.sceneIndex === scene.index - 1);
        const prevEnd = prevSceneLastLine
          ? lineStartTimes[prevSceneLastLine.lineIndex] + (lineDurations[prevSceneLastLine.lineIndex] ?? 0)
          : sceneStartTimes[sceneStartTimes.length - 1] ?? 0;
        sceneStartTimes.push(prevEnd + 1.5);
      }
    }

    // ── Step 6a: Parallel Lyria ambiance music generation ────────────────────
    const musicList: MusicItem[] = [];
    const nonNullMusic = musicSuggestions.filter((s): s is MusicSuggestion & { prompt: string } => s.prompt !== null);
    if (nonNullMusic.length > 0) {
      const musicResults = await Promise.allSettled(nonNullMusic.map(async ({ sceneIndex, prompt }) => {
        const startTime = sceneStartTimes[sceneIndex] ?? 0;
        const nextStart = sceneIndex + 1 < sceneStartTimes.length ? sceneStartTimes[sceneIndex + 1] : cursor;
        const duration = Math.max(6, Math.min(nextStart - startTime + 2, 25));
        console.log(`[LYRIA] scene ${sceneIndex} | prompt: "${prompt}" | duration: ${duration.toFixed(1)}s`);
        const audioBase64 = await generateLyriaMusic(genai, prompt, duration);
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
