import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { ElevenLabsClient } from "elevenlabs";

interface Segment {
  type: "narration" | "dialogue";
  text: string;
  speaker?: string;
}

interface SegmentTiming { segmentIndex: number; startTime: number; endTime: number; }
interface SfxItem { segmentIndex: number; prompt: string; startTime: number; audioBase64: string; }
interface SfxSuggestion { segmentIndex: number; prompt: string | null; }
interface DialogueTone { segmentIndex: number; stability: number; style: number; emotion: string; }

interface VoiceAssignment {
  voiceId: string;
  voiceName: string;
  description: string;
}

async function streamToBuffer(stream: AsyncIterable<Buffer>): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
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

    // Step 1: Parse text + fetch all ElevenLabs voices in parallel
    const parsePrompt = `You are a literary text parser. Analyze the following book excerpt and split it into segments of narration and dialogue.

Rules:
- Quoted text (inside " " or ' ') spoken by a character is "dialogue". Identify the speaker name from the surrounding text.
- All other text (descriptions, action, attribution phrases) is "narration".
- Preserve the original text of each segment EXACTLY as it appears.
- Return ONLY a valid JSON array with no markdown fences, no explanation. Each element must have:
  - "type": "narration" or "dialogue"
  - "text": the exact text of this segment
  - "speaker": (only for dialogue) the character's name, or "unknown" if not identifiable

Text to parse:
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

    let segments: Segment[];
    try {
      segments = JSON.parse(cleaned);
    } catch {
      return NextResponse.json(
        { error: "Gemini returned invalid JSON", raw: rawJson },
        { status: 500 }
      );
    }

    // Build voice catalogue from ElevenLabs — premade voices only, with descriptions
    const allVoices = (voicesResult.voices ?? []).filter(v => v.category === "premade");
    const voiceById = new Map(allVoices.map(v => [v.voice_id, v]));

    const voiceCatalogue = allVoices.map(v => {
      const l = v.labels ?? {};
      const desc = [l["description"], l["age"], l["accent"], l["gender"], l["use case"]]
        .filter(Boolean).join(", ");
      return `${v.name} (${v.voice_id}): ${desc || "no description"}`;
    }).join("\n");

    // Step 2: Gemini casts voices for narrator + every character
    const speakers = [...new Set(
      segments.filter(s => s.type === "dialogue" && s.speaker).map(s => s.speaker!)
    )];

    const castingPrompt = `You are a casting director for an immersive audiobook. Assign the most fitting ElevenLabs voice to each role based on the character's implied age, gender, accent, and personality from the passage.

Roles to cast:
- narrator (omniscient storytelling voice — prefer gravitas, clarity, and range)
${speakers.map(s => `- ${s}`).join("\n")}

Rules:
- Never assign the same voice to two roles
- Match gender, age, and accent to what the narrative implies about each character
- For expressive characters prefer voices described as "expressive", "warm", or with character — avoid flat/neutral voices for leads
- Return ONLY a valid JSON object, no markdown: {"narrator":"<voiceId>","<speaker>":"<voiceId>",...}

Available voices:
${voiceCatalogue}

Passage (for character context):
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

    // Build voiceMap — validate each assigned ID exists, fall back to a sensible default
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
      return {
        voiceId: resolved,
        voiceName: v?.name ?? resolved,
        description: desc || "ElevenLabs voice",
      };
    }

    const voiceMap: Record<string, VoiceAssignment> = {};
    voiceMap["narrator"] = resolveVoice(castingRaw["narrator"], FALLBACK_NARRATOR);
    usedIds.add(voiceMap["narrator"].voiceId);

    let fallbackIdx = 0;
    for (const speaker of speakers) {
      let fallback = FALLBACK_VOICES[fallbackIdx % FALLBACK_VOICES.length];
      while (usedIds.has(fallback)) fallbackIdx++;
      fallback = FALLBACK_VOICES[fallbackIdx % FALLBACK_VOICES.length];

      const assigned = resolveVoice(castingRaw[speaker], fallback);
      // If Gemini picked a voice already used, pick the next fallback
      if (usedIds.has(assigned.voiceId)) {
        while (usedIds.has(fallback)) { fallbackIdx++; fallback = FALLBACK_VOICES[fallbackIdx % FALLBACK_VOICES.length]; }
        voiceMap[speaker] = resolveVoice(fallback, fallback);
      } else {
        voiceMap[speaker] = assigned;
      }
      usedIds.add(voiceMap[speaker].voiceId);
      fallbackIdx++;
    }

    // Step 3: SFX suggestions + dialogue tone analysis in parallel
    const narrationSegs = segments.map((seg, idx) => ({ seg, idx })).filter(({ seg }) => seg.type === "narration");
    const dialogueSegs = segments.map((seg, idx) => ({ seg, idx })).filter(({ seg }) => seg.type === "dialogue");

    const fullPassage = segments.map((seg, idx) =>
      seg.type === "narration"
        ? `${idx} [narration]: "${seg.text.slice(0, 150)}"`
        : `${idx} [dialogue - ${seg.speaker ?? "unknown"}]: "${seg.text.slice(0, 150)}"`
    ).join("\n");

    const sfxGeminiPrompt = `You are a sound designer writing prompts for an AI sound effects generator (ElevenLabs). Your prompts must describe the ACOUSTIC CHARACTER of the sound — its scale, texture, and environment — not just label the action. The generator responds well to vivid, sensory descriptions.

GOOD prompts (evocative, acoustic):
- "thunderous crowd applause filling a large hall" (not "clapping hands")
- "roaring fireplace with crackling wood pops" (not "fire burning")
- "heavy oak door slamming shut with echo" (not "door closing")
- "torrential rain hammering a tin roof" (not "rain outside")
- "footsteps crunching on gravel path" (not "someone walking")
- "teacup clinking against china saucer" (not "cup sound")

BAD — return null for:
- Voices, speech, or any human vocal sound
- Silence or absence of sound
- Music or melodic elements
- Abstract emotions with no physical sound source

For each narration segment, write a 4-10 word evocative sound prompt if a physical sound is present, or null if not. Be specific about scale and environment.
Reply ONLY with a valid JSON array, no markdown. Format: [{"segmentIndex":<number>,"prompt":<string|null>}]

Segments:
${narrationSegs.map(({ seg, idx }) => `${idx}: "${seg.text.slice(0, 150)}"`).join("\n")}`;

    const toneGeminiPrompt = `You are a voice director for an immersive audiobook. Given the full passage below, suggest delivery settings for each DIALOGUE segment — capturing the character's emotional state, urgency, and tone based on the surrounding narrative context.

Parameters:
- stability (0.0–1.0): controls consistency vs expressiveness. Use LOW (0.05–0.25) for fear, panic, anger, excitement, grief. Use MEDIUM (0.3–0.5) for tension, urgency, unease. Use HIGH (0.55–0.75) for calm, controlled, cold, or authoritative delivery.
- style (0.0–1.0): style exaggeration. Use higher (0.5–0.85) for dramatic/emotional lines, lower (0.1–0.35) for understated or deadpan delivery.
- emotion: a 2-4 word label for how the line should sound (e.g. "hushed urgency", "cold warning", "barely suppressed panic", "weary resignation")

Return ONLY a JSON array for DIALOGUE segments: [{"segmentIndex":<n>,"stability":<f>,"style":<f>,"emotion":"<str>"}]

Full passage:
${fullPassage}`;

    let sfxSuggestions: SfxSuggestion[] = [];
    let dialogueTones: DialogueTone[] = [];

    const [sfxRawResult, toneRawResult] = await Promise.allSettled([
      narrationSegs.length > 0
        ? genai.models.generateContent({ model: "gemini-2.5-flash", contents: [{ role: "user", parts: [{ text: sfxGeminiPrompt }] }] })
        : Promise.resolve(null),
      dialogueSegs.length > 0
        ? genai.models.generateContent({ model: "gemini-2.5-flash", contents: [{ role: "user", parts: [{ text: toneGeminiPrompt }] }] })
        : Promise.resolve(null),
    ]);

    if (sfxRawResult.status === "fulfilled" && sfxRawResult.value) {
      try {
        const raw = (sfxRawResult.value.text ?? "").replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
        sfxSuggestions = JSON.parse(raw);
        console.log("[SFX] Gemini suggestions:", JSON.stringify(sfxSuggestions, null, 2));
      } catch (e) { console.error("[SFX] Gemini SFX parse failed:", e); }
    }

    if (toneRawResult.status === "fulfilled" && toneRawResult.value) {
      try {
        const raw = (toneRawResult.value.text ?? "").replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
        dialogueTones = JSON.parse(raw);
        console.log("[TONE] Gemini dialogue tones:", JSON.stringify(dialogueTones, null, 2));
      } catch (e) { console.error("[TONE] Gemini tone parse failed:", e); }
    }

    const toneMap: Record<number, DialogueTone> = {};
    for (const t of dialogueTones) toneMap[t.segmentIndex] = t;

    // Step 4: Generate audio per segment
    const narrationSettings = { stability: 0.65, similarity_boost: 0.8, style: 0.1 };
    const defaultDialogueSettings = { stability: 0.35, similarity_boost: 0.8, style: 0.6 };

    const audioBuffers: Buffer[] = [];
    const segmentTimings: SegmentTiming[] = [];
    let timeOffset = 0;

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      const speakerKey = seg.type === "narration" ? "narrator" : (seg.speaker ?? "unknown");
      const assignment = voiceMap[speakerKey] ?? voiceMap["narrator"];

      let settings: { stability: number; similarity_boost: number; style: number };
      if (seg.type === "narration") {
        settings = narrationSettings;
      } else {
        const tone = toneMap[i];
        settings = tone
          ? { stability: tone.stability, similarity_boost: 0.8, style: tone.style }
          : defaultDialogueSettings;
      }

      const toneLabel = toneMap[i] ? ` [${toneMap[i].emotion}]` : "";
      console.log(`[TTS] segment ${i} (${seg.type}${toneLabel}) → voice "${assignment.voiceName}" (${assignment.voiceId}) | stability: ${settings.stability} style: ${settings.style} | text: "${seg.text.slice(0, 80)}${seg.text.length > 80 ? "…" : ""}"`);

      const tsResponse = await elClient.textToSpeech.convertWithTimestamps(assignment.voiceId, {
        text: seg.text, model_id: "eleven_multilingual_v2", voice_settings: settings,
      });

      audioBuffers.push(Buffer.from(tsResponse.audio_base64, "base64"));
      const endTimes = tsResponse.alignment?.character_end_times_seconds ?? [];
      const segDuration = endTimes.length > 0 ? endTimes[endTimes.length - 1] : 0;
      segmentTimings.push({ segmentIndex: i, startTime: timeOffset, endTime: timeOffset + segDuration });
      timeOffset += segDuration;
    }

    const segmentsAudio = audioBuffers.map(b => b.toString("base64"));

    // Step 5: Parallel SFX generation
    const sfxList: SfxItem[] = [];
    const nonNullSfx = sfxSuggestions.filter((s): s is SfxSuggestion & { prompt: string } => s.prompt !== null);
    if (nonNullSfx.length > 0) {
      const results = await Promise.allSettled(nonNullSfx.map(async ({ segmentIndex, prompt }) => {
        console.log(`[SFX] generating → segment ${segmentIndex} | prompt: "${prompt}" | duration: 3s | prompt_influence: 0.3`);
        const sfxStream = await elClient.textToSoundEffects.convert({ text: prompt, duration_seconds: 5, prompt_influence: 0.5 });
        const sfxBuffer = await streamToBuffer(sfxStream as AsyncIterable<Buffer>);
        const timing = segmentTimings.find(t => t.segmentIndex === segmentIndex)!;
        console.log(`[SFX] done → segment ${segmentIndex} | startTime: ${timing.startTime.toFixed(2)}s | bytes: ${sfxBuffer.length}`);
        return { segmentIndex, prompt, startTime: timing.startTime, audioBase64: sfxBuffer.toString("base64") } satisfies SfxItem;
      }));
      for (const r of results) {
        if (r.status === "fulfilled") sfxList.push(r.value);
        else console.error("[SFX] generation failed:", r.reason);
      }
    }

    // Build clean voice map for response
    const responseVoiceMap: Record<string, { name: string; description: string }> = {};
    for (const [speaker, assignment] of Object.entries(voiceMap)) {
      responseVoiceMap[speaker] = { name: assignment.voiceName, description: assignment.description };
    }

    return NextResponse.json({ segments, voiceMap: responseVoiceMap, segmentsAudio, segmentTimings, sfxList, dialogueTones });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
