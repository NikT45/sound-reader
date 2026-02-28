import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { ElevenLabsClient } from "elevenlabs";

// Voice pool: narrator + up to 5 character slots + overflow
const VOICE_POOL: Record<string, { id: string; name: string }> = {
  narrator: { id: "JBFqnCBsd6RMkjVDRZzb", name: "George" },
  slot0: { id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel" },
  slot1: { id: "ErXwobaYiN019PkySvjV", name: "Antoni" },
  slot2: { id: "MF3mGyEYCl7XYWbV9V6O", name: "Elli" },
  slot3: { id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh" },
  slot4: { id: "VR6AewLTigWG4xSOukaG", name: "Arnold" },
  overflow: { id: "AZnzlk1XvdvUeBnXmlld", name: "Domi" },
};

interface Segment {
  type: "narration" | "dialogue";
  text: string;
  speaker?: string;
}

interface VoiceAssignment {
  voiceId: string;
  voiceName: string;
}

function buildVoiceMap(segments: Segment[]): Record<string, VoiceAssignment> {
  const voiceMap: Record<string, VoiceAssignment> = {
    narrator: { voiceId: VOICE_POOL.narrator.id, voiceName: VOICE_POOL.narrator.name },
  };
  let slotIndex = 0;

  for (const seg of segments) {
    if (seg.type === "dialogue" && seg.speaker && !voiceMap[seg.speaker]) {
      const slotKey = slotIndex < 5 ? `slot${slotIndex}` : "overflow";
      voiceMap[seg.speaker] = {
        voiceId: VOICE_POOL[slotKey].id,
        voiceName: VOICE_POOL[slotKey].name,
      };
      if (slotIndex < 5) slotIndex++;
    }
  }

  return voiceMap;
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

    // Step 1: Parse text with Gemini
    const genai = new GoogleGenAI({ apiKey: geminiKey });
    const geminiPrompt = `You are a literary text parser. Analyze the following book excerpt and split it into segments of narration and dialogue.

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

    const result = await genai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [{ role: "user", parts: [{ text: geminiPrompt }] }],
    });

    const rawJson = result.text ?? "";
    // Strip possible markdown fences
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

    // Step 2: Build voice map
    const voiceMap = buildVoiceMap(segments);

    // Step 3: Generate audio per segment
    const elClient = new ElevenLabsClient({ apiKey: elKey });
    const narrationSettings = { stability: 0.65, similarity_boost: 0.8, style: 0.1 };
    const dialogueSettings = { stability: 0.35, similarity_boost: 0.8, style: 0.6 };

    const audioBuffers: Buffer[] = [];

    for (const seg of segments) {
      const speakerKey = seg.type === "narration" ? "narrator" : (seg.speaker ?? "unknown");
      const assignment = voiceMap[speakerKey] ?? voiceMap["narrator"];
      const settings = seg.type === "narration" ? narrationSettings : dialogueSettings;

      const audioStream = await elClient.textToSpeech.convert(assignment.voiceId, {
        text: seg.text,
        model_id: "eleven_multilingual_v2",
        voice_settings: settings,
      });

      const buffer = await streamToBuffer(audioStream as AsyncIterable<Buffer>);
      audioBuffers.push(buffer);
    }

    const combinedAudio = Buffer.concat(audioBuffers);
    const audioBase64 = combinedAudio.toString("base64");

    // Build clean voice map for response (speaker -> voice name)
    const responseVoiceMap: Record<string, string> = {};
    for (const [speaker, assignment] of Object.entries(voiceMap)) {
      responseVoiceMap[speaker] = assignment.voiceName;
    }

    return NextResponse.json({ segments, voiceMap: responseVoiceMap, audioBase64 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
