import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateElevenSpeech } from './elevenTts.mjs';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const outputPath = path.join(__dirname, 'eleven-test.mp3');
const text = 'This is a minimal ElevenLabs smoke test from the QuinnOS backend.';

const voiceId = String(process.env.ELEVENLABS_VOICE_ID || '').trim();
const modelId = String(process.env.ELEVENLABS_MODEL_ID || '').trim();
const audioBuffer = await generateElevenSpeech({ text, format: 'mp3' });

if (!audioBuffer.length) {
  throw new Error('ElevenLabs returned an empty audio buffer.');
}

await fs.writeFile(outputPath, audioBuffer);

console.log(
  JSON.stringify(
    {
      ok: true,
      outputPath,
      bytes: audioBuffer.length,
      voiceId,
      modelId,
    },
    null,
    2
  )
);
