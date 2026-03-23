import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateFishSpeech } from './fishTts.mjs';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const outputPath = path.join(__dirname, 'fish-test.mp3');
const text = 'This is a minimal Fish smoke test from the QuinnOS backend.';

const referenceId = String(process.env.FISH_REFERENCE_ID || '').trim();
const audioBuffer = await generateFishSpeech({ text, format: 'mp3' });

if (!audioBuffer.length) {
  throw new Error('Fish returned an empty audio buffer.');
}

await fs.writeFile(outputPath, audioBuffer);

console.log(
  JSON.stringify(
    {
      ok: true,
      outputPath,
      bytes: audioBuffer.length,
      referenceId,
    },
    null,
    2
  )
);
