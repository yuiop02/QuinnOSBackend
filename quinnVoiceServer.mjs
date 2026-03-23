import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { generateElevenSpeech } from './elevenTts.mjs';

dotenv.config();

const app = express();
const port = Number(process.env.PORT || process.env.FISH_VOICE_PORT || process.env.QUINN_VOICE_PORT || 8788);
const host = process.env.HOST || '0.0.0.0';
const SPEECH_CACHE_TTL_MS = Number(process.env.VOICE_CACHE_TTL_MS || 10 * 60 * 1000);
const speechCache = new Map();
const inFlightSpeech = new Map();

function getSpeechCacheKey(text, format = 'mp3') {
  return `${format}::${String(text || '').replace(/\s+/g, ' ').trim()}`;
}

function getCachedSpeech(text, format = 'mp3') {
  const key = getSpeechCacheKey(text, format);
  const entry = speechCache.get(key);

  if (!entry) {
    return null;
  }

  if (Date.now() - entry.createdAt > SPEECH_CACHE_TTL_MS) {
    speechCache.delete(key);
    return null;
  }

  return entry.audio;
}

function setCachedSpeech(text, audio, format = 'mp3') {
  const key = getSpeechCacheKey(text, format);
  speechCache.set(key, {
    audio,
    createdAt: Date.now(),
  });
}

async function getOrGenerateSpeech({ text, format = 'mp3' }) {
  const key = getSpeechCacheKey(text, format);
  const cached = getCachedSpeech(text, format);

  if (cached) {
    console.log('[VOICE CACHE HIT]', format, 'chars:', String(text || '').length);
    return cached;
  }

  const pending = inFlightSpeech.get(key);

  if (pending) {
    console.log('[VOICE IN-FLIGHT HIT]', format, 'chars:', String(text || '').length);
    return pending;
  }

  console.log('[VOICE CACHE MISS]', format, 'chars:', String(text || '').length);

  const generationPromise = (async () => {
    const audio = await generateElevenSpeech({
      text,
      format,
    });

    setCachedSpeech(text, audio, format);
    return audio;
  })();

  inFlightSpeech.set(key, generationPromise);

  try {
    return await generationPromise;
  } finally {
    inFlightSpeech.delete(key);
  }
}

app.use(cors());
app.use(express.json({ limit: '1mb' }));

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'quinn-eleven-voice',
    provider: 'elevenlabs',
    hasElevenApiKey: Boolean(process.env.ELEVENLABS_API_KEY),
    hasVoiceId: Boolean(process.env.ELEVENLABS_VOICE_ID),
    hasModelId: Boolean(process.env.ELEVENLABS_MODEL_ID),
    cacheEntries: speechCache.size,
    inFlightRequests: inFlightSpeech.size,
    cacheTtlMs: SPEECH_CACHE_TTL_MS,
    port,
  });
});

app.get('/speak', async (req, res) => {
  try {
    const textToSpeak = String(req.query?.text || '')
      .replace(/\s+/g, ' ')
      .trim();

    console.log('[VOICE GET /speak] chars:', textToSpeak.length);
    console.log('[VOICE GET /speak] text:', textToSpeak);

    if (!textToSpeak) {
      return res.status(400).json({
        ok: false,
        error: 'text is required',
      });
    }

    const audio = await getOrGenerateSpeech({
      text: textToSpeak,
      format: 'mp3',
    });

    console.log('[VOICE GET /speak] audio bytes:', audio.length);

    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', String(audio.length));
    res.setHeader('Content-Disposition', 'inline; filename="quinn.mp3"');
    res.setHeader('Cache-Control', 'public, max-age=600, immutable');
    return res.status(200).send(audio);
  } catch (error) {
    console.error('VOICE SERVER ERROR:', error);
    return res.status(500).json({
      ok: false,
      error: error?.message || 'Voice generation failed',
    });
  }
});

app.post('/speak', async (req, res) => {
  try {
    const textToSpeak = String(req.body?.text || '')
      .replace(/\s+/g, ' ')
      .trim();

    console.log('[VOICE POST /speak] chars:', textToSpeak.length);
    console.log('[VOICE POST /speak] text:', textToSpeak);

    if (!textToSpeak) {
      return res.status(400).json({
        ok: false,
        error: 'text is required',
      });
    }

    const audio = await getOrGenerateSpeech({
      text: textToSpeak,
      format: 'mp3',
    });

    console.log('[VOICE POST /speak] audio bytes:', audio.length);

    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', String(audio.length));
    res.setHeader('Content-Disposition', 'inline; filename="quinn.mp3"');
    res.setHeader('Cache-Control', 'public, max-age=600, immutable');
    return res.status(200).send(audio);
  } catch (error) {
    console.error('VOICE SERVER ERROR:', error);
    return res.status(500).json({
      ok: false,
      error: error?.message || 'Voice generation failed',
    });
  }
});

app.listen(port, host, () => {
  console.log(`Quinn voice server running on http://${host}:${port}`);
});
