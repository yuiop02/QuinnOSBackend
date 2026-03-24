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

function normalizeVoiceText(value, maxLength = 0) {
  const clean = String(value || '').replace(/\s+/g, ' ').trim();

  if (!clean) {
    return '';
  }

  if (!maxLength || clean.length <= maxLength) {
    return clean;
  }

  return `${clean.slice(0, maxLength - 3).trim()}...`;
}

function getSpeechCacheKey(
  text,
  format = 'mp3',
  { previousText = '', nextText = '' } = {}
) {
  return [
    format,
    normalizeVoiceText(text),
    normalizeVoiceText(previousText, 320),
    normalizeVoiceText(nextText, 320),
  ].join('::');
}

function getCachedSpeech(text, format = 'mp3', options = {}) {
  const key = getSpeechCacheKey(text, format, options);
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

function setCachedSpeech(text, audio, format = 'mp3', options = {}) {
  const key = getSpeechCacheKey(text, format, options);
  speechCache.set(key, {
    audio,
    createdAt: Date.now(),
  });
}

async function getOrGenerateSpeech({
  text,
  format = 'mp3',
  previousText = '',
  nextText = '',
}) {
  const normalizedOptions = {
    previousText: normalizeVoiceText(previousText, 320),
    nextText: normalizeVoiceText(nextText, 320),
  };
  const key = getSpeechCacheKey(text, format, normalizedOptions);
  const cached = getCachedSpeech(text, format, normalizedOptions);

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
      previousText: normalizedOptions.previousText,
      nextText: normalizedOptions.nextText,
    });

    setCachedSpeech(text, audio, format, normalizedOptions);
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
    const textToSpeak = normalizeVoiceText(req.query?.text);
    const previousText = normalizeVoiceText(
      req.query?.previous_text || req.query?.previousText,
      320
    );
    const nextText = normalizeVoiceText(req.query?.next_text || req.query?.nextText, 320);

    console.log('[VOICE GET /speak] chars:', textToSpeak.length);
    console.log('[VOICE GET /speak] text:', textToSpeak);
    console.log(
      '[VOICE GET /speak] continuity:',
      `prev=${previousText.length} chars`,
      `next=${nextText.length} chars`
    );

    if (!textToSpeak) {
      return res.status(400).json({
        ok: false,
        error: 'text is required',
      });
    }

    const audio = await getOrGenerateSpeech({
      text: textToSpeak,
      format: 'mp3',
      previousText,
      nextText,
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
    const textToSpeak = normalizeVoiceText(req.body?.text);
    const previousText = normalizeVoiceText(
      req.body?.previous_text || req.body?.previousText,
      320
    );
    const nextText = normalizeVoiceText(req.body?.next_text || req.body?.nextText, 320);

    console.log('[VOICE POST /speak] chars:', textToSpeak.length);
    console.log('[VOICE POST /speak] text:', textToSpeak);
    console.log(
      '[VOICE POST /speak] continuity:',
      `prev=${previousText.length} chars`,
      `next=${nextText.length} chars`
    );

    if (!textToSpeak) {
      return res.status(400).json({
        ok: false,
        error: 'text is required',
      });
    }

    const audio = await getOrGenerateSpeech({
      text: textToSpeak,
      format: 'mp3',
      previousText,
      nextText,
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
