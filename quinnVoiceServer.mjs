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

const VOICE_PROSODY_PROFILES = new Set([
  'neutralBalanced',
  'heldSoft',
  'tightFirm',
  'lightCurl',
  'magnetized',
  'settledWarm',
]);
const VOICE_PROSODY_PACE = new Set(['held', 'balanced', 'quick']);
const VOICE_PROSODY_LANDING = new Set(['soft', 'balanced', 'firm']);
const VOICE_PROSODY_SMOOTHNESS = new Set(['smooth', 'balanced', 'crisp']);
const VOICE_PROSODY_CONTOUR = new Set(['settled', 'lightLift', 'alive']);

function normalizeVoiceProsodySpeed(value) {
  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return null;
  }

  return Math.min(1.08, Math.max(0.98, Math.round(numeric * 100) / 100));
}

function normalizeVoiceProsodyHint(value) {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const profile = VOICE_PROSODY_PROFILES.has(String(value.profile))
    ? String(value.profile)
    : '';
  const pace = VOICE_PROSODY_PACE.has(String(value.pace)) ? String(value.pace) : '';
  const landing = VOICE_PROSODY_LANDING.has(String(value.landing))
    ? String(value.landing)
    : '';
  const smoothness = VOICE_PROSODY_SMOOTHNESS.has(String(value.smoothness))
    ? String(value.smoothness)
    : '';
  const contour = VOICE_PROSODY_CONTOUR.has(String(value.contour))
    ? String(value.contour)
    : '';
  const speed = normalizeVoiceProsodySpeed(value.speed);

  if (!profile || !pace || !landing || !smoothness || !contour || speed === null) {
    return null;
  }

  return {
    profile,
    speed,
    pace,
    landing,
    smoothness,
    contour,
  };
}

function readVoiceProsodyHintFromQuery(query) {
  return normalizeVoiceProsodyHint({
    profile: query?.voice_profile || query?.voiceProfile,
    speed: query?.voice_speed || query?.voiceSpeed,
    pace: query?.voice_pace || query?.voicePace,
    landing: query?.voice_landing || query?.voiceLanding,
    smoothness: query?.voice_smoothness || query?.voiceSmoothness,
    contour: query?.voice_contour || query?.voiceContour,
  });
}

function buildVoiceProsodyCacheKey(prosodyHint) {
  const normalized = normalizeVoiceProsodyHint(prosodyHint);

  if (!normalized) {
    return '';
  }

  return [
    normalized.profile,
    normalized.speed,
    normalized.pace,
    normalized.landing,
    normalized.smoothness,
    normalized.contour,
  ].join('|');
}

function getSpeechCacheKey(
  text,
  format = 'mp3',
  { previousText = '', nextText = '', prosodyHint = null } = {}
) {
  return [
    format,
    normalizeVoiceText(text),
    normalizeVoiceText(previousText, 320),
    normalizeVoiceText(nextText, 320),
    buildVoiceProsodyCacheKey(prosodyHint),
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
  prosodyHint = null,
}) {
  const normalizedOptions = {
    previousText: normalizeVoiceText(previousText, 320),
    nextText: normalizeVoiceText(nextText, 320),
    prosodyHint: normalizeVoiceProsodyHint(prosodyHint),
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
    service: 'quinn-fish-voice',
    provider: 'fish',
    hasFishApiKey: Boolean(process.env.FISH_API_KEY),
    hasReferenceId: Boolean(process.env.FISH_REFERENCE_ID),
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
    const prosodyHint = readVoiceProsodyHintFromQuery(req.query);

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
      prosodyHint,
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
    const prosodyHint = normalizeVoiceProsodyHint(
      req.body?.prosody_hint || req.body?.prosodyHint
    );

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
      prosodyHint,
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


app.get('/voice-info', async (_req, res) => {
  const voiceId = String(process.env.ELEVENLABS_VOICE_ID || '').trim();

  res.json({
    ok: true,
    service: 'quinn-voice',
    provider: 'elevenlabs',
    modelId: String(process.env.ELEVENLABS_MODEL_ID || '').trim() || null,
    hasElevenLabsApiKey: Boolean(process.env.ELEVENLABS_API_KEY),
    hasElevenLabsVoiceId: Boolean(voiceId),
    voiceIdLast4: voiceId ? voiceId.slice(-4) : null,
    cacheTtlMs: SPEECH_CACHE_TTL_MS,
  });
});

app.listen(port, host, () => {
  console.log(`Quinn voice server running on http://${host}:${port}`);
});
