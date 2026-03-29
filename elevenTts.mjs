const ELEVEN_TTS_BASE_URL = 'https://api.elevenlabs.io/v1/text-to-speech';
const QUINN_OUTPUT_FORMAT = 'mp3_44100_128';
// Keep Quinn's existing custom voice identity, but bias the request toward a
// lighter, softer, brighter delivery without pushing it into an unnatural
// rendered register.
const QUINN_EXPRESSIVE_VOICE_SETTINGS = Object.freeze({
  stability: 0.54,
  similarity_boost: 0.63,
  style: 0,
  speed: 1.04,
  use_speaker_boost: false,
});

function normalizeProsodySpeed(value) {
  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return null;
  }

  return Math.min(1.08, Math.max(0.98, Math.round(numeric * 100) / 100));
}

function buildQuinnVoiceSettings(prosodyHint) {
  const hintedSpeed = normalizeProsodySpeed(prosodyHint?.speed);

  if (
    hintedSpeed === null ||
    Math.abs(hintedSpeed - QUINN_EXPRESSIVE_VOICE_SETTINGS.speed) < 0.01
  ) {
    return QUINN_EXPRESSIVE_VOICE_SETTINGS;
  }

  return {
    ...QUINN_EXPRESSIVE_VOICE_SETTINGS,
    speed: hintedSpeed,
  };
}

function normalizeElevenText(value, maxLength = 0) {
  const clean = String(value || '').replace(/\s+/g, ' ').trim();

  if (!clean) {
    return '';
  }

  if (!maxLength || clean.length <= maxLength) {
    return clean;
  }

  return `${clean.slice(0, maxLength - 3).trim()}...`;
}

async function requestElevenSpeech(voiceId, apiKey, body) {
  return fetch(
    `${ELEVEN_TTS_BASE_URL}/${encodeURIComponent(voiceId)}?output_format=${QUINN_OUTPUT_FORMAT}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        Accept: 'audio/mpeg',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }
  );
}

function buildRetryBody(baseBody, errorText) {
  const lowerError = String(errorText || '').toLowerCase();
  let nextBody = null;

  if (lowerError.includes('apply_text_normalization')) {
    const { apply_text_normalization, ...withoutNormalization } = baseBody;
    nextBody = withoutNormalization;
  }

  if (lowerError.includes('previous_text') || lowerError.includes('next_text')) {
    const source = nextBody || baseBody;
    const {
      previous_text,
      next_text,
      ...withoutContinuityHints
    } = source;
    nextBody = withoutContinuityHints;
  }

  if (lowerError.includes('style')) {
    const source = nextBody || baseBody;
    const voiceSettings =
      source && typeof source.voice_settings === 'object' && source.voice_settings
        ? source.voice_settings
        : null;

    if (voiceSettings && Object.prototype.hasOwnProperty.call(voiceSettings, 'style')) {
      const { style, ...withoutStyle } = voiceSettings;
      void style;
      nextBody = {
        ...source,
        voice_settings: withoutStyle,
      };
    }
  }

  return nextBody;
}

function getRequiredEnv(name) {
  const value = String(process.env[name] || '').trim();

  if (!value) {
    throw new Error(`Missing ${name} environment variable.`);
  }

  return value;
}

export async function generateElevenSpeech({
  text,
  format = 'mp3',
  previousText = '',
  nextText = '',
  prosodyHint = null,
} = {}) {
  const apiKey = getRequiredEnv('ELEVENLABS_API_KEY');
  const voiceId = getRequiredEnv('ELEVENLABS_VOICE_ID');
  const modelId = getRequiredEnv('ELEVENLABS_MODEL_ID');

  const cleanText = normalizeElevenText(text);
  const cleanPreviousText = normalizeElevenText(previousText, 320);

  if (!cleanText) {
    throw new Error('No text provided for ElevenLabs speech.');
  }

  if (format !== 'mp3') {
    throw new Error(`Unsupported ElevenLabs format: ${format}`);
  }

  const baseBody = {
    text: cleanText,
    model_id: modelId,
    voice_settings: buildQuinnVoiceSettings(prosodyHint),
    apply_text_normalization: 'on',
    ...(cleanPreviousText ? { previous_text: cleanPreviousText } : {}),
  };

  let response = await requestElevenSpeech(voiceId, apiKey, baseBody);

  if (!response.ok) {
    let errorText = await response.text();
    const retryBody = buildRetryBody(baseBody, errorText);

    if (retryBody) {
      response = await requestElevenSpeech(voiceId, apiKey, retryBody);

      if (!response.ok) {
        errorText = await response.text();
      }
    }

    if (!response.ok) {
      throw new Error(
        `ElevenLabs request failed: ${response.status} ${response.statusText}\n${errorText}`
      );
    }
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}
