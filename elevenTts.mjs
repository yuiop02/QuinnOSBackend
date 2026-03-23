const ELEVEN_TTS_BASE_URL = 'https://api.elevenlabs.io/v1/text-to-speech';

function getRequiredEnv(name) {
  const value = String(process.env[name] || '').trim();

  if (!value) {
    throw new Error(`Missing ${name} environment variable.`);
  }

  return value;
}

export async function generateElevenSpeech({ text, format = 'mp3' } = {}) {
  const apiKey = getRequiredEnv('ELEVENLABS_API_KEY');
  const voiceId = getRequiredEnv('ELEVENLABS_VOICE_ID');
  const modelId = getRequiredEnv('ELEVENLABS_MODEL_ID');

  const cleanText = String(text || '').replace(/\s+/g, ' ').trim();

  if (!cleanText) {
    throw new Error('No text provided for ElevenLabs speech.');
  }

  if (format !== 'mp3') {
    throw new Error(`Unsupported ElevenLabs format: ${format}`);
  }

  const response = await fetch(
    `${ELEVEN_TTS_BASE_URL}/${encodeURIComponent(voiceId)}?output_format=mp3_44100_128`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        Accept: 'audio/mpeg',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: cleanText,
        model_id: modelId,
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs request failed: ${response.status} ${response.statusText}\n${errorText}`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}
