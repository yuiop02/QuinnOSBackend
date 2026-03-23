const FISH_API_URL = 'https://api.fish.audio/v1/tts';

function getRequiredEnv(name) {
  const value = String(process.env[name] || '').trim();

  if (!value) {
    throw new Error(`Missing ${name} environment variable.`);
  }

  return value;
}

export async function generateFishSpeech({
  text,
  format = 'mp3',
  referenceId,
} = {}) {
  const apiKey = getRequiredEnv('FISH_API_KEY');
  const resolvedReferenceId = String(
    referenceId || getRequiredEnv('FISH_REFERENCE_ID')
  ).trim();

  const cleanText = String(text || '').replace(/\s+/g, ' ').trim();

  if (!cleanText) {
    throw new Error('No text provided for Fish speech.');
  }

  if (format !== 'mp3') {
    throw new Error(`Unsupported Fish format: ${format}`);
  }

  const response = await fetch(FISH_API_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      model: 's1',
    },
    body: JSON.stringify({
      text: cleanText,
      reference_id: resolvedReferenceId,
      format,
      mp3_bitrate: 128,
      temperature: 0.72,
      top_p: 0.8,
      latency: 'normal',
      prosody: {
        speed: 0.96,
        volume: 0,
      },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Fish request failed: ${response.status} ${response.statusText}\n${errorText}`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}
