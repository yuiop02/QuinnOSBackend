import fs from 'fs';
import os from 'os';
import path from 'path';
import multer from 'multer';
import OpenAI from 'openai';

const TEMP_DIR = path.join(os.tmpdir(), 'quinnos-transcriptions');
fs.mkdirSync(TEMP_DIR, { recursive: true });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const ALLOWED_EXTENSIONS = new Set([
  '.mp3',
  '.mp4',
  '.mpeg',
  '.mpga',
  '.m4a',
  '.wav',
  '.webm',
]);

function buildSafeName(originalName = 'recording.m4a') {
  const extension = path.extname(originalName).toLowerCase() || '.m4a';
  const base = path.basename(originalName, extension).replace(/[^a-z0-9-_]/gi, '-');
  return `${Date.now()}-${base || 'recording'}${extension}`;
}

function cleanupTempFile(filePath) {
  if (!filePath) {
    return;
  }

  fs.promises.unlink(filePath).catch(() => {});
}

function buildTranscriptionPrompt({ packetTitle, packetText, lastSummary }) {
  const safeTitle = String(packetTitle || '').replace(/\s+/g, ' ').trim();
  const safePacket = String(packetText || '').replace(/\s+/g, ' ').trim();
  const safeSummary = String(lastSummary || '').replace(/\s+/g, ' ').trim();

  return [
    'Transcribe the speaker accurately with punctuation.',
    'Do not summarize.',
    'Keep product terms exactly when spoken: Quinn, QuinnOS, Gravity, Canvas, Memory, Voice Mode.',
    safeTitle ? `Current packet title: ${safeTitle}` : '',
    safePacket ? `Current packet context: ${safePacket.slice(0, 500)}` : '',
    safeSummary ? `Latest Quinn summary context: ${safeSummary.slice(0, 250)}` : '',
  ]
    .filter(Boolean)
    .join('\n');
}

const storage = multer.diskStorage({
  destination: (_req, _file, callback) => {
    callback(null, TEMP_DIR);
  },
  filename: (_req, file, callback) => {
    callback(null, buildSafeName(file.originalname));
  },
});

function fileFilter(_req, file, callback) {
  const extension = path.extname(file.originalname || '').toLowerCase();

  if (!ALLOWED_EXTENSIONS.has(extension)) {
    callback(
      new Error(
        'Unsupported audio type. Use mp3, mp4, mpeg, mpga, m4a, wav, or webm.'
      )
    );
    return;
  }

  callback(null, true);
}

export const transcriptionUpload = multer({
  storage,
  limits: {
    fileSize: 25 * 1024 * 1024,
  },
  fileFilter,
});

export async function handleTranscriptionRoute(req, res) {
  const tempPath = req.file?.path;

  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        ok: false,
        error: 'OPENAI_API_KEY is not set on the backend.',
      });
    }

    if (!req.file) {
      return res.status(400).json({
        ok: false,
        error: 'No audio file received. Send multipart/form-data with field name "audio".',
      });
    }

    const model = process.env.OPENAI_TRANSCRIBE_MODEL || 'gpt-4o-mini-transcribe';

    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempPath),
      model,
      response_format: 'json',
      prompt: buildTranscriptionPrompt({
        packetTitle: req.body?.packetTitle,
        packetText: req.body?.packetText,
        lastSummary: req.body?.lastSummary,
      }),
    });

    const transcript = String(transcription?.text || '').trim();

    return res.status(200).json({
      ok: true,
      transcript,
      durationMillis: Number(req.body?.durationMillis || 0),
      provider: model,
      fileName: req.file.originalname || 'recording.m4a',
      createdAt: new Date().toISOString(),
    });
  } catch (error) {
    const message =
      error instanceof Error && error.message
        ? error.message
        : 'Transcription failed.';

    return res.status(500).json({
      ok: false,
      error: message,
    });
  } finally {
    cleanupTempFile(tempPath);
  }
}