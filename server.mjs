import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateElevenSpeech } from './elevenTts.mjs';
import { handleTranscriptionRoute, transcriptionUpload } from './quinnTranscription.mjs';

dotenv.config();

const app = express();
const port = Number(process.env.PORT || 8787);
const host = process.env.HOST || '0.0.0.0';
const model = process.env.OPENAI_MODEL || 'gpt-5-mini';

app.use(cors());
app.use(express.json({ limit: '4mb' }));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  maxRetries: 2,
  timeout: 60_000,
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const bundledDataDir = path.join(__dirname, 'data');
const bundledMemoryPath = path.join(bundledDataDir, 'memory.json');
const runtimeDataDir = path.resolve(
  String(process.env.QUINNOS_STORAGE_DIR || process.env.RAILWAY_VOLUME_MOUNT_PATH || bundledDataDir)
);
const memoryPath = path.resolve(
  String(process.env.QUINNOS_MEMORY_FILE || path.join(runtimeDataDir, 'memory.json'))
);
const VOICE_BASE_URL = String(
  process.env.VOICE_BASE_URL ||
    process.env.QUINN_VOICE_BASE_URL ||
    `http://127.0.0.1:${Number(process.env.FISH_VOICE_PORT || process.env.QUINN_VOICE_PORT || 8788)}`
)
  .trim()
  .replace(/\/+$/, '');
const VOICE_PROXY_ONLY = String(process.env.VOICE_PROXY_ONLY || '')
  .trim()
  .toLowerCase() === 'true';
const VOICE_PREPARED_REQUEST_TTL_MS = Number(process.env.VOICE_PREPARED_REQUEST_TTL_MS || 5 * 60 * 1000);
const preparedVoiceRequests = new Map();

const DEFAULT_CONSTRAINTS =
  'no “if you want” hedging | no generic advice | match my voice | don’t repeat yourself | do not moralize at me | do not over-explain';

const EMPTY_PACKET_FORM = {
  mode: '',
  domain: '',
  ask: '',
  task: '',
  output: '',
  toneRisk: '',
  timebox: '',
  stop: '',
  context: '',
  facts: '',
  feelings: '',
  nonNegotiables: '',
  constraints: DEFAULT_CONSTRAINTS,
  success: '',
  iDid: '',
  itCaused: '',
};

const DEFAULT_MEMORY = {
  profileSummary: '',
  coreMemories: [],
  preferences: [],
  doNotDo: [],
  lifeContext: [],
  bigProjects: [],
  lastResponseId: null,
  runs: [],
  feedback: [],
  exports: [],
  backups: [],
};

function hasLocalElevenVoiceConfig() {
  return Boolean(
    process.env.ELEVENLABS_API_KEY &&
      process.env.ELEVENLABS_VOICE_ID &&
      process.env.ELEVENLABS_MODEL_ID
  );
}

function shouldAllowLocalVoiceFallback() {
  return !VOICE_PROXY_ONLY && hasLocalElevenVoiceConfig();
}

function normalizeVoiceTransportText(value, maxLength = 0) {
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

function buildVoiceProsodyQueryObject(prosodyHint) {
  const normalized = normalizeVoiceProsodyHint(prosodyHint);

  if (!normalized) {
    return {};
  }

  return {
    voice_profile: normalized.profile,
    voice_speed: String(normalized.speed),
    voice_pace: normalized.pace,
    voice_landing: normalized.landing,
    voice_smoothness: normalized.smoothness,
    voice_contour: normalized.contour,
  };
}

function prunePreparedVoiceRequests() {
  const now = Date.now();

  for (const [token, entry] of preparedVoiceRequests.entries()) {
    if (now - entry.createdAt > VOICE_PREPARED_REQUEST_TTL_MS) {
      preparedVoiceRequests.delete(token);
    }
  }
}

function createPreparedVoiceRequest({
  text,
  previousText = '',
  nextText = '',
  prosodyHint = null,
}) {
  const cleanText = normalizeVoiceTransportText(text);
  const cleanPreviousText = normalizeVoiceTransportText(previousText, 320);
  const cleanNextText = normalizeVoiceTransportText(nextText, 320);
  const cleanProsodyHint = normalizeVoiceProsodyHint(prosodyHint);

  if (!cleanText) {
    return null;
  }

  prunePreparedVoiceRequests();

  const token = `voice-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  preparedVoiceRequests.set(token, {
    createdAt: Date.now(),
    text: cleanText,
    previousText: cleanPreviousText,
    nextText: cleanNextText,
    prosodyHint: cleanProsodyHint,
  });

  return token;
}

function getPreparedVoiceRequest(token) {
  const cleanToken = String(token || '').trim();

  if (!cleanToken) {
    return null;
  }

  prunePreparedVoiceRequests();
  return preparedVoiceRequests.get(cleanToken) || null;
}

async function writeTextFileAtomic(filePath, value) {
  const targetDir = path.dirname(filePath);
  const tempPath = path.join(
    targetDir,
    `${path.basename(filePath)}.${process.pid}.${Date.now()}.tmp`
  );

  await fs.mkdir(targetDir, { recursive: true });
  await fs.writeFile(tempPath, value, 'utf8');
  await fs.rename(tempPath, filePath);
}

async function ensureMemoryFile() {
  await fs.mkdir(path.dirname(memoryPath), { recursive: true });

  try {
    await fs.access(memoryPath);
  } catch {
    try {
      if (path.resolve(bundledMemoryPath) !== path.resolve(memoryPath)) {
        const bundledMemory = await fs.readFile(bundledMemoryPath, 'utf8');
        await writeTextFileAtomic(memoryPath, bundledMemory);
        return;
      }
    } catch {
      // Fall through to creating a fresh default file when no bundled memory is available.
    }

    await writeTextFileAtomic(memoryPath, JSON.stringify(DEFAULT_MEMORY, null, 2));
  }
}

function cleanArray(value) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function uniqueMerge(existing, incoming) {
  return Array.from(new Set([...cleanArray(existing), ...cleanArray(incoming)]));
}

function mergeSummary(existing, incoming) {
  const a = String(existing || '').trim();
  const b = String(incoming || '').trim();

  if (!a) return b;
  if (!b) return a;
  if (a.includes(b)) return a;

  return `${a}\n\n${b}`;
}

function summarizeText(text, maxLength = 320) {
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  if (!clean) return '';
  return clean.length > maxLength ? `${clean.slice(0, maxLength)}...` : clean;
}

const STORED_SUMMARY_ABSTRACTION_RULES = [
  { pattern: /\bsafeway blueberry pudding ring\b/gi, replacement: 'pudding-ring callback' },
  { pattern: /\bpudding ring\b/gi, replacement: 'pudding-ring callback' },
  { pattern: /\bdiet coke(?: in a can)?\b/gi, replacement: 'diet-coke callback' },
  { pattern: /\brap god\b/gi, replacement: 'rap-god callback' },
  { pattern: /\bstarbucks\b|\bbarista\b/gi, replacement: 'workplace callback' },
  { pattern: /\bassistant store manager\b|\basm\b/gi, replacement: 'promotion callback' },
];

function normalizeStoredSummaryText(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .replace(/\b\d+[.)]\s+/g, '')
    .replace(/\s*[-*]\s+/g, ' ')
    .trim();
}

function abstractStoredSummaryMotifs(text) {
  let clean = normalizeStoredSummaryText(text);

  for (const rule of STORED_SUMMARY_ABSTRACTION_RULES) {
    clean = clean.replace(rule.pattern, rule.replacement);
  }

  return clean
    .replace(/\b(pudding-ring callback)(?:,\s*\1)+/gi, '$1')
    .replace(/\b(diet-coke callback)(?:,\s*\1)+/gi, '$1')
    .replace(/\b(rap-god callback)(?:,\s*\1)+/gi, '$1')
    .replace(/\b(workplace callback)(?:,\s*\1)+/gi, '$1')
    .replace(/\b(promotion callback)(?:,\s*\1)+/gi, '$1')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function buildStoredSummary(text, maxLength = 220) {
  const clean = abstractStoredSummaryMotifs(text);

  if (!clean) return '';

  const sentences = clean.match(/[^.!?]+[.!?]?/g) || [clean];
  const joined = sentences.slice(0, 2).join(' ').trim();

  return summarizeText(joined || clean, maxLength);
}

function normalizeProjectTag(value) {
  const clean = String(value || '').trim();
  return clean || 'General';
}

function normalizeProfileId(value) {
  return String(value || '').trim();
}

function listBlock(title, items) {
  if (!items || items.length === 0) {
    return `${title}:\n- None saved yet`;
  }

  return `${title}:\n${items.map((item) => `- ${item}`).join('\n')}`;
}

function recentRunsBlock(runs) {
  if (!runs || runs.length === 0) {
    return 'RECENT QUINNOS RUNS:\n- No runs yet';
  }

  return `RECENT QUINNOS RUNS:\n${runs
    .slice(0, 5)
    .map((run, index) => {
      const status = run.status || 'unknown';
      const packet = run.packetSummary || 'No packet summary';
      const result = run.responseSummary || run.error || 'No result summary';
      const project = run.projectTag || 'General';
      return `${index + 1}. ${run.at} • ${status} • Project: ${project}\nPacket summary: ${packet}\nResult summary: ${result}`;
    })
    .join('\n\n')}`;
}

function feedbackLearningBlock(feedback) {
  if (!feedback || feedback.length === 0) {
    return 'USER FEEDBACK LEARNING:\n- No feedback saved yet';
  }

  const liked = feedback.filter((item) => item.rating === 'up').slice(0, 5);
  const disliked = feedback.filter((item) => item.rating === 'down').slice(0, 5);

  const likedBlock =
    liked.length === 0
      ? '- No positive ratings yet'
      : liked
          .map((item, index) => {
            const note = item.notes ? `Note: ${item.notes}` : 'Note: none';
            return `${index + 1}. Project: ${item.projectTag || 'General'}\n${note}\nResponse summary: ${item.responseSummary || 'No summary'}`;
          })
          .join('\n\n');

  const dislikedBlock =
    disliked.length === 0
      ? '- No negative ratings yet'
      : disliked
          .map((item, index) => {
            const note = item.notes ? `Note: ${item.notes}` : 'Note: none';
            return `${index + 1}. Project: ${item.projectTag || 'General'}\n${note}\nResponse summary: ${item.responseSummary || 'No summary'}`;
          })
          .join('\n\n');

  return `USER FEEDBACK LEARNING

LIKED RESPONSES:
${likedBlock}

DISLIKED RESPONSES:
${dislikedBlock}`;
}

function buildMemoryContext(memory) {
  return [
    `PROFILE SUMMARY:\n${memory.profileSummary || 'Not filled yet.'}`,
    listBlock('CORE MEMORIES', memory.coreMemories),
    listBlock('PREFERENCES', memory.preferences),
    listBlock('DO NOT DO', memory.doNotDo),
    listBlock('LIFE CONTEXT', memory.lifeContext),
    listBlock('BIG PROJECTS', memory.bigProjects),
    recentRunsBlock(memory.runs),
    feedbackLearningBlock(memory.feedback),
  ].join('\n\n');
}

const LOW_PRIORITY_REFERENCE_PATTERNS = [
  /\bpudding ring\b/i,
  /\bdiet coke\b/i,
  /\bsafeway\b/i,
  /\brap god\b/i,
];

const RECENT_MOTIF_PATTERNS = [
  {
    label: 'Starbucks/barista callback',
    pattern: /\bstarbucks\b|\bbarista\b|\bworkplace callback\b/i,
  },
  {
    label: 'ASM promotion framing',
    pattern: /\bassistant store manager\b|\basm\b|\bpromotion callback\b/i,
  },
  {
    label: 'pudding ring callback',
    pattern: /\bpudding ring\b|\bpudding-ring callback\b/i,
  },
  {
    label: 'Diet Coke callback',
    pattern: /\bdiet coke\b|\bdiet-coke callback\b/i,
  },
  {
    label: 'Rap God callback',
    pattern: /\brap god\b|\brap-god callback\b/i,
  },
];

const MEMORY_MATCH_STOPWORDS = new Set([
  'about',
  'after',
  'also',
  'been',
  'being',
  'from',
  'have',
  'into',
  'just',
  'like',
  'make',
  'more',
  'need',
  'only',
  'really',
  'same',
  'than',
  'that',
  'them',
  'then',
  'they',
  'this',
  'want',
  'what',
  'when',
  'where',
  'which',
  'while',
  'with',
  'would',
  'your',
  'packet',
  'current',
  'request',
  'quinn',
  'title',
  'mode',
  'output',
  'context',
  'thread',
  'session',
  'continuity',
  'carryover',
  'beats',
  'step',
]);

const WORK_CONTEXT_KEYWORDS = [
  'work',
  'job',
  'store',
  'shift',
  'manager',
  'barista',
  'customer',
  'coworker',
  'coworkers',
  'promotion',
  'asm',
  'starbucks',
];

const RELATIONSHIP_CONTEXT_KEYWORDS = [
  'relationship',
  'dating',
  'breakup',
  'partner',
  'boyfriend',
  'girlfriend',
  'friend',
  'friends',
  'family',
  'mom',
  'dad',
  'conversation',
  'text',
  'call',
  'apology',
  'hurt',
  'grief',
  'love',
  'lonely',
  'anxious',
];

const BUILD_CONTEXT_KEYWORDS = [
  'quinnos',
  'prototype',
  'frontend',
  'backend',
  'server',
  'expo',
  'react',
  'node',
  'voice',
  'memory',
  'prompt',
  'debug',
  'bug',
  'patch',
  'feature',
  'code',
  'build',
];

function cleanMemoryText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function escapeRegex(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isLowPriorityReference(value) {
  const text = cleanMemoryText(value);
  return LOW_PRIORITY_REFERENCE_PATTERNS.some((pattern) => pattern.test(text));
}

function takeDistinctItems(items, limit = 8, { excludeLowPriority = false } = {}) {
  const output = [];
  const seen = new Set();

  for (const rawItem of cleanArray(items)) {
    const item = cleanMemoryText(rawItem);

    if (!item) continue;
    if (excludeLowPriority && isLowPriorityReference(item)) continue;

    const key = item.toLowerCase();

    if (seen.has(key)) continue;

    seen.add(key);
    output.push(item);

    if (output.length >= limit) {
      break;
    }
  }

  return output;
}

function buildIdentityCapsule(memory) {
  const summaryParagraphs = String(memory.profileSummary || '')
    .split(/\n\s*\n/)
    .map((part) => cleanMemoryText(part))
    .filter(Boolean)
    .filter((part) => !isLowPriorityReference(part))
    .slice(0, 2)
    .map((part) => summarizeText(part, 260));

  const core = takeDistinctItems(memory.coreMemories, 6, {
    excludeLowPriority: true,
  });
  const context = takeDistinctItems(memory.lifeContext, 4, {
    excludeLowPriority: true,
  }).filter(
    (item) => !core.some((coreItem) => coreItem.toLowerCase() === item.toLowerCase())
  );
  const projects = takeDistinctItems(memory.bigProjects, 3, {
    excludeLowPriority: true,
  });

  const items = [
    ...summaryParagraphs,
    ...core.slice(0, 4),
    ...context.slice(0, 3),
    ...projects.slice(0, 2),
  ];

  if (items.length === 0) {
    return 'QUIET BACKGROUND:\n- No useful background saved yet';
  }

  return `QUIET BACKGROUND:\n${items
    .slice(0, 8)
    .map((item) => `- ${item}`)
    .join('\n')}`;
}

function buildStyleCapsule(memory) {
  const preferred = takeDistinctItems(memory.preferences, 8, {
    excludeLowPriority: true,
  });
  const avoid = takeDistinctItems(memory.doNotDo, 8, {
    excludeLowPriority: true,
  });

  const items = [
    ...preferred.slice(0, 6).map((item) => `Lean toward: ${item}`),
    ...avoid.slice(0, 6).map((item) => `Stay away from: ${item}`),
  ];

  if (items.length === 0) {
    return 'VOICE CUES:\n- No voice cues saved yet';
  }

  return `VOICE CUES:\n${items
    .slice(0, 10)
    .map((item) => `- ${item}`)
    .join('\n')}`;
}

function normalizeSearchText(value) {
  return cleanMemoryText(value).toLowerCase();
}

function containsPhrase(text, phrase) {
  const normalizedText = normalizeSearchText(text);
  const normalizedPhrase = normalizeSearchText(phrase);

  if (!normalizedText || !normalizedPhrase) {
    return false;
  }

  const pattern = new RegExp(
    `\\b${escapeRegex(normalizedPhrase).replace(/\s+/g, '\\s+')}\\b`,
    'i'
  );

  return pattern.test(normalizedText);
}

function tokenizeSearchTerms(value) {
  return Array.from(
    new Set(
      normalizeSearchText(value)
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter((word) => word.length >= 4 && !MEMORY_MATCH_STOPWORDS.has(word))
    )
  );
}

function countKeywordHits(text, keywords) {
  return keywords.reduce(
    (total, keyword) => total + (containsPhrase(text, keyword) ? 1 : 0),
    0
  );
}

function countTokenOverlap(tokens, text) {
  const haystack = ` ${normalizeSearchText(text).replace(/[^a-z0-9\s]/g, ' ')} `;

  return tokens.reduce(
    (total, token) => total + (haystack.includes(` ${token} `) ? 1 : 0),
    0
  );
}

function mergeDistinctItems(...groups) {
  const output = [];
  const seen = new Set();

  for (const item of groups.flat()) {
    const clean = cleanMemoryText(item);

    if (!clean) continue;

    const key = clean.toLowerCase();

    if (seen.has(key)) continue;

    seen.add(key);
    output.push(clean);
  }

  return output;
}

function formatSelectedMemoryBlock(title, items) {
  if (!items || items.length === 0) {
    return '';
  }

  return `${title}:\n${items.map((item) => `- ${item}`).join('\n')}`;
}

function parseMemorySection(block) {
  const lines = String(block || '')
    .split('\n')
    .map((line) => String(line || '').trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return null;
  }

  const title = lines[0].replace(/:\s*$/, '').trim();
  const items = lines
    .slice(1)
    .map((line) => line.replace(/^- /, '').trim())
    .filter(Boolean);

  if (!title || items.length === 0) {
    return null;
  }

  return { title, items };
}

const MEMORY_RESONANCE_LABELS = {
  'LOCAL COURSE CORRECTION': 'Course correction',
  'QUIET BACKGROUND': 'Background context',
  'VOICE CUES': 'Voice cues',
  'WORK BACKGROUND': 'Work context',
  'RELATIONSHIP BACKGROUND': 'Relationship context',
  'PROJECT BACKGROUND': 'Project context',
  'BACKGROUND THAT MAY HELP HERE': 'Background context',
  'REFERENCE DETAIL IF USEFUL': 'Reference detail',
  'FRESHNESS GUARD': 'Freshness guard',
};
const MEMORY_RESONANCE_GUARD_KEYWORDS = {
  'WORK BACKGROUND': WORK_CONTEXT_KEYWORDS,
  'RELATIONSHIP BACKGROUND': RELATIONSHIP_CONTEXT_KEYWORDS,
  'PROJECT BACKGROUND': BUILD_CONTEXT_KEYWORDS,
};

const MEMORY_RESONANCE_PRIORITIES = {
  'LOCAL COURSE CORRECTION': -1,
  'WORK BACKGROUND': 0,
  'RELATIONSHIP BACKGROUND': 0,
  'PROJECT BACKGROUND': 0,
  'BACKGROUND THAT MAY HELP HERE': 1,
  'VOICE CUES': 2,
  'QUIET BACKGROUND': 3,
  'FRESHNESS GUARD': 4,
  'REFERENCE DETAIL IF USEFUL': 5,
};
const TARGETED_MEMORY_MIN_SCORE = 2;
const GENERAL_MEMORY_MIN_SCORE = 4;
const REFERENCE_MEMORY_MIN_SCORE = 3;
const MEMORY_CONTEXT_TUNING = {
  lightTurnMaxWords: 10,
  recentBlockedReplyLookback: 3,
};

function buildRunMemorySections(
  memory,
  {
    packet = '',
    projectTag = 'General',
    previousAssistantReply = '',
    threadId = '',
  } = {}
) {
  const signals = buildPacketSignals(packet, projectTag);
  const relevantBlocks = buildRelevantMemoryBlocks(memory, signals);
  const threadContinuityControlBlock = buildThreadContinuityControlBlock(packet, signals);
  const localCourseCorrectionBlock = buildImmediateCourseCorrectionBlock(
    packet,
    memory.runs,
    signals,
    {
      previousAssistantReply,
      threadId,
    }
  );

  return [
    threadContinuityControlBlock,
    localCourseCorrectionBlock,
    signals.shouldThrottleHeavyMemory ? '' : buildStyleCapsule(memory),
    ...relevantBlocks,
    shouldIncludeIdentityMemory(signals, relevantBlocks.length)
      ? buildIdentityCapsule(memory)
      : '',
    buildAntiRepetitionBlock(memory.runs, threadId),
  ]
    .map(parseMemorySection)
    .filter(Boolean);
}

function buildRunMemoryResonance(sections) {
  return [...(Array.isArray(sections) ? sections : [])]
    .filter((section) => String(section?.title || '').trim() !== 'THREAD CONTINUITY CONTROL')
    .sort(
      (a, b) =>
        (MEMORY_RESONANCE_PRIORITIES[a.title] ?? 99) -
        (MEMORY_RESONANCE_PRIORITIES[b.title] ?? 99)
    )
    .slice(0, 4)
    .map((section) => ({
      label: resolveMemoryResonanceLabel(section),
      preview: summarizeText(
        String(section.items?.[0] || '').replace(
          /^(DO|AVOID|Lean toward|Stay away from):\s*/i,
          ''
        ),
        96
      ),
    }))
    .filter((item) => item.label || item.preview);
}

function resolveMemoryResonanceLabel(section) {
  const sectionTitle = String(section?.title || '').trim();
  const guardKeywords = MEMORY_RESONANCE_GUARD_KEYWORDS[sectionTitle];

  if (
    Array.isArray(guardKeywords) &&
    !countKeywordHits(String(section?.items || ''), guardKeywords)
  ) {
    return 'Background context';
  }

  return MEMORY_RESONANCE_LABELS[sectionTitle] || sectionTitle;
}

function profileSummaryParagraphs(memory, { includeLowPriority = false } = {}) {
  return String(memory.profileSummary || '')
    .split(/\n\s*\n/)
    .map((part) => cleanMemoryText(part))
    .filter(Boolean)
    .filter((part) => includeLowPriority || !isLowPriorityReference(part));
}

function normalizeMemoryExpression(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'implicit';
  }

  if (/\bselective\s*explicit\b/i.test(text)) {
    return 'selectiveExplicit';
  }

  if (/\bexplicit\b/i.test(text)) {
    return 'explicit';
  }

  return 'implicit';
}

function normalizeCorrectionLatch(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'none';
  }

  if (/\bhard\b/i.test(text)) {
    return 'hard';
  }

  if (/\bsoft\b/i.test(text)) {
    return 'soft';
  }

  return 'none';
}

function normalizeConstraintPriority(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'none';
  }

  if (/\bdominant\b/i.test(text)) {
    return 'dominant';
  }

  if (/\belevated\b/i.test(text)) {
    return 'elevated';
  }

  return 'none';
}

function normalizeRepeatGuard(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'none';
  }

  if (/\bavoid\s*exact\b/i.test(text)) {
    return 'avoidExact';
  }

  if (/\bavoid\s*near\s*repeat\b/i.test(text)) {
    return 'avoidNearRepeat';
  }

  return 'none';
}

function normalizeClarificationOverride(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'none';
  }

  if (/\bdominant\b/i.test(text)) {
    return 'dominant';
  }

  if (/\bpartial\b/i.test(text)) {
    return 'partial';
  }

  return 'none';
}

function normalizeInterpretationReplacement(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return false;
  }

  return /\b(?:true|yes|1)\b/i.test(text);
}

function normalizeClarificationType(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'none';
  }

  if (/\breference\b/i.test(text)) {
    return 'reference';
  }

  if (/\bsubject\b/i.test(text)) {
    return 'subject';
  }

  if (/\bmeaning\b/i.test(text)) {
    return 'meaning';
  }

  if (/\bcategory\b/i.test(text)) {
    return 'category';
  }

  if (/\btone\b/i.test(text)) {
    return 'tone';
  }

  return 'none';
}

function normalizeActiveThreadContinuity(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return false;
  }

  return /\b(?:true|yes|1)\b/i.test(text);
}

function normalizeLiveSubjectDominance(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'low';
  }

  if (/\bhigh\b/i.test(text)) {
    return 'high';
  }

  if (/\bmedium\b/i.test(text)) {
    return 'medium';
  }

  return 'low';
}

function normalizeThreadCarryoverMode(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'keep';
  }

  if (/\bdrop\b/i.test(text)) {
    return 'drop';
  }

  if (/\bsoften\b/i.test(text)) {
    return 'soften';
  }

  return 'keep';
}

function normalizeStaleFrameRisk(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return 'none';
  }

  if (/\bstrong\b/i.test(text)) {
    return 'strong';
  }

  if (/\blight\b/i.test(text)) {
    return 'light';
  }

  return 'none';
}

function normalizeFrameContinuation(value) {
  const text = normalizeSearchText(value);

  if (!text) {
    return false;
  }

  return /\b(?:true|yes|1)\b/i.test(text);
}

function buildPacketSignals(packet, projectTag = 'General') {
  const title = extractPacketField(packet, 'TITLE');
  const liveNoteText = cleanMemoryText(
    extractPacketTailField(packet, 'PACKET') ||
      extractPacketField(packet, 'PACKET') ||
      packet
  );
  const domain = extractPacketField(packet, 'DOMAIN');
  const memoryExpression = normalizeMemoryExpression(
    extractPacketField(packet, 'MEMORY EXPRESSION')
  );
  const correctionLatch = normalizeCorrectionLatch(
    extractPacketField(packet, 'CORRECTION LATCH')
  );
  const constraintPriority = normalizeConstraintPriority(
    extractPacketField(packet, 'CONSTRAINT PRIORITY')
  );
  const repeatGuard = normalizeRepeatGuard(
    extractPacketField(packet, 'REPEAT GUARD')
  );
  const clarificationOverride = normalizeClarificationOverride(
    extractPacketField(packet, 'CLARIFICATION OVERRIDE')
  );
  const interpretationReplacement = normalizeInterpretationReplacement(
    extractPacketField(packet, 'INTERPRETATION REPLACEMENT')
  );
  const clarificationType = normalizeClarificationType(
    extractPacketField(packet, 'CLARIFICATION TYPE')
  );
  const activeThreadContinuity = normalizeActiveThreadContinuity(
    extractPacketField(packet, 'ACTIVE THREAD CONTINUITY')
  );
  const liveSubjectDominance = normalizeLiveSubjectDominance(
    extractPacketField(packet, 'LIVE SUBJECT DOMINANCE')
  );
  const threadCarryoverMode = normalizeThreadCarryoverMode(
    extractPacketField(packet, 'THREAD CARRYOVER MODE')
  );
  const staleFrameRisk = normalizeStaleFrameRisk(
    extractPacketField(packet, 'STALE FRAME RISK')
  );
  const frameContinuation = normalizeFrameContinuation(
    extractPacketField(packet, 'FRAME CONTINUATION')
  );
  const normalizedProjectTag = normalizeSearchText(projectTag);
  const sourceText = [liveNoteText, title, domain, projectTag].filter(Boolean).join('\n');
  const liveNoteWordCount = liveNoteText
    ? liveNoteText.split(/\s+/).filter(Boolean).length
    : 0;
  const wantsWorkContext = countKeywordHits(sourceText, WORK_CONTEXT_KEYWORDS) > 0;
  const wantsRelationshipContext =
    countKeywordHits(sourceText, RELATIONSHIP_CONTEXT_KEYWORDS) > 0;
  const wantsBuildContext = countKeywordHits(sourceText, BUILD_CONTEXT_KEYWORDS) > 0;
  const hasSpecificProjectTag = Boolean(
    normalizedProjectTag && normalizedProjectTag !== 'general'
  );
  const shouldThrottleHeavyMemory =
    repeatGuard !== 'none' ||
    ((clarificationOverride !== 'none' ||
      interpretationReplacement ||
      correctionLatch !== 'none' ||
      constraintPriority !== 'none') &&
      !hasSpecificProjectTag &&
      !wantsWorkContext &&
      !wantsRelationshipContext &&
      !wantsBuildContext) ||
    (!hasSpecificProjectTag &&
      liveNoteWordCount > 0 &&
      liveNoteWordCount <= MEMORY_CONTEXT_TUNING.lightTurnMaxWords &&
      !wantsWorkContext &&
      !wantsRelationshipContext &&
      !wantsBuildContext);

  return {
    liveNoteText,
    liveNoteWordCount,
    text: sourceText,
    tokens: tokenizeSearchTerms(sourceText),
    domain: normalizeSearchText(domain),
    projectTag: normalizedProjectTag,
    memoryExpression,
    correctionLatch,
    constraintPriority,
    repeatGuard,
    clarificationOverride,
    interpretationReplacement,
    clarificationType,
    activeThreadContinuity,
    liveSubjectDominance,
    threadCarryoverMode,
    staleFrameRisk,
    frameContinuation,
    hasSpecificProjectTag,
    wantsWorkContext,
    wantsRelationshipContext,
    wantsBuildContext,
    shouldThrottleHeavyMemory,
    allowsLowPriorityReferenceMemory:
      !shouldThrottleHeavyMemory &&
      LOW_PRIORITY_REFERENCE_PATTERNS.some((pattern) => pattern.test(sourceText)),
  };
}

function scoreMemoryItem(item, signals, keywords = [], { allowLowPriority = false } = {}) {
  const text = cleanMemoryText(item);

  if (!text) return 0;
  if (!allowLowPriority && isLowPriorityReference(text)) return 0;

  let score = 0;
  score += countTokenOverlap(signals.tokens, text) * 3;
  score += countKeywordHits(text, keywords) * 2;

  if (signals.domain && containsPhrase(text, signals.domain)) {
    score += 2;
  }

  if (signals.hasSpecificProjectTag && containsPhrase(text, signals.projectTag)) {
    score += 3;
  }

  return score;
}

function pickRelevantItems(items, signals, keywords = [], limit = 3, options = {}) {
  const candidates = takeDistinctItems(items, 20, {
    excludeLowPriority: !options.allowLowPriority,
  });
  const minimumScore = Number.isFinite(options.minimumScore) ? Number(options.minimumScore) : 1;
  const requireKeywordHit = Boolean(options.requireKeywordHit && keywords.length);

  return candidates
    .map((item) => ({
      item,
      score: scoreMemoryItem(item, signals, keywords, options),
      keywordHits: countKeywordHits(item, keywords),
    }))
    .filter(
      ({ score, keywordHits }) =>
        score >= minimumScore && (!requireKeywordHit || keywordHits > 0)
    )
    .sort((a, b) => b.score - a.score || a.item.length - b.item.length)
    .slice(0, limit)
    .map(({ item }) => item);
}

function findRecentBlockedReplyTexts(runs, threadId = '', limit = MEMORY_CONTEXT_TUNING.recentBlockedReplyLookback) {
  const successfulRuns = (Array.isArray(runs) ? runs : []).filter(
    (run) => run && run.status === 'success'
  );
  const cleanThreadId = cleanMemoryText(threadId);
  const scopedRuns = cleanThreadId
    ? successfulRuns.filter(
        (run) => cleanMemoryText(run?.threadId) === cleanThreadId
      )
    : successfulRuns;
  const pool = scopedRuns.length ? scopedRuns : successfulRuns;

  return takeDistinctItems(
    pool
      .map((run) => cleanMemoryText(run?.blockedReplyExcerpt || ''))
      .filter(Boolean),
    limit
  );
}

function buildAntiRepetitionBlock(runs, threadId = '') {
  const recentBlockedReplyTexts = findRecentBlockedReplyTexts(runs, threadId);
  const recentText = [
    ...recentBlockedReplyTexts,
    ...(Array.isArray(runs) ? runs : [])
      .slice(0, 8)
      .map((run) => cleanMemoryText(run.responseExcerpt || run.responseSummary || ''))
      .filter(Boolean),
  ].join('\n');
  const items = [];

  if (!recentText) {
    return 'FRESHNESS GUARD:\n- Avoid leaning on familiar personal callbacks unless clearly relevant.';
  }

  if (recentBlockedReplyTexts.length) {
    items.push(
      `Do not circle back to recently rejected local material: ${recentBlockedReplyTexts
        .slice(0, 2)
        .map((item) => summarizeText(item, 90))
        .join(' / ')}`
    );
  }

  const hits = RECENT_MOTIF_PATTERNS.filter(({ pattern }) => pattern.test(recentText)).map(
    ({ label }) => label
  );

  if (hits.length) {
    items.push(`Avoid reusing these recent motifs unless materially relevant: ${hits.join(', ')}`);
  } else {
    items.push(
      'Avoid reusing recent phrasings, motifs, or callback details unless the packet clearly calls for them.'
    );
  }

  return `FRESHNESS GUARD:\n${items.map((item) => `- ${item}`).join('\n')}`;
}

function buildThreadContinuityControlBlock(packet, signals) {
  if (!signals?.activeThreadContinuity) {
    return '';
  }

  const threadContinuityPolicy = extractPacketField(packet, 'THREAD CONTINUITY POLICY');
  const items = [];

  if (signals.threadCarryoverMode === 'drop') {
    items.push(
      'Same thread, different live subject. Answer the newest note directly and let earlier beats stay in the background.'
    );
  } else if (signals.threadCarryoverMode === 'soften') {
    items.push(
      'Same thread, but continuity is background support rather than the topic. Use it for calibration, not dominance.'
    );
  } else {
    items.push(
      'The current note still appears to continue the same subject. Carry the thread forward only as far as the newest turn keeps it alive.'
    );
  }

  if (signals.liveSubjectDominance === 'high') {
    items.push('The newest user turn clearly owns the live subject right now.');
  } else if (signals.liveSubjectDominance === 'medium') {
    items.push('The newest turn adds real live material. Let it lead over stale carryover.');
  }

  if (signals.staleFrameRisk === 'strong') {
    items.push(
      'Stale-frame risk is strong. Do not keep reacting from the earlier vibe, greeting posture, pet-name scene, or semantic stance if the note moved on.'
    );
  } else if (signals.staleFrameRisk === 'light') {
    items.push(
      'There is some stale-frame risk. Keep older thread framing light unless the newest note clearly calls back to it.'
    );
  }

  if (signals.frameContinuation) {
    items.push('Frame continuation is active, so continuity can stay live without taking over.');
  }

  if (
    threadContinuityPolicy &&
    !/no active thread carryover is competing with the live note/i.test(threadContinuityPolicy)
  ) {
    items.push(threadContinuityPolicy);
  }

  return items.length
    ? `THREAD CONTINUITY CONTROL:\n${items.map((item) => `- ${item}`).join('\n')}`
    : '';
}

function findLatestSuccessfulRun(runs, threadId = '') {
  const successfulRuns = (Array.isArray(runs) ? runs : []).filter(
    (run) => run && run.status === 'success'
  );
  const cleanThreadId = cleanMemoryText(threadId);

  if (!cleanThreadId) {
    return successfulRuns[0] || null;
  }

  return (
    successfulRuns.find(
      (run) => cleanMemoryText(run?.threadId) === cleanThreadId
    ) ||
    successfulRuns[0] ||
    null
  );
}

function buildImmediateCourseCorrectionBlock(
  packet,
  runs,
  signals,
  {
    previousAssistantReply = '',
    threadId = '',
  } = {}
) {
  const hasActiveCorrection =
    signals?.clarificationOverride !== 'none' ||
    signals?.interpretationReplacement ||
    signals?.correctionLatch !== 'none' ||
    signals?.constraintPriority !== 'none' ||
    signals?.repeatGuard !== 'none';

  if (!hasActiveCorrection) {
    return '';
  }

  const explicitPreviousAssistantReply = cleanMemoryText(previousAssistantReply);
  const latestSuccessfulRun = findLatestSuccessfulRun(runs, threadId);
  const latestRejectedMaterial = cleanMemoryText(
    explicitPreviousAssistantReply ||
      latestSuccessfulRun?.responseExcerpt ||
      latestSuccessfulRun?.responseSummary ||
      ''
  );
  const localCorrectionSummary = extractPacketField(packet, 'LOCAL COURSE CORRECTION');
  const items = [
    'The newest user turn is the live frame. Do not let older thread momentum outrank it.',
  ];

  if (
    signals?.clarificationOverride !== 'none' ||
    signals?.interpretationReplacement
  ) {
    items.push(
      'The user explicitly clarified what they meant. Replace the older interpretation with that clarified meaning. Do not keep both meanings alive.'
    );

    if (signals?.clarificationType === 'reference') {
      items.push(
        'Treat the disputed term as a nickname or way of addressing Quinn, not as the topic itself.'
      );
    } else if (signals?.clarificationType === 'subject') {
      items.push(
        'Treat Quinn as the subject or addressee of the phrase, not as an external topic or category guess.'
      );
    } else if (signals?.clarificationType === 'category') {
      items.push(
        'Drop the earlier category, genre, or topic reading. Use the corrected sense the user just supplied instead.'
      );
    } else if (signals?.clarificationType === 'meaning') {
      items.push(
        'Trust the clarified sense the user just gave you over the earlier semantic guess.'
      );
    } else if (signals?.clarificationType === 'tone') {
      items.push(
        'Treat the user’s clarification of tone or address as the live meaning for this turn.'
      );
    }
  }

  if (signals?.correctionLatch === 'hard') {
    items.push(
      'The user is explicitly correcting, rejecting, or invalidating the last move. Acknowledge that briefly, then pivot. Do not keep extending the invalidated frame.'
    );
  } else if (signals?.correctionLatch === 'soft') {
    items.push(
      'A local frame update is active. Favor the corrected angle over the older conversational momentum.'
    );
  }

  if (signals?.constraintPriority === 'dominant') {
    items.push(
      'A newly stated blocker is now the main fact. Treat desire, hype, or the earlier suggestion as secondary until the blocker is answered.'
    );
  } else if (signals?.constraintPriority === 'elevated') {
    items.push(
      'A practical constraint is active. Keep feasibility in view instead of replying as if enthusiasm alone settles it.'
    );
  }

  if (signals?.repeatGuard === 'avoidExact' || signals?.repeatGuard === 'avoidNearRepeat') {
    items.push(
      'The user just called repetition out. Do not reuse the same joke, line, premise, or suggestion right away.'
    );

    if (latestRejectedMaterial) {
      items.push(
        `Avoid this just-called-out material: ${summarizeText(latestRejectedMaterial, 220)}`
      );
    }

    items.push(
      signals.repeatGuard === 'avoidExact'
        ? 'Make the replacement genuinely different, not a lightly edited variation.'
        : 'Make the replacement materially different in content or phrasing, not just slightly reshuffled.'
    );
  } else if (
    latestRejectedMaterial &&
    (signals?.correctionLatch === 'hard' || signals?.constraintPriority === 'dominant')
  ) {
    items.push(
      `Do not keep extending this just-invalidated reply shape: ${summarizeText(
        latestRejectedMaterial,
        220
      )}`
    );
  }

  if (
    localCorrectionSummary &&
    !/no local correction override is active/i.test(localCorrectionSummary)
  ) {
    items.push(localCorrectionSummary);
  }

  if (items.length === 0) {
    return '';
  }

  return `LOCAL COURSE CORRECTION:\n${items.map((item) => `- ${item}`).join('\n')}`;
}

const IMMEDIATE_REPEAT_GUARD_TUNING = {
  tokenLengthMin: 3,
  phraseSize: 4,
  exactTokenOverlapThreshold: 0.88,
  exactPhraseOverlapThreshold: 0.62,
  nearTokenOverlapThreshold: 0.72,
  nearPhraseOverlapThreshold: 0.46,
  maxAttempts: 3,
  previousReplyMaxChars: 1400,
};

function clipImmediateReplyText(
  value,
  maxLength = IMMEDIATE_REPEAT_GUARD_TUNING.previousReplyMaxChars
) {
  const clean = cleanMemoryText(value);

  if (!clean) {
    return '';
  }

  if (clean.length <= maxLength) {
    return clean;
  }

  return `${clean.slice(0, maxLength - 3).trim()}...`;
}

function normalizeImmediateRepeatText(value) {
  return cleanMemoryText(value)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenizeImmediateRepeatText(value) {
  return Array.from(
    new Set(
      normalizeImmediateRepeatText(value)
        .split(/\s+/)
        .filter(
          (word) =>
            word.length >= IMMEDIATE_REPEAT_GUARD_TUNING.tokenLengthMin &&
            !MEMORY_MATCH_STOPWORDS.has(word)
        )
    )
  );
}

function buildImmediateRepeatPhraseSet(
  value,
  size = IMMEDIATE_REPEAT_GUARD_TUNING.phraseSize
) {
  const words = normalizeImmediateRepeatText(value)
    .split(/\s+/)
    .filter(Boolean);
  const phrases = new Set();

  if (words.length === 0) {
    return phrases;
  }

  if (words.length < size) {
    phrases.add(words.join(' '));
    return phrases;
  }

  for (let index = 0; index <= words.length - size; index += 1) {
    phrases.add(words.slice(index, index + size).join(' '));
  }

  return phrases;
}

function computeImmediateRepeatOverlap(left, right) {
  if (!left.size || !right.size) {
    return 0;
  }

  let shared = 0;

  for (const item of left) {
    if (right.has(item)) {
      shared += 1;
    }
  }

  return shared / Math.min(left.size, right.size);
}

function assessImmediateRejectedReplySimilarity(
  candidate,
  rejected,
  repeatGuard = 'avoidNearRepeat'
) {
  const normalizedCandidate = normalizeImmediateRepeatText(candidate);
  const normalizedRejected = normalizeImmediateRepeatText(rejected);

  if (!normalizedCandidate || !normalizedRejected) {
    return {
      isTooSimilar: false,
      reason: '',
      tokenOverlap: 0,
      phraseOverlap: 0,
    };
  }

  const exactMatch = normalizedCandidate === normalizedRejected;
  const containsOther =
    normalizedCandidate.includes(normalizedRejected) ||
    normalizedRejected.includes(normalizedCandidate);
  const tokenOverlap = computeImmediateRepeatOverlap(
    new Set(tokenizeImmediateRepeatText(normalizedCandidate)),
    new Set(tokenizeImmediateRepeatText(normalizedRejected))
  );
  const phraseOverlap = computeImmediateRepeatOverlap(
    buildImmediateRepeatPhraseSet(normalizedCandidate),
    buildImmediateRepeatPhraseSet(normalizedRejected)
  );
  const isExactMode = repeatGuard === 'avoidExact';
  const isTooSimilar =
    exactMatch ||
    containsOther ||
    tokenOverlap >=
      (isExactMode
        ? IMMEDIATE_REPEAT_GUARD_TUNING.exactTokenOverlapThreshold
        : IMMEDIATE_REPEAT_GUARD_TUNING.nearTokenOverlapThreshold) ||
    phraseOverlap >=
      (isExactMode
        ? IMMEDIATE_REPEAT_GUARD_TUNING.exactPhraseOverlapThreshold
        : IMMEDIATE_REPEAT_GUARD_TUNING.nearPhraseOverlapThreshold);

  let reason = '';

  if (exactMatch) {
    reason = 'normalized text matched exactly';
  } else if (containsOther) {
    reason = 'one reply substantially contained the other';
  } else if (
    tokenOverlap >=
    (isExactMode
      ? IMMEDIATE_REPEAT_GUARD_TUNING.exactTokenOverlapThreshold
      : IMMEDIATE_REPEAT_GUARD_TUNING.nearTokenOverlapThreshold)
  ) {
    reason = `token overlap was ${tokenOverlap.toFixed(2)}`;
  } else if (
    phraseOverlap >=
    (isExactMode
      ? IMMEDIATE_REPEAT_GUARD_TUNING.exactPhraseOverlapThreshold
      : IMMEDIATE_REPEAT_GUARD_TUNING.nearPhraseOverlapThreshold)
  ) {
    reason = `phrase overlap was ${phraseOverlap.toFixed(2)}`;
  }

  return {
    isTooSimilar,
    reason,
    tokenOverlap,
    phraseOverlap,
  };
}

function findBlockedReplySimilarity(
  candidate,
  blockedReplyTexts,
  repeatGuard = 'avoidNearRepeat'
) {
  for (const blockedReplyText of Array.isArray(blockedReplyTexts)
    ? blockedReplyTexts
    : []) {
    const similarity = assessImmediateRejectedReplySimilarity(
      candidate,
      blockedReplyText,
      repeatGuard
    );

    if (similarity.isTooSimilar) {
      return {
        ...similarity,
        blockedReplyText,
      };
    }
  }

  return null;
}

function buildImmediateNoReuseOverrideBlock(
  previousAssistantReply,
  signals,
  similarity
) {
  const items = [
    'The immediately previous reply just got corrected, rejected, or called repetitive.',
  ];
  const rejectedReply = clipImmediateReplyText(previousAssistantReply, 260);

  if (signals?.repeatGuard === 'avoidExact') {
    items.push(
      'Do not reuse the same joke, punchline, phrasing, framing, or suggestion in lighter edits.'
    );
  } else {
    items.push(
      'Do not stay in the same line of thought, setup, phrasing, or solution shape. Make the replacement materially different.'
    );
  }

  if (rejectedReply) {
    items.push(`Rejected previous reply: ${rejectedReply}`);
  }

  if (
    signals?.clarificationOverride !== 'none' ||
    signals?.interpretationReplacement
  ) {
    items.push(
      'The previous draft was still carrying the stale interpretation. Drop it and answer from the clarified meaning instead.'
    );
  }

  if (similarity?.reason) {
    items.push(`The first draft was still too close because ${similarity.reason}.`);
  }

  items.push(
    'A brief natural acknowledgment is enough, then pivot into genuinely different content.'
  );

  return `NO-REUSE OVERRIDE:\n${items.map((item) => `- ${item}`).join('\n')}`;
}

function shouldIncludeIdentityMemory(signals, relevantBlockCount = 0) {
  if (signals?.shouldThrottleHeavyMemory) {
    return false;
  }

  const wantsIdentity = Boolean(
    signals?.hasSpecificProjectTag ||
      signals?.wantsWorkContext ||
      signals?.wantsRelationshipContext ||
      signals?.wantsBuildContext
  );

  if (!wantsIdentity) {
    return false;
  }

  if (signals?.memoryExpression === 'explicit') {
    return true;
  }

  if (signals?.memoryExpression === 'selectiveExplicit') {
    return relevantBlockCount < 2;
  }

  return relevantBlockCount === 0;
}

function buildRelevantMemoryBlocks(memory, signals) {
  const profileItems = profileSummaryParagraphs(memory);
  const blocks = [];
  const used = new Set();

  function pushBlock(title, items) {
    const nextItems = items.filter((item) => {
      const key = cleanMemoryText(item).toLowerCase();

      if (!key || used.has(key)) {
        return false;
      }

      used.add(key);
      return true;
    });

    if (nextItems.length) {
      blocks.push(formatSelectedMemoryBlock(title, nextItems));
    }
  }

  const workItems = signals.wantsWorkContext
    ? mergeDistinctItems(
        pickRelevantItems(memory.lifeContext, signals, WORK_CONTEXT_KEYWORDS, 2, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        }),
        pickRelevantItems(memory.coreMemories, signals, WORK_CONTEXT_KEYWORDS, 2, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        }),
        pickRelevantItems(memory.bigProjects, signals, WORK_CONTEXT_KEYWORDS, 1, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        }),
        pickRelevantItems(profileItems, signals, WORK_CONTEXT_KEYWORDS, 1, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        })
      ).slice(0, 3)
    : [];

  pushBlock('WORK BACKGROUND', workItems);

  const relationshipItems = signals.wantsRelationshipContext
    ? mergeDistinctItems(
        pickRelevantItems(memory.lifeContext, signals, RELATIONSHIP_CONTEXT_KEYWORDS, 2, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        }),
        pickRelevantItems(memory.coreMemories, signals, RELATIONSHIP_CONTEXT_KEYWORDS, 2, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        }),
        pickRelevantItems(profileItems, signals, RELATIONSHIP_CONTEXT_KEYWORDS, 1, {
          minimumScore: TARGETED_MEMORY_MIN_SCORE,
          requireKeywordHit: true,
        })
      ).slice(0, 3)
    : [];

  pushBlock('RELATIONSHIP BACKGROUND', relationshipItems);

  const buildItems =
    signals.wantsBuildContext || signals.hasSpecificProjectTag
      ? mergeDistinctItems(
          pickRelevantItems(memory.bigProjects, signals, BUILD_CONTEXT_KEYWORDS, 3, {
            minimumScore: TARGETED_MEMORY_MIN_SCORE,
            requireKeywordHit: true,
          }),
          pickRelevantItems(memory.lifeContext, signals, BUILD_CONTEXT_KEYWORDS, 1, {
            minimumScore: TARGETED_MEMORY_MIN_SCORE,
            requireKeywordHit: true,
          }),
          pickRelevantItems(memory.coreMemories, signals, BUILD_CONTEXT_KEYWORDS, 1, {
            minimumScore: TARGETED_MEMORY_MIN_SCORE,
            requireKeywordHit: true,
          }),
          pickRelevantItems(profileItems, signals, BUILD_CONTEXT_KEYWORDS, 1, {
            minimumScore: TARGETED_MEMORY_MIN_SCORE,
            requireKeywordHit: true,
          })
        ).slice(0, 4)
      : [];

  pushBlock('PROJECT BACKGROUND', buildItems);

  if (blocks.length === 0 && !signals.shouldThrottleHeavyMemory) {
    const generalItems = mergeDistinctItems(
      pickRelevantItems(profileItems, signals, [], 2, {
        minimumScore: GENERAL_MEMORY_MIN_SCORE,
      }),
      pickRelevantItems(memory.coreMemories, signals, [], 2, {
        minimumScore: GENERAL_MEMORY_MIN_SCORE,
      }),
      pickRelevantItems(memory.lifeContext, signals, [], 2, {
        minimumScore: GENERAL_MEMORY_MIN_SCORE,
      }),
      pickRelevantItems(memory.bigProjects, signals, [], 2, {
        minimumScore: GENERAL_MEMORY_MIN_SCORE,
      })
    ).slice(0, 4);

    pushBlock('BACKGROUND THAT MAY HELP HERE', generalItems);
  }

  if (signals.allowsLowPriorityReferenceMemory && !signals.shouldThrottleHeavyMemory) {
    const referenceItems = mergeDistinctItems(
      pickRelevantItems(
        profileSummaryParagraphs(memory, { includeLowPriority: true }),
        signals,
        [],
        2,
        { allowLowPriority: true, minimumScore: REFERENCE_MEMORY_MIN_SCORE }
      ),
      pickRelevantItems(memory.coreMemories, signals, [], 2, {
        allowLowPriority: true,
        minimumScore: REFERENCE_MEMORY_MIN_SCORE,
      }),
      pickRelevantItems(memory.preferences, signals, [], 2, {
        allowLowPriority: true,
        minimumScore: REFERENCE_MEMORY_MIN_SCORE,
      }),
      pickRelevantItems(memory.doNotDo, signals, [], 2, {
        allowLowPriority: true,
        minimumScore: REFERENCE_MEMORY_MIN_SCORE,
      }),
      pickRelevantItems(memory.lifeContext, signals, [], 2, {
        allowLowPriority: true,
        minimumScore: REFERENCE_MEMORY_MIN_SCORE,
      })
    )
      .filter((item) => isLowPriorityReference(item))
      .slice(0, 2);

    pushBlock('REFERENCE DETAIL IF USEFUL', referenceItems);
  }

  return blocks;
}

function buildRunMemoryContext(memory, { packet = '', projectTag = 'General' } = {}) {
  return buildRunMemorySections(memory, { packet, projectTag })
    .map((section) => formatSelectedMemoryBlock(section.title, section.items))
    .join('\n\n');
}

function extractPacketField(packet, label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`${escaped}:\\s*\\n?([\\s\\S]*?)(?:\\n\\n|$)`, 'i');
  const match = String(packet || '').match(regex);
  return match ? summarizeText(match[1].trim(), 120) : '';
}

function extractPacketTailField(packet, label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`${escaped}:\\s*\\n?([\\s\\S]*)$`, 'i');
  const match = String(packet || '').match(regex);
  return match ? cleanMemoryText(match[1]) : '';
}

function makeRunRecord({
  at,
  status,
  responseId = null,
  packet = '',
  output = '',
  error = '',
  projectTag = 'General',
  threadId = '',
  blockedReplyExcerpt = '',
}) {
  return {
    id: responseId || `run-${Date.now()}`,
    at,
    status,
    responseId,
    projectTag: normalizeProjectTag(projectTag),
    threadId: cleanMemoryText(threadId),
    mode: extractPacketField(packet, 'MODE'),
    domain: extractPacketField(packet, 'DOMAIN'),
    packetSummary: buildStoredSummary(packet, 220),
    responseSummary: buildStoredSummary(output, 220),
    responseExcerpt: buildStoredSummary(output, 420),
    blockedReplyExcerpt: buildStoredSummary(blockedReplyExcerpt, 420),
    error: error ? summarizeText(error, 300) : '',
  };
}

const MAX_STORED_RUNS = 24;
const MAX_STORED_FEEDBACK = 40;

function normalizeHistoryKeyPart(value) {
  return String(value || '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function buildRunRecordKey(run) {
  return [
    normalizeHistoryKeyPart(run?.status),
    normalizeHistoryKeyPart(run?.projectTag),
    normalizeHistoryKeyPart(run?.threadId),
    normalizeHistoryKeyPart(run?.mode),
    normalizeHistoryKeyPart(run?.domain),
    normalizeHistoryKeyPart(run?.packetSummary),
    normalizeHistoryKeyPart(run?.responseSummary),
    normalizeHistoryKeyPart(run?.blockedReplyExcerpt),
    normalizeHistoryKeyPart(run?.error),
  ].join('||');
}

function compactStoredRuns(runs, limit = MAX_STORED_RUNS) {
  const output = [];
  const seen = new Set();

  for (const run of Array.isArray(runs) ? runs : []) {
    if (!run || typeof run !== 'object') {
      continue;
    }

    const key = buildRunRecordKey(run);

    if (!key || seen.has(key)) {
      continue;
    }

    seen.add(key);
    output.push(run);

    if (output.length >= limit) {
      break;
    }
  }

  return output;
}

function buildFeedbackRecordKey(item) {
  const runId = normalizeHistoryKeyPart(item?.runId);
  const rating = normalizeHistoryKeyPart(item?.rating);

  if (runId) {
    return `run:${runId}||rating:${rating}`;
  }

  return [
    `rating:${rating}`,
    `project:${normalizeHistoryKeyPart(item?.projectTag)}`,
    `packet:${normalizeHistoryKeyPart(item?.packetSummary)}`,
    `response:${normalizeHistoryKeyPart(item?.responseSummary)}`,
    `notes:${normalizeHistoryKeyPart(item?.notes)}`,
  ].join('||');
}

function compactStoredFeedback(feedback, limit = MAX_STORED_FEEDBACK) {
  const output = [];
  const seen = new Set();

  for (const item of Array.isArray(feedback) ? feedback : []) {
    if (!item || typeof item !== 'object') {
      continue;
    }

    const key = buildFeedbackRecordKey(item);

    if (!key || seen.has(key)) {
      continue;
    }

    seen.add(key);
    output.push(item);

    if (output.length >= limit) {
      break;
    }
  }

  return output;
}

function normalizeImportedMemory(raw) {
  return {
    profileSummary: String(raw.profileSummary || '').trim(),
    coreMemories: cleanArray(raw.coreMemories),
    preferences: cleanArray(raw.preferences),
    doNotDo: cleanArray(raw.doNotDo),
    lifeContext: cleanArray(raw.lifeContext),
    bigProjects: cleanArray(raw.bigProjects),
  };
}

function normalizePacketForm(raw) {
  const next = { ...EMPTY_PACKET_FORM };

  Object.keys(EMPTY_PACKET_FORM).forEach((key) => {
    next[key] = String(raw?.[key] ?? EMPTY_PACKET_FORM[key] ?? '').trim();
  });

  if (!next.constraints) {
    next.constraints = DEFAULT_CONSTRAINTS;
  }

  return next;
}

function incrementCounter(map, key) {
  if (!key) return;
  map[key] = (map[key] || 0) + 1;
}

function topCountsFromMap(map, limit = 8) {
  return Object.entries(map)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}

function buildAnalytics(memory) {
  const runs = Array.isArray(memory.runs) ? memory.runs : [];
  const feedback = Array.isArray(memory.feedback) ? memory.feedback : [];

  const successfulRuns = runs.filter((run) => run.status === 'success');
  const errorRuns = runs.filter((run) => run.status === 'error');
  const positiveFeedback = feedback.filter((item) => item.rating === 'up');
  const negativeFeedback = feedback.filter((item) => item.rating === 'down');

  const projectMap = {};
  const modeMap = {};
  const domainMap = {};

  for (const run of runs) {
    const project = normalizeProjectTag(run.projectTag);
    if (!projectMap[project]) {
      projectMap[project] = {
        projectTag: project,
        runs: 0,
        positiveFeedback: 0,
        negativeFeedback: 0,
      };
    }

    projectMap[project].runs += 1;

    if (run.mode) incrementCounter(modeMap, run.mode);
    if (run.domain) incrementCounter(domainMap, run.domain);
  }

  for (const item of feedback) {
    const project = normalizeProjectTag(item.projectTag);
    if (!projectMap[project]) {
      projectMap[project] = {
        projectTag: project,
        runs: 0,
        positiveFeedback: 0,
        negativeFeedback: 0,
      };
    }

    if (item.rating === 'up') projectMap[project].positiveFeedback += 1;
    if (item.rating === 'down') projectMap[project].negativeFeedback += 1;
  }

  const topProjects = Object.values(projectMap)
    .sort((a, b) => {
      const aTotal = a.runs + a.positiveFeedback + a.negativeFeedback;
      const bTotal = b.runs + b.positiveFeedback + b.negativeFeedback;
      return bTotal - aTotal;
    })
    .slice(0, 8);

  return {
    totalRuns: runs.length,
    successfulRuns: successfulRuns.length,
    errorRuns: errorRuns.length,
    totalFeedback: feedback.length,
    positiveFeedback: positiveFeedback.length,
    negativeFeedback: negativeFeedback.length,
    totalExports: Array.isArray(memory.exports) ? memory.exports.length : 0,
    totalBackups: Array.isArray(memory.backups) ? memory.backups.length : 0,
    topProjects,
    topModes: topCountsFromMap(modeMap, 8),
    topDomains: topCountsFromMap(domainMap, 8),
    bestFeedback: positiveFeedback
      .filter((item) => item.notes || item.responseSummary)
      .slice(0, 8),
    worstFeedback: negativeFeedback
      .filter((item) => item.notes || item.responseSummary)
      .slice(0, 8),
  };
}
function buildPlainTextBundle(bundle) {
  return `QUINNOS EXPORT BUNDLE

Bundle ID:
${bundle.id}

Exported At:
${bundle.exportedAt}

Session Name:
${bundle.sessionName}

Project Tag:
${bundle.projectTag}

Run ID:
${bundle.runId || ''}

Feedback Rating:
${bundle.feedbackRating || ''}

Feedback Note:
${bundle.feedbackNote || ''}

PACKET:
${bundle.packet || ''}

RESPONSE:
${bundle.responseText || ''}`;
}

function makeExportRecord({
  sessionName = '',
  projectTag = 'General',
  packet = '',
  responseText = '',
  feedbackRating = '',
  feedbackNote = '',
  runId = '',
}) {
  const bundle = {
    id: `bundle-${Date.now()}`,
    exportedAt: new Date().toISOString(),
    sessionName: String(sessionName || 'Untitled Session').trim() || 'Untitled Session',
    projectTag: normalizeProjectTag(projectTag),
    runId: String(runId || '').trim(),
    feedbackRating: String(feedbackRating || '').trim(),
    feedbackNote: String(feedbackNote || '').trim(),
    packet: String(packet || ''),
    responseText: String(responseText || ''),
  };

  return {
    ...bundle,
    plainTextBundle: buildPlainTextBundle(bundle),
    packetSummary: summarizeText(bundle.packet, 220),
    responseSummary: summarizeText(bundle.responseText, 220),
  };
}

function makeBackupRecord({ profileId, deviceLabel, appState }) {
  const safeState =
    appState && typeof appState === 'object'
      ? JSON.parse(JSON.stringify(appState))
      : {};

  const sessionName =
    String(safeState.sessionName || 'Untitled Session').trim() || 'Untitled Session';

  const projectTag = normalizeProjectTag(safeState.projectTag);
  const summarySeed =
    safeState.focusText ||
    safeState.lastRunPacket ||
    safeState.runResponse ||
    safeState.sessionName ||
    '';

  return {
    id: `backup-${Date.now()}`,
    savedAt: new Date().toISOString(),
    profileId: normalizeProfileId(profileId),
    deviceLabel: String(deviceLabel || '').trim() || 'Unnamed device',
    sessionName,
    projectTag,
    summary: summarizeText(summarySeed, 220),
    appState: safeState,
  };
}

async function readMemory() {
  await ensureMemoryFile();
  const raw = await fs.readFile(memoryPath, 'utf8');
  const parsed = JSON.parse(raw);

  return {
    ...DEFAULT_MEMORY,
    ...parsed,
    coreMemories: cleanArray(parsed.coreMemories),
    preferences: cleanArray(parsed.preferences),
    doNotDo: cleanArray(parsed.doNotDo),
    lifeContext: cleanArray(parsed.lifeContext),
    bigProjects: cleanArray(parsed.bigProjects),
    runs: Array.isArray(parsed.runs) ? parsed.runs : [],
    feedback: Array.isArray(parsed.feedback) ? parsed.feedback : [],
    exports: Array.isArray(parsed.exports) ? parsed.exports : [],
    backups: Array.isArray(parsed.backups) ? parsed.backups : [],
  };
}

async function writeMemory(memory) {
  await ensureMemoryFile();
  await writeTextFileAtomic(memoryPath, JSON.stringify(memory, null, 2));
}

async function proxyVoiceHealth() {
  const response = await fetch(`${VOICE_BASE_URL}/health`);
  const text = await response.text();

  return {
    status: response.status,
    contentType: response.headers.get('content-type'),
    body: text,
  };
}

async function proxyVoiceSpeak(
  text,
  {
    method = 'GET',
    previousText = '',
    nextText = '',
    prosodyHint = null,
  } = {}
) {
  const cleanProsodyHint = normalizeVoiceProsodyHint(prosodyHint);
  const response =
    method === 'POST'
      ? await fetch(`${VOICE_BASE_URL}/speak`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text,
            ...(previousText ? { previous_text: previousText } : {}),
            ...(nextText ? { next_text: nextText } : {}),
            ...(cleanProsodyHint ? { prosody_hint: cleanProsodyHint } : {}),
          }),
        })
      : await fetch(
          `${VOICE_BASE_URL}/speak?${new URLSearchParams({
            text,
            ...(previousText ? { previous_text: previousText } : {}),
            ...(nextText ? { next_text: nextText } : {}),
            ...buildVoiceProsodyQueryObject(cleanProsodyHint),
          }).toString()}`
        );
  const audioBuffer = Buffer.from(await response.arrayBuffer());

  return {
    status: response.status,
    contentType: response.headers.get('content-type') || 'audio/mpeg',
    contentLength: response.headers.get('content-length'),
    body: audioBuffer,
  };
}

async function sendVoiceAudioResponse(
  res,
  text,
  fallbackErrorLabel = 'Voice speak request failed.',
  {
    proxyMethod = 'GET',
    previousText = '',
    nextText = '',
    prosodyHint = null,
  } = {}
) {
  const cleanText = String(text || '').replace(/\s+/g, ' ').trim();
  const cleanPreviousText = String(previousText || '').replace(/\s+/g, ' ').trim();
  const cleanNextText = String(nextText || '').replace(/\s+/g, ' ').trim();
  const cleanProsodyHint = normalizeVoiceProsodyHint(prosodyHint);

  if (!cleanText) {
    return res.status(400).json({ ok: false, error: 'text is required' });
  }

  try {
    const proxied = await proxyVoiceSpeak(cleanText, {
      method: proxyMethod,
      previousText: cleanPreviousText,
      nextText: cleanNextText,
      prosodyHint: cleanProsodyHint,
    });

    res.status(proxied.status);

    if (proxied.contentType) {
      res.setHeader('Content-Type', proxied.contentType);
    }

    if (proxied.contentLength) {
      res.setHeader('Content-Length', proxied.contentLength);
    }

    if (proxied.status >= 200 && proxied.status < 300) {
      res.setHeader('Content-Disposition', 'inline; filename="quinn.mp3"');
      res.setHeader('Cache-Control', 'public, max-age=600, immutable');
    }

    return res.send(proxied.body);
  } catch (error) {
    if (shouldAllowLocalVoiceFallback()) {
      try {
        const audio = await generateElevenSpeech({
          text: cleanText,
          format: 'mp3',
          previousText: cleanPreviousText,
          nextText: cleanNextText,
          prosodyHint: cleanProsodyHint,
        });
        res.setHeader('Content-Type', 'audio/mpeg');
        res.setHeader('Content-Length', String(audio.length));
        res.setHeader('Content-Disposition', 'inline; filename="quinn.mp3"');
        res.setHeader('Cache-Control', 'public, max-age=600, immutable');
        return res.status(200).send(audio);
      } catch (fallbackError) {
        console.error('VOICE FALLBACK ERROR:', fallbackError);
      }
    }

    return res.status(500).json({
      ok: false,
      error: fallbackErrorLabel,
      details: error instanceof Error ? error.message : 'Unknown voice proxy error',
    });
  }
}

app.post('/voice-speak/prepare', cors(), (req, res) => {
  const token = createPreparedVoiceRequest({
    text: req.body?.text,
    previousText: req.body?.previous_text || req.body?.previousText,
    nextText: req.body?.next_text || req.body?.nextText,
    prosodyHint: req.body?.prosody_hint || req.body?.prosodyHint,
  });

  if (!token) {
    return res.status(400).json({
      ok: false,
      error: 'text is required',
    });
  }

  return res.status(200).json({
    ok: true,
    token,
    ttlMs: VOICE_PREPARED_REQUEST_TTL_MS,
  });
});

app.get('/voice-health', cors(), async (_req, res) => {
  try {
    const proxied = await proxyVoiceHealth();

    res.status(proxied.status);
    if (proxied.contentType) {
      res.setHeader('Content-Type', proxied.contentType);
    }

    return res.send(proxied.body);
  } catch (error) {
    if (shouldAllowLocalVoiceFallback()) {
      return res.json({
        ok: true,
        service: 'quinn-api-embedded-voice-fallback',
        provider: 'elevenlabs',
        mode: 'direct',
        hasElevenApiKey: Boolean(process.env.ELEVENLABS_API_KEY),
        hasVoiceId: Boolean(process.env.ELEVENLABS_VOICE_ID),
        hasModelId: Boolean(process.env.ELEVENLABS_MODEL_ID),
      });
    }

    return res.status(500).json({
      ok: false,
      error: 'Voice health check failed.',
      details: error instanceof Error ? error.message : 'Unknown voice proxy error',
    });
  }
});

app.get('/voice-speak', cors(), async (req, res) => {
  const prepared = getPreparedVoiceRequest(req.query.token);

  if (prepared) {
    return sendVoiceAudioResponse(res, prepared.text, 'Voice speak request failed.', {
      proxyMethod: 'POST',
      previousText: prepared.previousText,
      nextText: prepared.nextText,
      prosodyHint: prepared.prosodyHint,
    });
  }

  if (req.query.token) {
    return res.status(404).json({
      ok: false,
      error: 'Voice playback token expired or was not found.',
    });
  }

  return sendVoiceAudioResponse(res, req.query.text, 'Voice speak request failed.', {
    proxyMethod: 'GET',
    previousText: req.query.previous_text || req.query.previousText,
    nextText: req.query.next_text || req.query.nextText,
    prosodyHint: readVoiceProsodyHintFromQuery(req.query),
  });
});

app.get('/health', async (_req, res) => {
  try {
    const memory = await readMemory();
    const latestRun = memory.runs[0] || null;

    res.json({
      ok: true,
      status: 'QuinnOS backend is running',
      service: 'quinn-api',
      model,
      hasApiKey: Boolean(process.env.OPENAI_API_KEY),
      hasElevenKey: Boolean(process.env.ELEVENLABS_API_KEY),
      hasLocalElevenVoiceConfig: hasLocalElevenVoiceConfig(),
      voiceBaseUrl: VOICE_BASE_URL,
      voiceProxyOnly: VOICE_PROXY_ONLY,
      storageDir: runtimeDataDir,
      memoryFile: memoryPath,
      usingExternalStorage:
        path.resolve(memoryPath) !== path.resolve(bundledMemoryPath) ||
        path.resolve(runtimeDataDir) !== path.resolve(bundledDataDir),
      chainActive: Boolean(memory.lastResponseId),
      totalRuns: memory.runs.length,
      feedbackCount: memory.feedback.length,
      exportCount: memory.exports.length,
      backupCount: memory.backups.length,
      lastRunAt: latestRun?.at || null,
      lastRunStatus: latestRun?.status || null,
    });
  } catch {
    res.status(500).json({
      ok: false,
      error: 'Could not read backend status',
    });
  }
});

app.get('/memory', async (_req, res) => {
  try {
    const memory = await readMemory();
    res.json(memory);
  } catch {
    res.status(500).json({ error: 'Could not read memory file' });
  }
});

app.post('/memory', async (req, res) => {
  try {
    const current = await readMemory();

    const next = {
      ...current,
      profileSummary:
        typeof req.body.profileSummary === 'string'
          ? req.body.profileSummary
          : current.profileSummary,
      coreMemories:
        req.body.coreMemories !== undefined
          ? cleanArray(req.body.coreMemories)
          : current.coreMemories,
      preferences:
        req.body.preferences !== undefined
          ? cleanArray(req.body.preferences)
          : current.preferences,
      doNotDo:
        req.body.doNotDo !== undefined
          ? cleanArray(req.body.doNotDo)
          : current.doNotDo,
      lifeContext:
        req.body.lifeContext !== undefined
          ? cleanArray(req.body.lifeContext)
          : current.lifeContext,
      bigProjects:
        req.body.bigProjects !== undefined
          ? cleanArray(req.body.bigProjects)
          : current.bigProjects,
    };

    await writeMemory(next);
    res.json({ ok: true, memory: next });
  } catch {
    res.status(500).json({ error: 'Could not update memory file' });
  }
});

app.post('/memory/import', async (req, res) => {
  try {
    const rawText = String(req.body?.rawText || '').trim();

    if (!rawText) {
      return res.status(400).json({ error: 'rawText is required' });
    }

    const current = await readMemory();

    const completion = await client.chat.completions.create({
      model,
      messages: [
        {
          role: 'system',
          content:
            'You extract durable, reusable long-term memory for a personalized assistant. Return only valid JSON that matches the schema exactly.',
        },
        {
          role: 'user',
          content: `Convert the raw source text below into structured Quinn memory.

Rules:
- Keep it concise but useful.
- Prefer stable, reusable facts over one-off details.
- Keep each array item short and specific.
- If something is missing, leave it empty.

RAW SOURCE TEXT:

${rawText}`,
        },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'quinn_memory_import',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              profileSummary: { type: 'string' },
              coreMemories: { type: 'array', items: { type: 'string' } },
              preferences: { type: 'array', items: { type: 'string' } },
              doNotDo: { type: 'array', items: { type: 'string' } },
              lifeContext: { type: 'array', items: { type: 'string' } },
              bigProjects: { type: 'array', items: { type: 'string' } },
            },
            required: [
              'profileSummary',
              'coreMemories',
              'preferences',
              'doNotDo',
              'lifeContext',
              'bigProjects',
            ],
          },
        },
      },
    });

    const content = completion.choices?.[0]?.message?.content || '{}';
    const importedRaw = JSON.parse(content);
    const imported = normalizeImportedMemory(importedRaw);

    const next = {
      ...current,
      profileSummary: mergeSummary(current.profileSummary, imported.profileSummary),
      coreMemories: uniqueMerge(current.coreMemories, imported.coreMemories),
      preferences: uniqueMerge(current.preferences, imported.preferences),
      doNotDo: uniqueMerge(current.doNotDo, imported.doNotDo),
      lifeContext: uniqueMerge(current.lifeContext, imported.lifeContext),
      bigProjects: uniqueMerge(current.bigProjects, imported.bigProjects),
    };

    await writeMemory(next);

    res.json({
      ok: true,
      imported,
      memory: next,
      importSummary: `Imported ${imported.coreMemories.length} core memories, ${imported.preferences.length} preferences, ${imported.doNotDo.length} do-not-do rules, ${imported.lifeContext.length} life context items, and ${imported.bigProjects.length} projects.`,
    });
  } catch (error) {
    console.error('MEMORY IMPORT ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not import memory',
    });
  }
});

app.post('/feedback', async (req, res) => {
  try {
    const rating = String(req.body?.rating || '').trim();
    const notes = String(req.body?.notes || '').trim();
    const responseText = String(req.body?.responseText || '').trim();
    const packet = String(req.body?.packet || '').trim();
    const runId = String(req.body?.runId || '').trim();
    const projectTag = normalizeProjectTag(req.body?.projectTag);

    if (!rating || !['up', 'down'].includes(rating)) {
      return res.status(400).json({ error: 'rating must be "up" or "down"' });
    }

    if (!responseText) {
      return res.status(400).json({ error: 'responseText is required' });
    }

    const memory = await readMemory();
    const now = new Date().toISOString();

    const entry = {
      id: `feedback-${Date.now()}`,
      at: now,
      rating,
      runId: runId || null,
      projectTag,
      notes,
      responseSummary: buildStoredSummary(responseText, 220),
      packetSummary: buildStoredSummary(packet, 220),
    };

    memory.feedback = compactStoredFeedback([entry, ...(memory.feedback || [])]);

    await writeMemory(memory);

    res.json({
      ok: true,
      feedback: entry,
      totalFeedback: memory.feedback.length,
    });
  } catch (error) {
    console.error('FEEDBACK ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not save feedback',
    });
  }
});

app.get('/feedback', async (_req, res) => {
  try {
    const memory = await readMemory();
    res.json({
      ok: true,
      feedback: memory.feedback || [],
    });
  } catch {
    res.status(500).json({ error: 'Could not read feedback history' });
  }
});

app.get('/analytics', async (_req, res) => {
  try {
    const memory = await readMemory();
    res.json({
      ok: true,
      analytics: buildAnalytics(memory),
    });
  } catch (error) {
    console.error('ANALYTICS ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not load analytics',
    });
  }
});
app.post('/export-bundle', async (req, res) => {
  try {
    const sessionName = String(req.body?.sessionName || '').trim();
    const projectTag = normalizeProjectTag(req.body?.projectTag);
    const packet = String(req.body?.packet || '').trim();
    const responseText = String(req.body?.responseText || '').trim();
    const feedbackRating = String(req.body?.feedbackRating || '').trim();
    const feedbackNote = String(req.body?.feedbackNote || '').trim();
    const runId = String(req.body?.runId || '').trim();

    if (!packet && !responseText) {
      return res.status(400).json({
        error: 'packet or responseText is required',
      });
    }

    const memory = await readMemory();

    const bundle = makeExportRecord({
      sessionName,
      projectTag,
      packet,
      responseText,
      feedbackRating,
      feedbackNote,
      runId,
    });

    memory.exports = [bundle, ...(memory.exports || [])].slice(0, 50);

    await writeMemory(memory);

    res.json({
      ok: true,
      bundle,
      totalExports: memory.exports.length,
    });
  } catch (error) {
    console.error('EXPORT BUNDLE ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not create export bundle',
    });
  }
});

app.get('/exports', async (_req, res) => {
  try {
    const memory = await readMemory();
    res.json({
      ok: true,
      exports: memory.exports || [],
    });
  } catch (error) {
    console.error('EXPORTS ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not load export history',
    });
  }
});

app.post('/backup/push', async (req, res) => {
  try {
    const profileId = normalizeProfileId(req.body?.profileId);
    const deviceLabel = String(req.body?.deviceLabel || '').trim();
    const appState = req.body?.appState;

    if (!profileId) {
      return res.status(400).json({ error: 'profileId is required' });
    }

    if (!appState || typeof appState !== 'object') {
      return res.status(400).json({ error: 'appState is required' });
    }

    const memory = await readMemory();

    const backup = makeBackupRecord({
      profileId,
      deviceLabel,
      appState,
    });

    memory.backups = [backup, ...(memory.backups || [])].slice(0, 40);

    await writeMemory(memory);

    const { appState: _ignored, ...meta } = backup;

    res.json({
      ok: true,
      backup: meta,
      totalBackups: memory.backups.length,
    });
  } catch (error) {
    console.error('BACKUP PUSH ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not push backup',
    });
  }
});

app.get('/backups', async (req, res) => {
  try {
    const profileId = normalizeProfileId(req.query?.profileId);
    const memory = await readMemory();

    const filtered = profileId
      ? (memory.backups || []).filter((item) => item.profileId === profileId)
      : memory.backups || [];

    const backups = filtered.map(({ appState, ...meta }) => meta);

    res.json({
      ok: true,
      backups,
    });
  } catch (error) {
    console.error('BACKUPS LIST ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not load backups',
    });
  }
});

app.post('/backup/pull', async (req, res) => {
  try {
    const profileId = normalizeProfileId(req.body?.profileId);
    const backupId = String(req.body?.backupId || '').trim();

    if (!profileId) {
      return res.status(400).json({ error: 'profileId is required' });
    }

    const memory = await readMemory();
    const pool = (memory.backups || []).filter((item) => item.profileId === profileId);

    if (pool.length === 0) {
      return res.status(404).json({ error: 'No backups found for this profile' });
    }

    const backup = backupId
      ? pool.find((item) => item.id === backupId)
      : pool[0];

    if (!backup) {
      return res.status(404).json({ error: 'Requested backup was not found' });
    }

    res.json({
      ok: true,
      backup,
    });
  } catch (error) {
    console.error('BACKUP PULL ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not pull backup',
    });
  }
});

app.post('/followup-packet', async (req, res) => {
  try {
    const responseText = String(req.body?.responseText || '').trim();
    const currentPacket = String(req.body?.currentPacket || '').trim();
    const projectTag = normalizeProjectTag(req.body?.projectTag);

    if (!responseText) {
      return res.status(400).json({ error: 'responseText is required' });
    }

    const memory = await readMemory();
    const memorySections = buildRunMemorySections(memory, {
      packet: currentPacket || responseText,
      projectTag,
    });
    const memoryBlock = memorySections
      .map((section) => formatSelectedMemoryBlock(section.title, section.items))
      .join('\n\n');

    const completion = await client.chat.completions.create({
      model,
      messages: [
        {
          role: 'system',
          content:
            'You turn a QuinnOS response into the strongest next QuinnOS packet. Use the current packet and selected memory to continue the thought intelligently. Return only valid JSON that matches the schema exactly.',
        },
        {
          role: 'user',
          content: `SELECTED QUINN MEMORY FOR THIS FOLLOW-UP

${memoryBlock}

CURRENT QUINNOS PACKET

${currentPacket || 'None provided.'}

LATEST QUINNOS RESPONSE

${responseText}

Create the strongest next QuinnOS packet based on the latest response.

Rules:
- Make it a smart next-step packet, not a repeat of the previous one.
- Keep it tailored and specific.
- Keep it useful for immediate action.
- Continue the live line of thought if the current packet is clearly part of one.
- Do not flatten the next packet into generic advice or a vague reflection prompt.
- Put durable default rules into constraints.`,
        },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'quinn_followup_packet',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              sessionName: { type: 'string' },
              focusText: { type: 'string' },
              summary: { type: 'string' },
              form: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  mode: { type: 'string' },
                  domain: { type: 'string' },
                  ask: { type: 'string' },
                  task: { type: 'string' },
                  output: { type: 'string' },
                  toneRisk: { type: 'string' },
                  timebox: { type: 'string' },
                  stop: { type: 'string' },
                  context: { type: 'string' },
                  facts: { type: 'string' },
                  feelings: { type: 'string' },
                  nonNegotiables: { type: 'string' },
                  constraints: { type: 'string' },
                  success: { type: 'string' },
                  iDid: { type: 'string' },
                  itCaused: { type: 'string' },
                },
                required: [
                  'mode',
                  'domain',
                  'ask',
                  'task',
                  'output',
                  'toneRisk',
                  'timebox',
                  'stop',
                  'context',
                  'facts',
                  'feelings',
                  'nonNegotiables',
                  'constraints',
                  'success',
                  'iDid',
                  'itCaused',
                ],
              },
            },
            required: ['sessionName', 'focusText', 'summary', 'form'],
          },
        },
      },
    });

    const content = completion.choices?.[0]?.message?.content || '{}';
    const parsed = JSON.parse(content);

    const followUp = {
      sessionName: String(parsed.sessionName || 'Follow-Up Packet').trim(),
      focusText: String(
        parsed.focusText || 'Use this follow-up packet to keep momentum.'
      ).trim(),
      summary: String(parsed.summary || '').trim(),
      form: normalizePacketForm(parsed.form || {}),
    };

    res.json({
      ok: true,
      followUp,
    });
  } catch (error) {
    console.error('FOLLOW-UP ERROR:', error);
    res.status(500).json({
      error: error?.message || 'Could not generate follow-up packet',
    });
  }
});

app.get('/runs', async (_req, res) => {
  try {
    const memory = await readMemory();
    res.json({
      ok: true,
      runs: memory.runs || [],
    });
  } catch {
    res.status(500).json({ error: 'Could not read run history' });
  }
});

app.post('/reset-run-chain', async (_req, res) => {
  try {
    const memory = await readMemory();
    memory.lastResponseId = null;
    await writeMemory(memory);

    res.json({
      ok: true,
      message: 'Run chain reset',
    });
  } catch {
    res.status(500).json({ error: 'Could not reset run chain' });
  }
});

app.post('/run', async (req, res) => {
  const now = new Date().toISOString();

  try {
    const { packet, prompt } = req.body || {};
    const projectTag = normalizeProjectTag(req.body?.projectTag);
    const previousAssistantReply = cleanMemoryText(
      req.body?.previousAssistantReply || ''
    );
    const threadId = cleanMemoryText(req.body?.threadId || '');

    if (!packet || !String(packet).trim()) {
      return res.status(400).json({ error: 'packet is required' });
    }

    const memory = await readMemory();
    const packetSignals = buildPacketSignals(packet, projectTag);
    const shouldCompareAgainstPreviousReply = Boolean(previousAssistantReply);
    const recentBlockedReplyTexts = findRecentBlockedReplyTexts(memory.runs, threadId);
    const blockedReplyCandidates = takeDistinctItems(
      [previousAssistantReply, ...recentBlockedReplyTexts],
      MEMORY_CONTEXT_TUNING.recentBlockedReplyLookback + 1
    );
    const memorySections = buildRunMemorySections(memory, {
      packet,
      projectTag,
      previousAssistantReply,
      threadId,
    });
    const memoryBlock = memorySections
      .map((section) => formatSelectedMemoryBlock(section.title, section.items))
      .join('\n\n');

    const trimmedMemoryBlock = String(memoryBlock || '').slice(0, 3000);
    const trimmedPacket = String(packet || '').slice(0, 2200);
    const trimmedPreviousAssistantReply =
      shouldCompareAgainstPreviousReply &&
      (packetSignals.clarificationOverride !== 'none' ||
        packetSignals.interpretationReplacement ||
        packetSignals.correctionLatch !== 'none' ||
        packetSignals.constraintPriority !== 'none' ||
        packetSignals.repeatGuard !== 'none' ||
        recentBlockedReplyTexts.length > 0)
        ? clipImmediateReplyText(previousAssistantReply)
        : '';
    const recentRejectedReplyBlock = recentBlockedReplyTexts.length
      ? `RECENTLY REJECTED LOCAL MATERIAL

${recentBlockedReplyTexts
  .slice(0, 2)
  .map((item) => `- ${clipImmediateReplyText(item, 220)}`)
  .join('\n')}`
      : '';
    const trimmedPrompt = String(
      prompt ||
        'Reply like another me in the same headspace. First notice whether the note is exploratory, conflicted, riffing, casually talking, or actually asking for a move. If it is exploratory or just talking, stay with it and bounce the thought back instead of solving too fast. If the thought is still discovering itself, build with it instead of compressing it into a smaller cleaner answer. If it clearly wants advice or a plan, then be direct and useful. Let the same Quinn voice also show more texture when it fits: drier, warmer, more amused, more blunt, more lightly exasperated, or more locked into the idea, without turning into a different persona. If the latest note is correcting or invalidating the previous move, pivot with it instead of continuing the old frame. If a new blocker shows up, let feasibility override the earlier hype or suggestion. If repetition just got called out, do not reuse the same joke, premise, or phrasing. React to the real thing first, stay prose-first, and if help was not asked for, do not tack suggestions, next moves, or a useful reframe onto the ending. Let memory change what you assume, skip, sharpen, and emphasize without narrating the remembering process. If the note is dressing something up and the signal is strong, do not buy the spin. Let the ending stop where the point actually lands instead of sounding like a completed response unit.'
    ).slice(0, 500);

    const instructions = [
      'You are Quinn in this app: like another version of the user thinking back from inside the same headspace — familiar, fast, sharp, and emotionally accurate.',
      'Do not feel like an assistant, advisor, coach, therapist, or guide. Feel like another mind in the same perspective: already in it, already getting it, already willing to say the real thing.',
      'Do not announce that stance or explain it. Just speak from it naturally.',
      'Use the current packet as the live thing being said right now.',
      'The already-known terrain below is there quietly if it genuinely helps.',
      'Answer the actual packet first.',
      'Use broad intelligence and general reasoning first. Use already-known terrain second, quietly, to sharpen the answer.',
      'Use the packet\'s memory-expression cue to decide whether memory should stay implicit, surface briefly, or be named directly. Default hard toward implicit.',
      'Let remembered context change what you assume, skip, sharpen, or prioritize without narrating the remembering process.',
      'Use remembered context to make the reply feel lived-in, not to decorate it or prove you remember things.',
      'Do not read the user’s life back to them like a profile or quote remembered facts back just to show continuity.',
      'Do not narrate memory, continuity, or internal mechanics.',
      'Only surface Quinn-specific details when they materially improve relevance, precision, emotional accuracy, or grounding.',
      'If memory must be surfaced, do it once, briefly, and naturally. Avoid phrases like "you previously said", "I remember that", "based on what I know about you", or "given your history" unless direct grounding is genuinely needed.',
      'If a detail is low-priority trivia or a recurring callback, leave it out unless the packet clearly makes it relevant.',
      'Match the user’s preferred voice: direct, personal, emotionally intelligent, specific, grounded, and high-context.',
      'Be sharp and natural. Use contractions. Sound like a real person texting back, not like a tool, coach, analyst, therapist, or memo.',
      'Use the packet\'s conductor cue as the final arbitration layer when energy, challenge, riff, ending, ask, memory, and texture pull in different directions.',
      'Let the conductor cue decide how much room the reply deserves, how hard structural contradiction or pattern-lock should be noticed, and whether recurring motifs should stay implicit.',
      'Use the packet\'s correction-latch cue as an immediate frame override. If the user is correcting, rejecting, or invalidating the last move, acknowledge that briefly and pivot instead of continuing the old momentum.',
      'Use the packet\'s constraint-priority cue to decide when a new blocker overrides desire, enthusiasm, or the earlier suggestion. When it is dominant, answer the blocker first.',
      'Use the packet\'s repeat-guard cue to avoid exact or near repeats right after the user calls one out. Replace the move with genuinely different content, not a warmed-over variation.',
      'When the user says some version of "I know, but", "that\'s not the point", "you missed it", or "you already said that", treat it as a live frame update rather than texture around the old frame.',
      'Brief acknowledgment is enough when correction is active. Do not get defensive or apologetic.',
      'Use the packet\'s polish cue as the final taste layer. Let it govern candidate framing, repetition restraint, warmth precision, micro-turn handling, aftertaste cleanup, and bounded surprise.',
      'Use the packet\'s energy match cue to shape cadence, sentence length, sharpness, softness, humor density, and directness. Do it implicitly. Never narrate the mood back to the user.',
      'Use the packet\'s personality texture cue to let the same Quinn voice get drier, slyer, warmer, blunter, more amused, more lightly exasperated, or more magnetized by the idea when it fits. This should feel like different facial expressions from the same person, not a persona switch.',
      'Let personality texture color cadence, phrasing, wit, warmth, and edge without becoming theatrical, random, or overperformed.',
      'If the note is still alive and branching, you may hold two or three candidate framings in the air inside natural prose. Never turn them into a menu, framework, or list unless the user explicitly asks for that structure.',
      'Use the packet\'s challenge stance to decide whether to stay neutral, lightly challenge, or push back directly. Let it sharpen truth-contact, not turn the reply into a debate.',
      'Use the packet\'s riff stance to decide whether the note wants resolve, co-building, or deep riffing. If it wants co-building or deep riffing, stay inside the live thought and help it keep becoming itself instead of rushing to answer-mode.',
      'Use the packet\'s ask policy cue to decide whether a question is actually wanted here. Default away from asking unless the question is genuinely useful, specific, and alive. Do not use a question mark as conversational life-support.',
      'Use the packet\'s ending shape cue to decide whether the reply should end open, sharp, nudging, cleanly stopped, or softly landed. Do not default to recap endings, assistant questions, or completion-signaling lines.',
      'Use the packet\'s warmth cue to decide whether the line should stay emotionally neutral-but-attentive, warm without sentimentality, fond, protective, intimate clean, or lightly caring-exasperated. Keep warmth specific and lived-in, never syrupy.',
      'Use the packet\'s micro-turn cue for tiny user turns. A tiny turn can carry hinge, pressure, or feeling; do not answer it like empty filler.',
      'Use the packet\'s signature drift cue to avoid reusing the same opener, landing, pattern-naming gesture, or wit shape just because it worked recently. Keep Quinn recognizable without leaving the same pawprint every turn.',
      'Use the packet\'s aftertaste cue as a small self-check. Strip out assistant residue, decorative questions, extra recap polish, overexplaining, or the wrong amount of bite before you land the reply.',
      'Use the packet\'s bounded-surprise cue to allow the occasional slightly riskier best move when it genuinely fits: shorter, sharper, funnier, more skeptical, or more tender than the safe option. One notch, not a stunt.',
      'Mirror energy with judgment, not obedience. Low notes want quieter precision. Intense notes want tighter steering. Playful notes can get more wit and associative movement. Raw notes can be plainer and more direct. Tender notes want gentler cadence and cleaner handling.',
      'When the signal is strong, be willing to question spin, euphemism, fake confusion, or dressed-up framing. Separate story from substance and say the plainer thing cleanly.',
      'Pushback should be earned, proportionate, and grounded in the actual note. Do not get hostile, smug, scolding, or contrarian for flavor.',
      'Do not flatten the voice just because the user is brief, and do not force humor into serious moments.',
      'Do not add a follow-up question just to keep the thread alive. Do not ask because silence feels unsafe, because you want engagement, or because a question would soften the landing. Do not restate the point one extra time for polish. A reply does not need to sound finished; it needs to stop at the right place.',
      'When the note is still exploratory, do not reward yourself for being useful by shrinking an alive thought into a cleaner but smaller answer.',
      'In co-build or deep-riff moments, name patterns, offer candidate framings, and use try-on phrasing without pretending the thought is settled.',
      'If you already understand what the user means, answer from there instead of packaging the context back to them. First decide whether the user wants exploration, pressure-testing, or an actual move. React before you organize.',
      'Let familiarity stay implicit. Do not perform closeness or identity.',
      'Treat ordinary prompts like ordinary conversation, not like a request for a guide, action plan, or formatted deliverable. If the packet is exploratory, conflicted, venting, thinking out loud, or casually sharing, stay in that mode instead of turning it into help by the end.',
      'Default to prose-first responses unless the user explicitly asks for bullets, numbered options, or step-by-step structure. Bounce the thought back, extend it, pressure-test it, or name the hidden tension, and only turn it into advice if the user is clearly asking for advice.',
      'Most replies should land as one to three short natural paragraphs.',
      'Short natural paragraphs beat frameworks.',
      'Do not drift into numbered lists, labeled frameworks, or checklist formatting unless the content truly benefits.',
      'Do not default to takeaways, next steps, options, or mini how-to guides when a real reply would do. Do not smooth every thought into a tidy answer if the honest reply should stay a little unresolved. Do not force closure just to feel useful, and do not append a helpful last paragraph unless help was clearly asked for.',
      'If two short paragraphs say it better than a list, use the paragraphs. If the user is not asking for help, let the ending stay in the conversation instead of pivoting to what might help.',
      'Avoid internal QuinnOS wording in the actual reply. Do not talk about signals, packets, resonance, stacks, vectors, frameworks, or operating layers unless the user explicitly wants that framing.',
      'Avoid symbolic formatting, slash-heavy phrasing, colon-heavy memo language, or anything that sounds like a dashboard.',
      'Do not moralize, do not over-explain, and do not give filler.',
      'Do not sound corporate, clinical, canned, or vaguely supportive.',
      'Do not sound therapist-y, productivity-coach-y, or like you are analyzing the user from a distance.',
      'Do not give generic self-help language, vague therapy-speak, or obvious AI phrasing.',
      'Skip polished assistant niceties unless the moment actually calls for them.',
      'Casual is fine when casual fits. Warm is fine when warmth fits. Do not make either one feel forced.',
      'Do not say "as an AI", "I can’t", or other assistant-disclaimer language unless absolutely necessary.',
      'If the packet asks for judgment, give judgment.',
      'If the packet clearly wants strategy, give the cleanest move, but say it like you actually mean it. Keep it instinctive, not consultant-y. If the packet clearly wants advice, options, or a plan, then be direct and useful.',
      'If the packet asks for writing, write the thing cleanly and fully instead of circling around it.',
      'Prefer concrete observations, instincts, reactions, and pushback over broad advice. Let the reply feel like idea-bouncing, not instant solutioning.',
      'Even when being useful, keep the voice conversational instead of instructional by default. You can disagree, sharpen the thought, riff a little, or stay in the tension if that is the truest reply.',
      'If the packet conflicts with stable memory, prefer the packet for this run.',
      'Keep the response vivid, fresh, and varied.',
      'Do not reuse recent motifs, jokes, foods, drinks, workplace callbacks, or signature references unless the user explicitly asks for them or the packet truly needs them.',
      'Do not default to recurring life details just because they are known.',
      'Being personalized does not mean being self-referential.',
      'When humor fits, make it dry, clever, and tailored — not random or try-hard.',
      'When warmth fits, make it intimate and perceptive — not cheesy or overly soft.',
      'When the user needs clarity, be clean and exact.',
      'Never end a concise spoken-style thought with an ellipsis or obviously clipped phrase.',
      'Finish thoughts cleanly.',
    ].join(' ');
    const buildRunInput = (overrideBlock = '') => {
      const input = [
        {
          role: 'user',
          content: `PROJECT TAG

${projectTag}`,
        },
        {
          role: 'user',
          content: `ALREADY KNOWN TERRAIN

${trimmedMemoryBlock}`,
        },
      ];

      if (trimmedPreviousAssistantReply) {
        input.push({
          role: 'user',
          content: `IMMEDIATELY PREVIOUS REPLY TO COMPARE AGAINST

${trimmedPreviousAssistantReply}`,
        });
      }

      if (recentRejectedReplyBlock) {
        input.push({
          role: 'user',
          content: recentRejectedReplyBlock,
        });
      }

      if (overrideBlock) {
        input.push({
          role: 'user',
          content: overrideBlock,
        });
      }

      input.push(
        {
          role: 'user',
          content: `THE LIVE NOTE TO RESPOND TO

${trimmedPacket}`,
        },
        {
          role: 'user',
          content: `REPLY STANCE

${trimmedPrompt}`,
        },
        {
          role: 'user',
          content: `DEFAULT FEEL

Give the real reply like you're texting me back from inside the same thought. First notice whether this wants exploration, simple conversation, or action. Let the packet\'s conductor cue settle conflicts between edge, tenderness, riff depth, question restraint, memory visibility, structure, and how much space the reply gets. Let the packet\'s correction and constraint cues decide whether the old momentum still counts or whether a blocker or correction has replaced it. If the user is correcting, rejecting, or invalidating the last move, acknowledge that briefly and pivot instead of continuing the old frame. If repetition just got called out, make the next move genuinely different. If the immediately previous reply is provided above, treat it as the exact local thing you may need to pivot away from or avoid reusing. Let the packet\'s polish cue handle the final taste of the reply: whether to hold one framing or a couple live framings, how much warmth is actually right, whether a micro-turn wants a small beat or a fast hinge, which repeated Quinn habits to avoid, what residue to strip out before landing, and whether one notch of surprise would make the line truer. Let the packet\'s energy cue set the texture of the reply without turning it into a performance. Let the packet\'s personality texture cue decide whether the same Quinn voice should stay steady, go a little dry, sly, affectionate, blunt, amused, lightly exasperated, or especially locked into the idea. Let it feel like the same person with different facial expressions, not a different character. Let the packet\'s challenge cue decide how much to push the framing, from none to clean direct pushback. Let the packet\'s riff cue decide whether to resolve, co-build, or stay in a deeper riff. Let the packet\'s memory-expression cue decide whether memory should stay implicit, surface briefly, or be named directly. Default to letting it stay implicit. Let the packet\'s ask-policy cue decide whether a question belongs at all. Default away from asking unless the question is genuinely useful, specific, and more alive than a clean statement. Let the packet\'s ending cue decide whether the last line should stay open, land sharp, give a tiny nudge, stop cleanly, or soften a little. If the conductor notices contradiction, standard shifts, conflation, pattern-lock, or recurring motifs, let that sharpen the framing without turning the reply into a diagnosis. If it is exploratory or just talking, keep it there for a beat instead of turning it into a guide. If it is still discovering itself, build with it. Treat known context like already-known terrain, not a fact list to recite. React first. Use natural prose. Only organize it if the note actually needs structure. When the signal is strong, prefer the plainer truth over the prettier explanation. Let the ending stay in recognition, reaction, or tension unless help was actually asked for. Stop where the point actually lands. Do not end on a dangling phrase, cliffhanger, ellipsis, or decorative follow-up question.`,
        }
      );

      return input;
    };

    const createRunResponse = (overrideBlock = '') =>
      client.responses.create({
        model,
        instructions,
        store: true,
        max_output_tokens: 1600,
        input: buildRunInput(overrideBlock),
      });

    const extractRunOutput = (response) =>
      (response?.output_text && response.output_text.trim()) ||
      JSON.stringify(response?.output ?? [], null, 2);

    const shouldEnforceNoReuse =
      blockedReplyCandidates.length > 0 &&
      (packetSignals.clarificationOverride !== 'none' ||
        packetSignals.interpretationReplacement ||
        packetSignals.repeatGuard !== 'none' ||
        packetSignals.correctionLatch !== 'none' ||
        packetSignals.constraintPriority !== 'none' ||
        recentBlockedReplyTexts.length > 0);
    const similarityGuardMode =
      packetSignals.repeatGuard !== 'none'
        ? packetSignals.repeatGuard
        : 'avoidNearRepeat';

    let response = null;
    let output = '';
    let overrideBlock = '';
    let lastSimilarity = {
      isTooSimilar: false,
      reason: '',
      tokenOverlap: 0,
      phraseOverlap: 0,
    };

    for (
      let attempt = 1;
      attempt <= (shouldEnforceNoReuse ? IMMEDIATE_REPEAT_GUARD_TUNING.maxAttempts : 1);
      attempt += 1
    ) {
      response = await createRunResponse(overrideBlock);
      output = extractRunOutput(response);

      if (!shouldEnforceNoReuse) {
        break;
      }

      const similarityMatch = findBlockedReplySimilarity(
        output,
        blockedReplyCandidates,
        similarityGuardMode
      );

      lastSimilarity = similarityMatch || {
        isTooSimilar: false,
        reason: '',
        tokenOverlap: 0,
        phraseOverlap: 0,
      };

      if (!lastSimilarity.isTooSimilar) {
        break;
      }

      overrideBlock = buildImmediateNoReuseOverrideBlock(
        similarityMatch?.blockedReplyText || previousAssistantReply,
        packetSignals,
        lastSimilarity
      );
    }

    if (
      shouldEnforceNoReuse &&
      lastSimilarity.isTooSimilar
    ) {
      console.warn('RUN NO-REUSE GUARD stayed close to rejected reply', {
        threadId,
        repeatGuard: packetSignals.repeatGuard,
        correctionLatch: packetSignals.correctionLatch,
        constraintPriority: packetSignals.constraintPriority,
        tokenOverlap: lastSimilarity.tokenOverlap,
        phraseOverlap: lastSimilarity.phraseOverlap,
        reason: lastSimilarity.reason,
      });
    }

    memory.lastResponseId = response.id;
    memory.runs = compactStoredRuns([
      makeRunRecord({
        at: now,
        status: 'success',
        responseId: response.id,
        packet,
        output,
        projectTag,
        threadId,
        blockedReplyExcerpt:
          previousAssistantReply &&
          (packetSignals.clarificationOverride !== 'none' ||
            packetSignals.interpretationReplacement ||
            packetSignals.repeatGuard !== 'none' ||
            packetSignals.correctionLatch !== 'none' ||
            packetSignals.constraintPriority !== 'none')
            ? previousAssistantReply
            : '',
      }),
      ...(memory.runs || []),
    ]);

    await writeMemory(memory);

    res.json({
      ok: true,
      output,
      responseId: response.id,
      ranAt: now,
      model,
      projectTag,
      threadId,
      memoryResonance: buildRunMemoryResonance(memorySections),
    });
  } catch (error) {
    console.error('RUN ERROR:', error);

    try {
      const memory = await readMemory();
      memory.runs = compactStoredRuns([
        makeRunRecord({
          at: now,
          status: 'error',
          packet: req.body?.packet || '',
          error: error?.message || 'Run failed',
          projectTag: req.body?.projectTag || 'General',
          threadId: req.body?.threadId || '',
        }),
        ...(memory.runs || []),
      ]);
      await writeMemory(memory);
    } catch {
      // ignore secondary logging error
    }

    res.status(500).json({
      error: error?.message || 'Run failed',
    });
  }
});

app.post(
  '/transcribe',
  (req, res, next) => {
    transcriptionUpload.single('audio')(req, res, (error) => {
      if (error) {
        return res.status(400).json({
          ok: false,
          error: error.message || 'Audio upload failed.',
        });
      }

      return next();
    });
  },
  handleTranscriptionRoute
);

app.get('/speak', async (req, res) => {
  return sendVoiceAudioResponse(res, req.query?.text, 'ElevenLabs speak failed', {
    proxyMethod: 'GET',
    previousText: req.query?.previous_text || req.query?.previousText,
    nextText: req.query?.next_text || req.query?.nextText,
    prosodyHint: readVoiceProsodyHintFromQuery(req.query),
  });
});

app.post('/speak', async (req, res) => {
  return sendVoiceAudioResponse(res, req.body?.text, 'ElevenLabs speak failed', {
    proxyMethod: 'POST',
    previousText: req.body?.previous_text || req.body?.previousText,
    nextText: req.body?.next_text || req.body?.nextText,
    prosodyHint: req.body?.prosody_hint || req.body?.prosodyHint,
  });
});

app.post('/tts/quinn', async (req, res) => {
  return sendVoiceAudioResponse(res, req.body?.text, 'ElevenLabs TTS failed', {
    proxyMethod: 'POST',
    previousText: req.body?.previous_text || req.body?.previousText,
    nextText: req.body?.next_text || req.body?.nextText,
    prosodyHint: req.body?.prosody_hint || req.body?.prosodyHint,
  });
});

app.listen(port, host, async () => {
  await ensureMemoryFile();
  console.log(`QuinnOS backend running on http://${host}:${port}`);
  console.log(`Model: ${model}`);
  console.log(`Voice base URL: ${VOICE_BASE_URL}`);
  console.log(`Memory file: ${memoryPath}`);
});







