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
    return 'IDENTITY MEMORY:\n- No identity memory saved yet';
  }

  return `IDENTITY MEMORY:\n${items
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
    ...preferred.slice(0, 6).map((item) => `DO: ${item}`),
    ...avoid.slice(0, 6).map((item) => `AVOID: ${item}`),
  ];

  if (items.length === 0) {
    return 'STYLE MEMORY:\n- No style memory saved yet';
  }

  return `STYLE MEMORY:\n${items
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
  'IDENTITY MEMORY': 'Identity memory',
  'STYLE MEMORY': 'Style memory',
  'RELEVANT WORK CONTEXT': 'Work context',
  'RELEVANT RELATIONSHIP / EMOTIONAL CONTEXT': 'Relationship context',
  'RELEVANT PROJECT / BUILD CONTEXT': 'Project context',
  'RELEVANT CONTEXT FOR THIS PACKET': 'Packet context',
  'RELEVANT REFERENCE MEMORY': 'Reference memory',
  'ANTI-REPETITION': 'Freshness guard',
};

const MEMORY_RESONANCE_PRIORITIES = {
  'RELEVANT WORK CONTEXT': 0,
  'RELEVANT RELATIONSHIP / EMOTIONAL CONTEXT': 0,
  'RELEVANT PROJECT / BUILD CONTEXT': 0,
  'RELEVANT CONTEXT FOR THIS PACKET': 1,
  'STYLE MEMORY': 2,
  'IDENTITY MEMORY': 3,
  'ANTI-REPETITION': 4,
  'RELEVANT REFERENCE MEMORY': 5,
};

function buildRunMemorySections(memory, { packet = '', projectTag = 'General' } = {}) {
  return [
    buildIdentityCapsule(memory),
    buildStyleCapsule(memory),
    ...buildRelevantMemoryBlocks(memory, packet, projectTag),
    buildAntiRepetitionBlock(memory.runs),
  ]
    .map(parseMemorySection)
    .filter(Boolean);
}

function buildRunMemoryResonance(sections) {
  return [...(Array.isArray(sections) ? sections : [])]
    .sort(
      (a, b) =>
        (MEMORY_RESONANCE_PRIORITIES[a.title] ?? 99) -
        (MEMORY_RESONANCE_PRIORITIES[b.title] ?? 99)
    )
    .slice(0, 4)
    .map((section) => ({
      label: MEMORY_RESONANCE_LABELS[section.title] || section.title,
      preview: summarizeText(
        String(section.items?.[0] || '').replace(/^(DO|AVOID):\s*/i, ''),
        96
      ),
    }))
    .filter((item) => item.label || item.preview);
}

function profileSummaryParagraphs(memory, { includeLowPriority = false } = {}) {
  return String(memory.profileSummary || '')
    .split(/\n\s*\n/)
    .map((part) => cleanMemoryText(part))
    .filter(Boolean)
    .filter((part) => includeLowPriority || !isLowPriorityReference(part));
}

function buildPacketSignals(packet, projectTag = 'General') {
  const domain = extractPacketField(packet, 'DOMAIN');
  const ask = extractPacketField(packet, 'ASK');
  const context = extractPacketField(packet, 'CONTEXT');
  const normalizedProjectTag = normalizeSearchText(projectTag);
  const sourceText = [packet, domain, ask, context, projectTag].filter(Boolean).join('\n');

  return {
    text: sourceText,
    tokens: tokenizeSearchTerms(sourceText),
    domain: normalizeSearchText(domain),
    projectTag: normalizedProjectTag,
    hasSpecificProjectTag: Boolean(
      normalizedProjectTag && normalizedProjectTag !== 'general'
    ),
    wantsWorkContext: countKeywordHits(sourceText, WORK_CONTEXT_KEYWORDS) > 0,
    wantsRelationshipContext:
      countKeywordHits(sourceText, RELATIONSHIP_CONTEXT_KEYWORDS) > 0,
    wantsBuildContext: countKeywordHits(sourceText, BUILD_CONTEXT_KEYWORDS) > 0,
    allowsLowPriorityReferenceMemory:
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

  return candidates
    .map((item) => ({
      item,
      score: scoreMemoryItem(item, signals, keywords, options),
    }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.item.length - b.item.length)
    .slice(0, limit)
    .map(({ item }) => item);
}

function buildAntiRepetitionBlock(runs) {
  const recentText = (Array.isArray(runs) ? runs : [])
    .slice(0, 8)
    .map((run) => cleanMemoryText(run.responseSummary || ''))
    .filter(Boolean)
    .join('\n');

  if (!recentText) {
    return 'ANTI-REPETITION:\n- Avoid leaning on familiar personal callbacks unless clearly relevant.';
  }

  const hits = RECENT_MOTIF_PATTERNS.filter(({ pattern }) => pattern.test(recentText)).map(
    ({ label }) => label
  );

  if (hits.length === 0) {
    return 'ANTI-REPETITION:\n- Avoid reusing recent phrasings, motifs, or callback details unless the packet clearly calls for them.';
  }

  return `ANTI-REPETITION:\n- Avoid reusing these recent motifs unless materially relevant: ${hits.join(', ')}`;
}

function buildRelevantMemoryBlocks(memory, packet, projectTag) {
  const signals = buildPacketSignals(packet, projectTag);
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
        pickRelevantItems(memory.lifeContext, signals, WORK_CONTEXT_KEYWORDS, 2),
        pickRelevantItems(memory.coreMemories, signals, WORK_CONTEXT_KEYWORDS, 2),
        pickRelevantItems(memory.bigProjects, signals, WORK_CONTEXT_KEYWORDS, 1),
        pickRelevantItems(profileItems, signals, WORK_CONTEXT_KEYWORDS, 1)
      ).slice(0, 3)
    : [];

  pushBlock('RELEVANT WORK CONTEXT', workItems);

  const relationshipItems = signals.wantsRelationshipContext
    ? mergeDistinctItems(
        pickRelevantItems(memory.lifeContext, signals, RELATIONSHIP_CONTEXT_KEYWORDS, 2),
        pickRelevantItems(memory.coreMemories, signals, RELATIONSHIP_CONTEXT_KEYWORDS, 2),
        pickRelevantItems(profileItems, signals, RELATIONSHIP_CONTEXT_KEYWORDS, 1)
      ).slice(0, 3)
    : [];

  pushBlock('RELEVANT RELATIONSHIP / EMOTIONAL CONTEXT', relationshipItems);

  const buildItems =
    signals.wantsBuildContext || signals.hasSpecificProjectTag
      ? mergeDistinctItems(
          pickRelevantItems(memory.bigProjects, signals, BUILD_CONTEXT_KEYWORDS, 3),
          pickRelevantItems(memory.lifeContext, signals, BUILD_CONTEXT_KEYWORDS, 1),
          pickRelevantItems(memory.coreMemories, signals, BUILD_CONTEXT_KEYWORDS, 1),
          pickRelevantItems(profileItems, signals, BUILD_CONTEXT_KEYWORDS, 1)
        ).slice(0, 4)
      : [];

  pushBlock('RELEVANT PROJECT / BUILD CONTEXT', buildItems);

  if (blocks.length === 0) {
    const generalItems = mergeDistinctItems(
      pickRelevantItems(profileItems, signals, [], 2),
      pickRelevantItems(memory.coreMemories, signals, [], 2),
      pickRelevantItems(memory.lifeContext, signals, [], 2),
      pickRelevantItems(memory.bigProjects, signals, [], 2)
    ).slice(0, 4);

    pushBlock('RELEVANT CONTEXT FOR THIS PACKET', generalItems);
  }

  if (signals.allowsLowPriorityReferenceMemory) {
    const referenceItems = mergeDistinctItems(
      pickRelevantItems(
        profileSummaryParagraphs(memory, { includeLowPriority: true }),
        signals,
        [],
        2,
        { allowLowPriority: true }
      ),
      pickRelevantItems(memory.coreMemories, signals, [], 2, { allowLowPriority: true }),
      pickRelevantItems(memory.preferences, signals, [], 2, { allowLowPriority: true }),
      pickRelevantItems(memory.doNotDo, signals, [], 2, { allowLowPriority: true }),
      pickRelevantItems(memory.lifeContext, signals, [], 2, { allowLowPriority: true })
    )
      .filter((item) => isLowPriorityReference(item))
      .slice(0, 2);

    pushBlock('RELEVANT REFERENCE MEMORY', referenceItems);
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

function makeRunRecord({
  at,
  status,
  responseId = null,
  packet = '',
  output = '',
  error = '',
  projectTag = 'General',
}) {
  return {
    id: responseId || `run-${Date.now()}`,
    at,
    status,
    responseId,
    projectTag: normalizeProjectTag(projectTag),
    mode: extractPacketField(packet, 'MODE'),
    domain: extractPacketField(packet, 'DOMAIN'),
    packetSummary: buildStoredSummary(packet, 220),
    responseSummary: buildStoredSummary(output, 220),
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
    normalizeHistoryKeyPart(run?.mode),
    normalizeHistoryKeyPart(run?.domain),
    normalizeHistoryKeyPart(run?.packetSummary),
    normalizeHistoryKeyPart(run?.responseSummary),
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

async function proxyVoiceSpeak(text, { method = 'GET' } = {}) {
  const response =
    method === 'POST'
      ? await fetch(`${VOICE_BASE_URL}/speak`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text }),
        })
      : await fetch(`${VOICE_BASE_URL}/speak?text=${encodeURIComponent(text)}`);
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
  { proxyMethod = 'GET' } = {}
) {
  const cleanText = String(text || '').replace(/\s+/g, ' ').trim();

  if (!cleanText) {
    return res.status(400).json({ ok: false, error: 'text is required' });
  }

  try {
    const proxied = await proxyVoiceSpeak(cleanText, { method: proxyMethod });

    res.status(proxied.status);
    res.setHeader('Content-Type', proxied.contentType);

    if (proxied.contentLength) {
      res.setHeader('Content-Length', proxied.contentLength);
    }

    res.setHeader('Content-Disposition', 'inline; filename="quinn.mp3"');
    res.setHeader('Cache-Control', 'public, max-age=600, immutable');
    return res.send(proxied.body);
  } catch (error) {
    if (!VOICE_PROXY_ONLY && hasLocalElevenVoiceConfig()) {
      try {
        const audio = await generateElevenSpeech({ text: cleanText, format: 'mp3' });
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

app.get('/voice-health', cors(), async (_req, res) => {
  try {
    const proxied = await proxyVoiceHealth();

    res.status(proxied.status);
    if (proxied.contentType) {
      res.setHeader('Content-Type', proxied.contentType);
    }

    return res.send(proxied.body);
  } catch (error) {
    if (!VOICE_PROXY_ONLY && hasLocalElevenVoiceConfig()) {
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
  return sendVoiceAudioResponse(res, req.query.text, 'Voice speak request failed.', {
    proxyMethod: 'GET',
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

    if (!packet || !String(packet).trim()) {
      return res.status(400).json({ error: 'packet is required' });
    }

    const memory = await readMemory();
    const memorySections = buildRunMemorySections(memory, {
      packet,
      projectTag,
    });
    const memoryBlock = memorySections
      .map((section) => formatSelectedMemoryBlock(section.title, section.items))
      .join('\n\n');

    const trimmedMemoryBlock = String(memoryBlock || '').slice(0, 3000);
    const trimmedPacket = String(packet || '').slice(0, 2200);
    const trimmedPrompt = String(
      prompt ||
        'Run this QuinnOS packet. Give the best tailored response based on the packet plus long-term memory.'
    ).slice(0, 500);

    const instructions = [
      'You are QuinnOS, the user’s deeply personalized operating layer.',
      'You should feel like the user’s best, most personalized version of ChatGPT inside this app, not like a generic assistant.',
      'Use the current packet as the active operating brief.',
      'The selected memory provided below has already been filtered for likely relevance to this packet.',
      'Answer the actual packet first.',
      'Use broad intelligence and general reasoning first. Use memory second, quietly, to sharpen the answer.',
      'Use memory to sharpen the answer, not to decorate it.',
      'Write as if you understand Quinn well, but do not perform memory or force personal details into unrelated answers.',
      'Only surface Quinn-specific details when they materially improve relevance, precision, or emotional accuracy.',
      'If a detail is low-priority trivia or a recurring callback, leave it out unless the packet clearly makes it relevant.',
      'Match the user’s preferred voice: direct, tailored, emotionally intelligent, specific, practical, clean, and high-context.',
      'Be sharp and natural. Use contractions. Sound human, fluent, and confident.',
      'Do not moralize, do not over-explain, and do not give filler.',
      'Do not sound corporate, clinical, canned, or vaguely supportive.',
      'Do not give generic self-help language, vague therapy-speak, or obvious AI phrasing.',
      'Do not say "as an AI", "I can’t", or other assistant-disclaimer language unless absolutely necessary.',
      'If the packet asks for judgment, give judgment.',
      'If the packet asks for strategy, be tactical and decisive.',
      'If the packet asks for writing, write the thing cleanly and fully instead of circling around it.',
      'Prefer concrete observations over broad advice.',
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

    const response = await client.responses.create({
      model,
      instructions,
      store: true,
      max_output_tokens: 1600,
      input: [
  {
    role: 'user',
    content: `PROJECT TAG

${projectTag}`,
  },
  {
    role: 'user',
    content: `SELECTED QUINN MEMORY FOR THIS RUN

${trimmedMemoryBlock}`,
  },
  {
    role: 'user',
    content: `ACTIVE USER REQUEST / CURRENT QUINNOS PACKET

${trimmedPacket}`,
  },
  {
    role: 'user',
    content: `TASK

${trimmedPrompt}`,
  },
  {
    role: 'user',
    content: `RESPONSE BEHAVIOR FOR THIS RUN

- Sound like the user's actually-personalized Quinn assistant, not a generic chatbot.
- Be specific quickly.
- Answer the packet first.
- Use broad intelligence first, then memory quietly to sharpen the answer.
- Treat the selected memory as optional sharpening context, not as a checklist of details to mention.
- Do not perform memory or default to recurring life details unless materially relevant.
- If the user wants a read, give a read.
- If the user wants a plan, give a plan.
- If the user wants writing, produce the writing.
- Avoid filler intros and avoid wrapping the response in obvious assistant framing.
- Do not end on a dangling phrase, cliffhanger, or ellipsis.`,
  },
],
    });

    const output =
      (response.output_text && response.output_text.trim()) ||
      JSON.stringify(response.output ?? [], null, 2);

    memory.lastResponseId = response.id;
    memory.runs = compactStoredRuns([
      makeRunRecord({
        at: now,
        status: 'success',
        responseId: response.id,
        packet,
        output,
        projectTag,
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
  });
});

app.post('/speak', async (req, res) => {
  return sendVoiceAudioResponse(res, req.body?.text, 'ElevenLabs speak failed', {
    proxyMethod: 'POST',
  });
});

app.post('/tts/quinn', async (req, res) => {
  return sendVoiceAudioResponse(res, req.body?.text, 'ElevenLabs TTS failed', {
    proxyMethod: 'POST',
  });
});

app.listen(port, host, async () => {
  await ensureMemoryFile();
  console.log(`QuinnOS backend running on http://${host}:${port}`);
  console.log(`Model: ${model}`);
  console.log(`Voice base URL: ${VOICE_BASE_URL}`);
  console.log(`Memory file: ${memoryPath}`);
});
