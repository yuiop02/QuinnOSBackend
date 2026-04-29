# QuinnOS Backend on Railway

This backend is ready to run as two Railway services:

1. `quinn-api`
2. `quinn-voice`

The frontend contract stays the same:

- `POST /run`
- `GET /voice-health`
- `GET /voice-speak`
- replay and the current 2-chunk voice flow

## Recommended Railway layout

### Service 1: `quinn-api`

- Root directory: `QuinnOSBackend`
- Start command: `npm run start:api`
- Public networking: enabled
- Volume: attach a Railway volume and mount it at `/data`

Required variables:

- `OPENAI_API_KEY`
- `PORT=8787`
- `HOST=0.0.0.0`
- `VOICE_BASE_URL=http://quinn-voice.railway.internal:8788`
- `VOICE_PROXY_ONLY=true`
- `QUINNOS_STORAGE_DIR=/data`

Optional variables:

- `OPENAI_MODEL`
- `OPENAI_TRANSCRIBE_MODEL`
- `QUINNOS_MEMORY_FILE`

### Service 2: `quinn-voice`

- Root directory: `QuinnOSBackend`
- Start command: `npm run start:voice`
- Public networking: optional, can stay private-only

Required variables:

- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- `ELEVENLABS_MODEL_ID`
- `PORT=8788`
- `HOST=0.0.0.0`

Optional variables:

- `VOICE_CACHE_TTL_MS=600000`

## How persistence works now

Quinn memory is stored in a JSON file. The API service now resolves the memory path like this:

1. `QUINNOS_MEMORY_FILE` if set
2. `QUINNOS_STORAGE_DIR/memory.json` if `QUINNOS_STORAGE_DIR` is set
3. `RAILWAY_VOLUME_MOUNT_PATH/memory.json` if that env is provided
4. local bundled `data/memory.json`

On first boot, if the target memory file does not exist and the bundled `data/memory.json` exists, the server seeds the runtime storage path from that bundled file automatically. This is useful when a fresh Railway volume is attached.

Writes are now atomic, so memory updates are safer in a hosted environment.

## Local development

API:

```powershell
cd C:\Users\mrbro\QuinnOSBackend
npm run start:api
```

Voice:

```powershell
cd C:\Users\mrbro\QuinnOSBackend
npm run start:voice
```

## Health checks

API:

- `/health`
- `/voice-health`

Voice:

- `/health`

The API health response now includes:

- `voiceBaseUrl`
- `voiceProxyOnly`
- `storageDir`
- `memoryFile`
- `usingExternalStorage`

The voice health response now includes:

- `provider`
- `cacheEntries`
- `inFlightRequests`
- `cacheTtlMs`

## Important deployment note

For Railway, the clean production setup is:

- keep Fish secrets only on the `quinn-voice` service
- point `quinn-api` at `quinn-voice` with `VOICE_BASE_URL`
- keep Quinn memory on a mounted volume attached to `quinn-api`

That gives you:

- a public API service
- an internal/private voice worker
- persistent Quinn memory across deploys
- preserved frontend behavior with no frontend changes required
