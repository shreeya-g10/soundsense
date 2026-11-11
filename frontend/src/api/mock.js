// Fake backend for frontend development: same functions as client.js, same
// response formats as contracts/API.md, all state held in memory (resets on
// page reload). Used when VITE_USE_MOCK=true; import client.js, not this file.

import { CATEGORIES, CATEGORY_LABELS, DEFAULT_SETTINGS } from './constants.js';
import { ApiError } from './errors.js';

const LEVEL_EVERY_MS = 1000;
const ALERT_EVERY_MS = 8000;
const NETWORK_DELAY_MS = 150;
const RECORD_DELAY_MS = 3000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const clone = (x) => structuredClone(x);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const round = (x, d = 2) => Number(x.toFixed(d));
const newId = () => Math.random().toString(36).slice(2, 10);

async function reply(value, delay = NETWORK_DELAY_MS) {
  await sleep(delay);
  return clone(value);
}

// ISO 8601 with local timezone offset, e.g. 2026-11-12T19:04:05+05:30
function isoLocal(date = new Date()) {
  const pad = (n) => String(Math.floor(Math.abs(n))).padStart(2, '0');
  const off = -date.getTimezoneOffset();
  const sign = off >= 0 ? '+' : '-';
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}` +
    `${sign}${pad(off / 60)}:${pad(off % 60)}`
  );
}

const localDate = (date) => isoLocal(date).slice(0, 10);

// ---------------------------------------------------------------------------
// In-memory state
// ---------------------------------------------------------------------------

const state = {
  monitoring: true,
  monitoringSince: Date.now(),
  settings: clone(DEFAULT_SETTINGS),
  whistle: { enabled: true, target: 3, count: 0 },
  customSounds: [{ id: 'p1', name: 'My Kitchen Cooker', category: 'pressure_cooker', examples: 3 }],
  history: [], // newest first
};

function makeEvent(category, date, extra = {}) {
  return {
    id: newId(),
    timestamp: isoLocal(date),
    category,
    label: CATEGORY_LABELS[category],
    confidence: round(0.6 + Math.random() * 0.39),
    priority: state.settings.categories[category].priority,
    alerted: true,
    acknowledged: false,
    feedback: null,
    custom_name: null,
    room: 'Laptop',
    ...extra,
  };
}

// Seed ~60 past events over the last 14 days.
for (let i = 0; i < 60; i++) {
  const date = new Date(Date.now() - Math.random() * 14 * 24 * 3600 * 1000);
  const r = Math.random();
  state.history.push(
    makeEvent(pick(CATEGORIES), date, {
      acknowledged: Math.random() < 0.8,
      feedback: r < 0.6 ? true : r < 0.7 ? false : null,
      alerted: Math.random() < 0.9,
    }),
  );
}
state.history.sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));

function findEvent(id) {
  const event = state.history.find((e) => e.id === id);
  if (!event) throw new ApiError(404, `Alert ${id} not found`);
  return event;
}

function findCustom(id) {
  const sound = state.customSounds.find((s) => s.id === id);
  if (!sound) throw new ApiError(404, `Custom sound ${id} not found`);
  return sound;
}

// ---------------------------------------------------------------------------
// Live messages (/ws/live)
// ---------------------------------------------------------------------------

const subscribers = new Set();
let levelTimer = null;
let alertTimer = null;

function broadcast(msg) {
  for (const fn of subscribers) fn(clone(msg));
}

function emitLevel() {
  if (!state.monitoring) return;
  const skipped = Math.random() < 0.4;
  broadcast({
    type: 'level',
    rms: round(skipped ? Math.random() * 0.01 : 0.05 + Math.random() * 0.4, 3),
    top: skipped ? null : pick(['Speech', 'Music', 'Television', 'Silence', 'Dishes, pots, and pans']),
    skipped,
  });
}

function emitAlert() {
  if (!state.monitoring) return;
  const enabled = CATEGORIES.filter((c) => state.settings.categories[c].enabled);
  if (enabled.length === 0) return;
  const category = pick(enabled);
  const extra = {};

  if (category === 'pressure_cooker') {
    const custom = state.customSounds.find((s) => s.category === 'pressure_cooker');
    if (custom) extra.custom_name = custom.name;
    if (state.whistle.enabled) {
      state.whistle.count += 1;
      extra.label = `Whistle ${state.whistle.count} of ${state.whistle.target}`;
    }
  }

  const event = makeEvent(category, new Date(), extra);
  state.history.unshift(event);
  broadcast({
    type: 'alert',
    id: event.id,
    category: event.category,
    label: event.label,
    confidence: event.confidence,
    priority: event.priority,
    timestamp: event.timestamp,
    room: event.room,
    custom_name: event.custom_name,
    escalated: event.priority === 'critical' && Math.random() < 0.3,
  });
}

function connectLive(onMessage) {
  subscribers.add(onMessage);
  setTimeout(() => onMessage({ type: 'status', monitoring: state.monitoring }), 0);
  if (!levelTimer) {
    levelTimer = setInterval(emitLevel, LEVEL_EVERY_MS);
    alertTimer = setInterval(emitAlert, ALERT_EVERY_MS);
  }
  return () => {
    subscribers.delete(onMessage);
    if (subscribers.size === 0) {
      clearInterval(levelTimer);
      clearInterval(alertTimer);
      levelTimer = alertTimer = null;
    }
  };
}

function setMonitoring(on) {
  if (state.monitoring !== on) {
    state.monitoring = on;
    state.monitoringSince = Date.now();
    broadcast({ type: 'status', monitoring: on });
  }
  return { monitoring: on };
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

const fakeCpu = () => round(3 + Math.random() * 10, 1);
const fakeRam = () => round(380 + Math.random() * 60, 1);

export const mockApi = {
  startMonitor: () => reply(setMonitoring(true)),
  stopMonitor: () => reply(setMonitoring(false)),

  getStatus: () => {
    const today = localDate(new Date());
    return reply({
      monitoring: state.monitoring,
      mic_ok: true,
      uptime_s: state.monitoring ? round((Date.now() - state.monitoringSince) / 1000, 1) : 0,
      events_today: state.history.filter((e) => e.timestamp.startsWith(today)).length,
      cpu_percent: fakeCpu(),
      ram_mb: fakeRam(),
    });
  },

  getSettings: () => reply(state.settings),
  updateSettings: async (settings) => {
    await sleep(NETWORK_DELAY_MS);
    const missing = Object.keys(DEFAULT_SETTINGS).filter((k) => !(k in (settings ?? {})));
    if (missing.length) throw new ApiError(422, `Settings missing: ${missing.join(', ')}`);
    const missingCats = CATEGORIES.filter((c) => !(c in settings.categories));
    if (missingCats.length) throw new ApiError(422, `Settings missing categories: ${missingCats.join(', ')}`);
    state.settings = clone(settings);
    return clone(state.settings);
  },

  getHistory: ({ category, from, to, q, limit = 100 } = {}) => {
    const needle = q?.toLowerCase();
    const events = state.history.filter(
      (e) =>
        (!category || e.category === category) &&
        (!from || new Date(e.timestamp) >= new Date(from)) &&
        (!to || new Date(e.timestamp) <= new Date(to)) &&
        (!needle ||
          e.label.toLowerCase().includes(needle) ||
          (e.custom_name ?? '').toLowerCase().includes(needle)),
    );
    return reply(events.slice(0, Number(limit)));
  },
  clearHistory: () => {
    state.history = [];
    return reply({ ok: true });
  },

  ackAlert: async (id) => {
    await sleep(NETWORK_DELAY_MS);
    findEvent(id).acknowledged = true;
    return { ok: true };
  },
  sendFeedback: async (id, correct) => {
    await sleep(NETWORK_DELAY_MS);
    findEvent(id).feedback = Boolean(correct);
    return { ok: true };
  },

  getStats: () => {
    const per_category = Object.fromEntries(CATEGORIES.map((c) => [c, 0]));
    for (const e of state.history) per_category[e.category] += 1;

    const counts = {};
    for (const e of state.history) counts[e.timestamp.slice(0, 10)] = (counts[e.timestamp.slice(0, 10)] ?? 0) + 1;
    const per_day = [];
    for (let i = 29; i >= 0; i--) {
      const date = localDate(new Date(Date.now() - i * 24 * 3600 * 1000));
      per_day.push({ date, count: counts[date] ?? 0 });
    }

    const rated = state.history.filter((e) => e.feedback !== null);
    const wrong = rated.filter((e) => e.feedback === false).length;

    return reply({
      per_category,
      per_day,
      false_alert_rate: rated.length ? round(wrong / rated.length, 3) : 0,
      avg_latency_ms: round(120 + Math.random() * 60, 1),
      cpu_percent: fakeCpu(),
      ram_mb: fakeRam(),
    });
  },

  getCustomSounds: () => reply(state.customSounds),
  createCustomSound: async (name, category) => {
    await sleep(NETWORK_DELAY_MS);
    if (!name) throw new ApiError(422, 'name is required');
    if (!CATEGORIES.includes(category)) throw new ApiError(422, `Unknown category ${category}`);
    const sound = { id: newId(), name, category, examples: 0 };
    state.customSounds.push(sound);
    return { id: sound.id };
  },
  recordCustomSound: async (id) => {
    await sleep(RECORD_DELAY_MS);
    const sound = findCustom(id);
    sound.examples += 1;
    return { examples: sound.examples };
  },
  testCustomSound: async (id) => {
    await sleep(RECORD_DELAY_MS);
    const sound = findCustom(id);
    const similarity = sound.examples ? round(0.6 + Math.random() * 0.38) : 0;
    return { similarity, matched: similarity >= 0.75 };
  },
  deleteCustomSound: async (id) => {
    await sleep(NETWORK_DELAY_MS);
    findCustom(id);
    state.customSounds = state.customSounds.filter((s) => s.id !== id);
    return { ok: true };
  },

  getWhistleCounter: () => reply(state.whistle),
  updateWhistleCounter: async (enabled, target) => {
    await sleep(NETWORK_DELAY_MS);
    if (!Number.isInteger(target) || target < 1) throw new ApiError(422, 'target must be an integer >= 1');
    state.whistle.enabled = Boolean(enabled);
    state.whistle.target = target;
    return clone(state.whistle);
  },
  resetWhistleCounter: () => {
    state.whistle.count = 0;
    return reply(state.whistle);
  },

  connectLive,
};
