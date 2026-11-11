// SoundSense API client: one function per endpoint in contracts/API.md.
//
// Set VITE_USE_MOCK=true in frontend/.env to use fake data from ./mock.js
// instead of the real backend at http://localhost:8000. Pages should import
// only from this file, never from mock.js directly.
//
// Every function returns a Promise of the JSON response and throws ApiError
// on failure.

import { ApiError } from './errors.js';
import { mockApi } from './mock.js';

export { ApiError };

export const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';
export const BASE_URL = 'http://localhost:8000';
const WS_URL = BASE_URL.replace(/^http/, 'ws') + '/ws/live';

async function request(method, path, body) {
  let res;
  try {
    res = await fetch(BASE_URL + path, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    throw new ApiError(0, `Cannot reach backend at ${BASE_URL} (${err.message})`);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail : JSON.stringify(data?.detail ?? res.statusText);
    throw new ApiError(res.status, detail);
  }
  return data;
}

function query(params) {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') qs.set(key, String(value));
  }
  const s = qs.toString();
  return s ? `?${s}` : '';
}

/**
 * Open the /ws/live WebSocket and call onMessage(msg) for every message
 * ({type: "level" | "alert" | "status", ...}). Reconnects automatically if the
 * connection drops. Returns a function that closes the connection for good.
 */
function connectLiveReal(onMessage) {
  let ws;
  let closed = false;
  let retry;

  const open = () => {
    ws = new WebSocket(WS_URL);
    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch (err) {
        console.error('Bad /ws/live message', event.data, err);
      }
    };
    ws.onclose = () => {
      if (!closed) retry = setTimeout(open, 2000);
    };
  };

  open();
  return () => {
    closed = true;
    clearTimeout(retry);
    ws?.close();
  };
}

const realApi = {
  // Monitoring
  startMonitor: () => request('POST', '/api/monitor/start'),
  stopMonitor: () => request('POST', '/api/monitor/stop'),

  // Status
  getStatus: () => request('GET', '/api/status'),

  // Settings (always send the full object)
  getSettings: () => request('GET', '/api/settings'),
  updateSettings: (settings) => request('PUT', '/api/settings', settings),

  // History: filters = {category, from, to, q, limit}, all optional
  getHistory: (filters = {}) => request('GET', '/api/history' + query(filters)),
  clearHistory: () => request('DELETE', '/api/history'),

  // Alerts
  ackAlert: (id) => request('POST', `/api/alerts/${encodeURIComponent(id)}/ack`),
  sendFeedback: (id, correct) =>
    request('POST', `/api/alerts/${encodeURIComponent(id)}/feedback`, { correct }),

  // Stats
  getStats: () => request('GET', '/api/stats'),

  // Custom sounds (record and test take ~3 s)
  getCustomSounds: () => request('GET', '/api/custom-sounds'),
  createCustomSound: (name, category) => request('POST', '/api/custom-sounds', { name, category }),
  recordCustomSound: (id) => request('POST', `/api/custom-sounds/${encodeURIComponent(id)}/record`),
  testCustomSound: (id) => request('POST', `/api/custom-sounds/${encodeURIComponent(id)}/test`),
  deleteCustomSound: (id) => request('DELETE', `/api/custom-sounds/${encodeURIComponent(id)}`),

  // Whistle counter
  getWhistleCounter: () => request('GET', '/api/whistle-counter'),
  updateWhistleCounter: (enabled, target) => request('PUT', '/api/whistle-counter', { enabled, target }),
  resetWhistleCounter: () => request('POST', '/api/whistle-counter/reset'),

  // Live WebSocket
  connectLive: connectLiveReal,
};

const api = USE_MOCK ? mockApi : realApi;

export const {
  startMonitor,
  stopMonitor,
  getStatus,
  getSettings,
  updateSettings,
  getHistory,
  clearHistory,
  ackAlert,
  sendFeedback,
  getStats,
  getCustomSounds,
  createCustomSound,
  recordCustomSound,
  testCustomSound,
  deleteCustomSound,
  getWhistleCounter,
  updateWhistleCounter,
  resetWhistleCounter,
  connectLive,
} = api;
