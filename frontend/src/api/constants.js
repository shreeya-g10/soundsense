// Mirrors contracts/types.py and contracts/API.md. Keep in sync with contracts/.

export const CATEGORIES = [
  'smoke_alarm',
  'glass_break',
  'pressure_cooker',
  'baby_cry',
  'door_knock',
  'doorbell',
  'dog_bark',
  'appliance_beep',
  'water_running',
];

export const CATEGORY_LABELS = {
  smoke_alarm: 'Smoke alarm',
  glass_break: 'Glass breaking',
  pressure_cooker: 'Pressure cooker',
  baby_cry: 'Baby crying',
  door_knock: 'Door knock',
  doorbell: 'Doorbell',
  dog_bark: 'Dog barking',
  appliance_beep: 'Appliance beep',
  water_running: 'Water running',
};

export const PRIORITIES = ['critical', 'high', 'medium', 'low'];

const DEFAULT_PRIORITY = {
  smoke_alarm: 'critical',
  glass_break: 'critical',
  pressure_cooker: 'high',
  baby_cry: 'high',
  door_knock: 'medium',
  doorbell: 'medium',
  dog_bark: 'medium',
  appliance_beep: 'low',
  water_running: 'low',
};

// The default settings object from contracts/API.md.
export const DEFAULT_SETTINGS = {
  categories: Object.fromEntries(
    CATEGORIES.map((c) => [c, { enabled: true, priority: DEFAULT_PRIORITY[c], sensitivity: 0.5 }]),
  ),
  quiet_hours: { enabled: false, start: '22:00', end: '07:00' },
  alert_style: 'full',
  tv_guard: true,
  cpu_saver: true,
  custom_method: 'embedding',
  telegram: { enabled: false, bot_token: '', chat_id: '' },
};
