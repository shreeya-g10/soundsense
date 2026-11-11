import { NavLink, Route, Routes } from 'react-router-dom';
import { USE_MOCK } from './api/client.js';
import Dashboard from './pages/Dashboard.jsx';
import LiveAlerts from './pages/LiveAlerts.jsx';
import Settings from './pages/Settings.jsx';
import CustomSounds from './pages/CustomSounds.jsx';
import History from './pages/History.jsx';

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/live', label: 'Live Alerts' },
  { to: '/history', label: 'History' },
  { to: '/custom-sounds', label: 'Custom Sounds' },
  { to: '/settings', label: 'Settings' },
];

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <nav className="flex flex-wrap items-center gap-4 border-b bg-white px-4 py-3">
        <span className="font-bold">SoundSense</span>
        {NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => (isActive ? 'font-semibold text-blue-600' : 'text-gray-600')}
          >
            {label}
          </NavLink>
        ))}
        {USE_MOCK && <span className="ml-auto rounded bg-yellow-100 px-2 py-0.5 text-xs">mock data</span>}
      </nav>
      <main className="p-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/live" element={<LiveAlerts />} />
          <Route path="/history" element={<History />} />
          <Route path="/custom-sounds" element={<CustomSounds />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}
