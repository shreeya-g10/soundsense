# listener/

**Owner:** Sneha (Role 3)

Phase 5 (optional) phone listener: a mobile browser page that turns a phone into an extra microphone for another room.

- Asks for a room name (e.g. "Kitchen") and opens `wss://<laptop-ip>:8000/ws/listener?room=Kitchen`.
- Captures the phone mic, converts it to Float32 PCM, 16 kHz, mono, and sends ~0.5 s binary chunks.
- Shows the `ack` and any `alert` messages for its room.
- Served by the backend over HTTPS (self-signed cert), because phone browsers only allow mic access on secure origins.

Audio is streamed only, never saved. The full protocol is in `contracts/API.md` (section `/ws/listener`). On the laptop, frames go to `RemoteAudioInput.push()` in `contracts/types.py`.
