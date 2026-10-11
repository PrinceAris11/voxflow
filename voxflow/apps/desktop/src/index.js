const { WebSocket } = require('ws');
const { simulateHotkey } = require('./hotkey.js');

const ws = new WebSocket('ws://127.0.0.1:8000/ws');

ws.on('open', () => {
  console.log('Connected to VoxFlow ML service');
  const hotkeyState = simulateHotkey();
  console.log('Hotkey state:', hotkeyState);
  ws.send(JSON.stringify({ type: 'audio', data: 'mock-audio-bytes' }));
  ws.send(JSON.stringify({ type: 'ping' }));
});

ws.on('message', (data) => {
  console.log('Server message:', data.toString());
});

ws.on('error', (err) => {
  console.error('WebSocket error:', err.message);
});

ws.on('close', () => {
  console.log('WebSocket closed');
});

process.on('SIGINT', () => {
  ws.close();
  process.exit(0);
});
