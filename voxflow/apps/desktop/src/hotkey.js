function simulateHotkey() {
  console.log('Hotkey fired: recording session started');
  return { status: 'recording', startedAt: Date.now() };
}

module.exports = { simulateHotkey };
