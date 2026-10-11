const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('voxflow', {
  /**
   * Request the main process to copy text and paste into the previous active application
   */
  pasteText: (text) => ipcRenderer.invoke('voxflow:paste-text', text),
  checkPermissions: () => ipcRenderer.invoke('voxflow:check-permissions'),

  /**
   * Request main process to minimize or hide the floating bar
   */
  hideWindow: () => ipcRenderer.send('voxflow:hide-window'),

  /**
   * Request main process to show and focus the floating bar
   */
  showWindow: () => ipcRenderer.send('voxflow:show-window'),

  /**
   * Listen for wake events triggered via Global Hotkey or Tray
   */
  onWakeCommand: (callback) => {
    ipcRenderer.on('voxflow:wake', (_event, data) => callback(data));
  },

  /**
   * Backend WebSocket URL
   */
  wsUrl: 'ws://127.0.0.1:8000/ws'
});
