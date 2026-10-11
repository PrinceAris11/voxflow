const { app, BrowserWindow, globalShortcut, ipcMain, screen, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const PasteHelper = require('./paste_helper');

let mainWindow = null;
let tray = null;
let isQuitting = false;

function createFloatingPillWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workArea;

  const winWidth = 420;
  const winHeight = 120;
  const posX = Math.round((screenWidth - winWidth) / 2);
  const posY = Math.round(screenHeight - winHeight - 36);

  mainWindow = new BrowserWindow({
    width: winWidth,
    height: winHeight,
    x: posX,
    y: posY,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.setAlwaysOnTop(true, 'floating', 1);
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  const htmlPath = path.join(__dirname, 'renderer', 'cylindrical_bar.html');
  mainWindow.loadFile(htmlPath);

  mainWindow.on('close', (e) => {
    if (!isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
}

function setupGlobalShortcut() {
  // Primary trigger: Ctrl+Shift+Space (or Cmd+Shift+Space on macOS)
  const shortcut = 'CommandOrControl+Shift+Space';
  const registered = globalShortcut.register(shortcut, () => {
    if (!mainWindow) return;
    if (mainWindow.isVisible()) {
      mainWindow.webContents.send('voxflow:wake', { source: 'hotkey' });
      mainWindow.focus();
    } else {
      mainWindow.show();
      mainWindow.webContents.send('voxflow:wake', { source: 'hotkey' });
      mainWindow.focus();
    }
  });

  if (!registered) {
    console.warn(`[VoxFlow] Failed to register global shortcut: ${shortcut}`);
  } else {
    console.log(`[VoxFlow] Registered global trigger shortcut: ${shortcut}`);
  }
}

function setupTray() {
  // Create a minimal programmatic SVG data-URL icon for the system tray
  const svgIcon = `
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
      <rect width="24" height="24" rx="12" fill="#7c3aed"/>
      <path d="M12 4a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V7a3 3 0 0 0-3-3zm5 7a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V20H9v2h6v-2h-2v-2.08A7 7 0 0 0 19 11h-2z" fill="#ffffff"/>
    </svg>
  `;
  const icon = nativeImage.createFromDataURL(`data:image/svg+xml;base64,${Buffer.from(svgIcon).toString('base64')}`);
  tray = new Tray(icon.resize({ width: 18, height: 18 }));
  tray.setToolTip('VoxFlow Desktop Assistant');

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Wake VoxFlow (Ctrl+Shift+Space)',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.webContents.send('voxflow:wake', { source: 'tray' });
        }
      }
    },
    {
      label: 'Toggle Floating Bar',
      click: () => {
        if (!mainWindow) return;
        if (mainWindow.isVisible()) mainWindow.hide();
        else mainWindow.show();
      }
    },
    { type: 'separator' },
    {
      label: 'ML Engine: ws://127.0.0.1:8000',
      enabled: false
    },
    { type: 'separator' },
    {
      label: 'Quit VoxFlow',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);

  tray.setContextMenu(contextMenu);
  tray.on('click', () => {
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

// IPC: Auto-paste refined transcript
ipcMain.handle('voxflow:paste-text', async (_event, text) => {
  try {
    // Hide the floating bar to return OS focus to the previously active window
    if (mainWindow) {
      mainWindow.hide();
    }

    // Run the multi-platform clipboard and keystroke paste automation
    const result = await PasteHelper.autoPaste(text);
    return result;
  } catch (err) {
    console.error('[VoxFlow IPC] Error during autoPaste:', err);
  }
});

ipcMain.handle('voxflow:check-permissions', async () => {
  try {
    return await PasteHelper.checkPermissions();
  } catch (err) {
    return { hasPermission: false, error: err.message };
  }
});

ipcMain.on('voxflow:hide-window', () => {
  if (mainWindow) mainWindow.hide();
});

ipcMain.on('voxflow:show-window', () => {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
  }
});

app.whenReady().then(() => {
  createFloatingPillWindow();
  setupGlobalShortcut();
  setupTray();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createFloatingPillWindow();
    } else if (mainWindow) {
      mainWindow.show();
    }
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
