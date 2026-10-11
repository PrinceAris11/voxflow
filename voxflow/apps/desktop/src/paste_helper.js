const { exec } = require('child_process');
const { clipboard, systemPreferences } = require('electron');

/**
 * Cross-Platform Keystroke & OS Text Injection Automation Helper.
 * Handles system permissions (macOS Accessibility, Linux Wayland vs X11, Windows SendKeys),
 * safe focus transition, clipboard staging, and keystroke synthesis.
 */
class PasteHelper {
  /**
   * Check OS-level permissions and available injection tools.
   * @returns {Promise<{hasPermission: boolean, platform: string, displayServer?: string, method: string, warning?: string}>}
   */
  static async checkPermissions() {
    const platform = process.platform;

    if (platform === 'darwin') {
      // macOS: Accessibility permissions check
      const isTrusted = systemPreferences && typeof systemPreferences.isTrustedAccessibilityClient === 'function'
        ? systemPreferences.isTrustedAccessibilityClient(false)
        : true;

      return {
        hasPermission: isTrusted,
        platform: 'darwin',
        method: isTrusted ? 'macos-osascript' : 'clipboard-only',
        warning: isTrusted
          ? null
          : 'Accessibility permission is required for auto-paste into other apps. Enable VoxFlow in System Settings → Privacy & Security → Accessibility.'
      };
    }

    if (platform === 'win32') {
      // Windows: standard desktop apps can receive SendKeys
      return {
        hasPermission: true,
        platform: 'win32',
        method: 'win32-sendkeys',
        warning: null
      };
    }

    if (platform === 'linux') {
      const isWayland = Boolean(process.env.WAYLAND_DISPLAY || process.env.XDG_SESSION_TYPE === 'wayland');
      const displayServer = isWayland ? 'wayland' : 'x11';

      return new Promise((resolve) => {
        if (isWayland) {
          exec('which wtype || which ydotool', (err, stdout) => {
            if (!err && (stdout.includes('wtype') || stdout.includes('ydotool'))) {
              resolve({
                hasPermission: true,
                platform: 'linux',
                displayServer: 'wayland',
                method: stdout.includes('wtype') ? 'wayland-wtype' : 'wayland-ydotool',
                warning: null
              });
            } else {
              resolve({
                hasPermission: false,
                platform: 'linux',
                displayServer: 'wayland',
                method: 'clipboard-only',
                warning: 'Wayland security model restricts synthetic keystrokes without wtype or ydotool. Text is safely copied to clipboard; press Ctrl+V to paste.'
              });
            }
          });
        } else {
          exec('which xdotool', (err, stdout) => {
            if (!err && stdout.includes('xdotool')) {
              resolve({
                hasPermission: true,
                platform: 'linux',
                displayServer: 'x11',
                method: 'x11-xdotool',
                warning: null
              });
            } else {
              resolve({
                hasPermission: false,
                platform: 'linux',
                displayServer: 'x11',
                method: 'clipboard-only',
                warning: 'xdotool is not installed. Text is copied to clipboard; press Ctrl+V to paste. (Install xdotool via: sudo apt install xdotool)'
              });
            }
          });
        }
      });
    }

    return { hasPermission: true, platform, method: 'clipboard-only' };
  }

  /**
   * Paste text into the currently active text box in the target application.
   * @param {string} text - Refined transcript to paste.
   * @param {object} [options] - Options (preserveClipboard, delayMs)
   * @returns {Promise<{success: boolean, method: string, error?: string}>}
   */
  static async autoPaste(text, options = {}) {
    if (!text || typeof text !== 'string') {
      return { success: false, method: 'none', error: 'Empty text' };
    }

    const previousClipboard = options.preserveClipboard ? clipboard.readText() : null;

    // Step 1: Write target text to OS clipboard
    clipboard.writeText(text);

    // Step 2: Give OS window manager time to restore focus to previously active window
    const delay = options.delayMs || 120;
    await new Promise((resolve) => setTimeout(resolve, delay));

    const platform = process.platform;

    return new Promise((resolve) => {
      const finish = (result) => {
        // Optionally restore previous clipboard content after paste has registered
        if (options.preserveClipboard && previousClipboard !== null) {
          setTimeout(() => {
            try { clipboard.writeText(previousClipboard); } catch (_) {}
          }, 600);
        }
        resolve(result);
      };

      if (platform === 'darwin') {
        const script = `osascript -e 'tell application "System Events" to keystroke "v" using command down'`;
        exec(script, (err) => {
          if (err) {
            console.warn('[PasteHelper] macOS AppleScript paste failed, text remains in clipboard:', err.message);
            finish({ success: false, method: 'clipboard-only', error: err.message });
          } else {
            finish({ success: true, method: 'macos-osascript' });
          }
        });
      } else if (platform === 'win32') {
        const psCommand = `powershell -NoProfile -NonInteractive -Command "$wshell = New-Object -ComObject WScript.Shell; $wshell.SendKeys('^v')"`;
        exec(psCommand, (err) => {
          if (err) {
            console.warn('[PasteHelper] Windows SendKeys failed, text remains in clipboard:', err.message);
            finish({ success: false, method: 'clipboard-only', error: err.message });
          } else {
            finish({ success: true, method: 'win32-sendkeys' });
          }
        });
      } else if (platform === 'linux') {
        const isWayland = Boolean(process.env.WAYLAND_DISPLAY || process.env.XDG_SESSION_TYPE === 'wayland');

        if (isWayland) {
          exec('which wtype || which ydotool', (whichErr, stdout) => {
            if (!whichErr && stdout.includes('wtype')) {
              exec('wtype -M ctrl -k v -m ctrl', (err) => {
                if (err) finish({ success: false, method: 'clipboard-only' });
                else finish({ success: true, method: 'wayland-wtype' });
              });
            } else if (!whichErr && stdout.includes('ydotool')) {
              exec('ydotool key 29:1 47:1 47:0 29:0', (err) => {
                if (err) finish({ success: false, method: 'clipboard-only' });
                else finish({ success: true, method: 'wayland-ydotool' });
              });
            } else {
              console.log('[PasteHelper] Wayland active: Text safely copied to clipboard.');
              finish({ success: true, method: 'clipboard-only' });
            }
          });
        } else {
          exec('xdotool key --clearmodifiers ctrl+v', (err) => {
            if (err) {
              console.log('[PasteHelper] X11 xdotool unavailable: Text safely copied to clipboard.');
              finish({ success: true, method: 'clipboard-only' });
            } else {
              finish({ success: true, method: 'x11-xdotool' });
            }
          });
        }
      } else {
        finish({ success: true, method: 'clipboard-only' });
      }
    });
  }

  /**
   * Check if a window title or app identifier appears to be a sensitive credential/password field.
   * @param {string} [appOrWindowTitle]
   * @returns {boolean}
   */
  static isSensitiveTarget(appOrWindowTitle) {
    if (!appOrWindowTitle) return false;
    const lower = String(appOrWindowTitle).toLowerCase();
    const sensitiveTokens = [
      'password', 'passcode', 'credential', 'keepass', '1password',
      'bitwarden', 'lastpass', 'dashlane', 'auth', 'sudo', 'terminal - su'
    ];
    return sensitiveTokens.some(token => lower.includes(token));
  }

  /**
   * Record injection event to local-only audit log (zero outbound transmission).
   * @param {{characters: number, method: string, success: boolean, sensitiveBlocked: boolean}} entry
   */
  static recordAuditLog(entry) {
    try {
      const fs = require('fs');
      const path = require('path');
      const auditDir = path.join(__dirname, '..', '..', '..', 'data');
      if (!fs.existsSync(auditDir)) {
        fs.mkdirSync(auditDir, { recursive: true });
      }
      const logFile = path.join(auditDir, 'injection_audit.jsonl');
      const record = {
        timestamp: new Date().toISOString(),
        characters: entry.characters || 0,
        method: entry.method,
        success: entry.success,
        sensitiveBlocked: entry.sensitiveBlocked || false,
      };
      fs.appendFileSync(logFile, JSON.stringify(record) + '\n', 'utf8');
    } catch (_) {
      // Non-blocking local audit logging
    }
  }
}

module.exports = PasteHelper;

