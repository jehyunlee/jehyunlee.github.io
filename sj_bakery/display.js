// Call from a player gesture: browsers require activation to enter fullscreen.
export function isGameFullscreen() {
  const doc = globalThis.document;
  return Boolean(doc && (doc.fullscreenElement || doc.webkitFullscreenElement || doc.webkitIsFullScreen));
}

export async function enterGameDisplay() {
  const root = globalThis.document?.documentElement;
  let landscape = false;

  if (root && !isGameFullscreen()) {
    try {
      // Keep the request before the first await so the Start click stays active.
      if (typeof root.requestFullscreen === 'function') {
        await root.requestFullscreen({ navigationUI: 'hide' });
      } else if (typeof root.webkitRequestFullscreen === 'function') {
        await root.webkitRequestFullscreen();
      }
    } catch {
      // The caller decides whether a rejected fullscreen request may start the game.
    }
  }

  const orientation = globalThis.screen?.orientation;
  if (typeof orientation?.lock === 'function') {
    try {
      await orientation.lock('landscape');
      landscape = true;
    } catch {
      // Devices without orientation locking keep their normal responsive view.
    }
  }

  return { fullscreen: isGameFullscreen(), landscape };
}

export async function exitGameDisplay() {
  const doc = globalThis.document;
  try {
    if (isGameFullscreen()) {
      if (typeof doc.exitFullscreen === 'function') {
        await doc.exitFullscreen();
      } else if (typeof doc.webkitExitFullscreen === 'function') {
        await doc.webkitExitFullscreen();
      }
    }
  } catch {
    // An unavailable exit API should not leave a rejected event-handler promise.
  } finally {
    const orientation = globalThis.screen?.orientation;
    if (typeof orientation?.unlock === 'function') {
      try { orientation.unlock(); } catch { /* Unsupported on some browsers. */ }
    }
  }
  return { fullscreen: isGameFullscreen(), landscape: false };
}
