/**
 * Client-side session for "La Mia Squadra".
 *
 * Why both localStorage + IndexedDB:
 * - The API keeps RosterContext in process memory with ~1h TTL.
 * - Persisting only the import JSON would leave a dead contextId after expiry
 *   or process restart.
 * - We therefore store (1) UI/session metadata in localStorage and (2) the
 *   original Excel blob in IndexedDB so we can silently re-import when needed.
 *
 * Client TTL is 15 days — longer than the server TTL by design.
 */

import {
  RosterImportResponse,
  Ruleset,
} from '../../core/models/my-team.models';

export const MY_TEAM_SESSION_VERSION = 1 as const;
export const MY_TEAM_CLIENT_TTL_MS = 15 * 24 * 60 * 60 * 1000; // 15 days

const LS_KEY = 'baudo-spoon:my-team-session:v1';
const IDB_NAME = 'baudo-spoon-my-team';
const IDB_STORE = 'files';
const IDB_FILE_KEY = 'roster-xlsx';

export interface MyTeamClaimSnapshot {
  sheetName: string;
  teamName: string;
}

export interface MyTeamSessionSnapshot {
  version: typeof MY_TEAM_SESSION_VERSION;
  savedAt: number;
  expiresAt: number;
  ruleset: Ruleset;
  importResult: RosterImportResponse;
  claimed: MyTeamClaimSnapshot | null;
  sourceFilename: string | null;
}

function canUseStorage(): boolean {
  try {
    return typeof window !== 'undefined' && typeof localStorage !== 'undefined';
  } catch {
    return false;
  }
}

/** Load session metadata; returns null if missing, corrupt, or expired. */
export function loadMyTeamSession(): MyTeamSessionSnapshot | null {
  if (!canUseStorage()) return null;
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as MyTeamSessionSnapshot;
    if (parsed?.version !== MY_TEAM_SESSION_VERSION) {
      clearMyTeamSession();
      return null;
    }
    if (!parsed.importResult?.contextId || !parsed.expiresAt) {
      clearMyTeamSession();
      return null;
    }
    if (Date.now() > parsed.expiresAt) {
      clearMyTeamSession();
      return null;
    }
    return parsed;
  } catch {
    clearMyTeamSession();
    return null;
  }
}

export function saveMyTeamSession(
  partial: Omit<MyTeamSessionSnapshot, 'version' | 'savedAt' | 'expiresAt'> & {
    savedAt?: number;
  },
): MyTeamSessionSnapshot {
  const savedAt = partial.savedAt ?? Date.now();
  const snapshot: MyTeamSessionSnapshot = {
    version: MY_TEAM_SESSION_VERSION,
    savedAt,
    expiresAt: savedAt + MY_TEAM_CLIENT_TTL_MS,
    ruleset: partial.ruleset,
    importResult: partial.importResult,
    claimed: partial.claimed,
    sourceFilename: partial.sourceFilename,
  };
  if (canUseStorage()) {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(snapshot));
    } catch (err) {
      // Quota exceeded or private mode — session still works in-memory.
      console.warn('[my-team-session] localStorage write failed', err);
    }
  }
  return snapshot;
}

export function clearMyTeamSession(): void {
  if (canUseStorage()) {
    try {
      localStorage.removeItem(LS_KEY);
    } catch {
      /* ignore */
    }
  }
  void clearStoredRosterFile();
}

/** Days remaining until client TTL expiry (ceil, min 0). */
export function sessionDaysRemaining(expiresAt: number): number {
  const ms = expiresAt - Date.now();
  if (ms <= 0) return 0;
  return Math.ceil(ms / (24 * 60 * 60 * 1000));
}

// ── IndexedDB file blob ─────────────────────────────────────────────────────

function openIdb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB unavailable'));
      return;
    }
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IDB open failed'));
  });
}

export async function saveRosterFile(file: File): Promise<void> {
  try {
    const buffer = await file.arrayBuffer();
    const db = await openIdb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).put(
        {
          name: file.name,
          type: file.type || 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          lastModified: file.lastModified,
          buffer,
        },
        IDB_FILE_KEY,
      );
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('IDB put failed'));
    });
    db.close();
  } catch (err) {
    console.warn('[my-team-session] IndexedDB file save failed', err);
  }
}

export async function loadRosterFile(): Promise<File | null> {
  try {
    const db = await openIdb();
    const record = await new Promise<{
      name: string;
      type: string;
      lastModified: number;
      buffer: ArrayBuffer;
    } | null>((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).get(IDB_FILE_KEY);
      req.onsuccess = () => resolve((req.result as typeof record) ?? null);
      req.onerror = () => reject(req.error ?? new Error('IDB get failed'));
    });
    db.close();
    if (!record?.buffer) return null;
    return new File([record.buffer], record.name, {
      type: record.type,
      lastModified: record.lastModified,
    });
  } catch {
    return null;
  }
}

export async function clearStoredRosterFile(): Promise<void> {
  try {
    const db = await openIdb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).delete(IDB_FILE_KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('IDB delete failed'));
    });
    db.close();
  } catch {
    /* ignore */
  }
}

/** True when HTTP error indicates a missing/expired roster context. */
export function isContextMissingError(err: unknown): boolean {
  const e = err as { status?: number; error?: { detail?: string }; message?: string };
  if (e?.status === 404) return true;
  const detail = String(e?.error?.detail ?? e?.message ?? '').toLowerCase();
  return (
    detail.includes('context non trovato') ||
    detail.includes('scaduto') ||
    detail.includes('not found')
  );
}
