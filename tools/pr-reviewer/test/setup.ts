import { afterEach, beforeEach, vi } from 'vitest';
beforeEach(() => { vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('Network disabled in tests'); })); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });
