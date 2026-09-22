import { describe, expect, it, vi } from 'vitest';
import { retryLedger } from '../src/ledger-connection';

describe('bounded idempotent RPC retries', () => {
  it('stops after three newly connected attempts', async () => {
    const getByName = vi.fn(() => ({} as ReturnType<ReviewerEnv['LEDGER']['getByName']>));
    const operation = vi.fn(async () => { throw Object.assign(new Error('reset'), { retryable: true }); });
    await expect(retryLedger({ LEDGER: { getByName } }, operation)).rejects.toThrow('reset');
    expect(getByName).toHaveBeenCalledTimes(3);
    expect(operation).toHaveBeenCalledTimes(3);
  });
  it.each([Object.assign(new Error('overloaded'), { overloaded: true, retryable: true }), new Error('invalid_review_session')])('does not retry overload or permanent errors: %s', async error => {
    const getByName = vi.fn(() => ({} as ReturnType<ReviewerEnv['LEDGER']['getByName']>));
    await expect(retryLedger({ LEDGER: { getByName } }, async () => { throw error; })).rejects.toThrow(error.message);
    expect(getByName).toHaveBeenCalledTimes(1);
  });
});
