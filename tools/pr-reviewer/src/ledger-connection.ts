type LedgerEnvironment = { LEDGER: Pick<ReviewerEnv['LEDGER'], 'getByName'> };
type LedgerStub = ReturnType<LedgerEnvironment['LEDGER']['getByName']>;

export function transientLedgerError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const flags = error as Error & { retryable?: boolean; overloaded?: boolean };
  if (flags.overloaded) return false;
  return flags.retryable === true || error.message.includes('this Durable Object instance is no longer active');
}

// Only use with idempotent RPCs: a disconnect can occur after the write commits.
// In particular, never wrap reserve/register/openSession or paid model requests.
// Each attempt needs a new stub; an errored DO stub can remain permanently broken.
export async function retryLedger<T>(env: LedgerEnvironment, operation: (ledger: LedgerStub) => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try { return await operation(env.LEDGER.getByName('budget')); }
    catch (error) {
      if (attempt >= 2 || !transientLedgerError(error)) throw error;
      await new Promise(resolve => setTimeout(resolve, 100 * 2 ** attempt));
    }
  }
}
