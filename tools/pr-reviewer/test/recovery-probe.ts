// Disposable Cloudflare integration harness. No provider key or GitHub publisher.
import { DurableObject, WorkerEntrypoint, WorkflowEntrypoint, type WorkflowEvent, type WorkflowStep } from 'cloudflare:workers';
import { runReview } from '../src/review';
import { ReviewLedger } from '../src/ledger';
import { CodeRunner, ReviewSandbox, ContainerProxy } from '../src/sandbox';
import type { Job } from '../src/common';
export { ReviewLedger, CodeRunner, ReviewSandbox, ContainerProxy };
interface ProbeEnv {
  LEDGER: DurableObjectNamespace<ReviewLedger>;
  STATS: DurableObjectNamespace<ProbeStats>;
  RUNNER: Service<CodeRunner>;
}
export class ProbeStats extends DurableObject {
  async increment(key: string) {
    const count = (await this.ctx.storage.get<number>(key) ?? 0) + 1;
    await this.ctx.storage.put(key, count); return count;
  }
  async counts(): Promise<{ modelTurns: number; launches: number; recoveryFaults: number }> { return { modelTurns: await this.ctx.storage.get<number>('modelTurns') ?? 0, launches: await this.ctx.storage.get<number>('launches') ?? 0, recoveryFaults: await this.ctx.storage.get<number>('recoveryFaults') ?? 0 }; }
}
const head = '696fbaa9a00d7c345a81dd179fa10934f51ade89';
const base = '4d35fe4b81a643c0e61e8d287810b3c26f9d386a';
const fixture = { worthwhile: 'YES', scope: 'ESTABLISHED', verdict: 'REQUIRE_REVIEWER', recommended_action: 'COMPLETE_REVIEW', rationale: 'Synthetic recovery probe. This is not a PR review.', blockers: [], contract_and_test_review: 'Infrastructure test only.', checks: [], limitations: ['No real model reviewed this PR.'], sufficient_review: false, decision_needed: '', body: 'Injected output for transport validation; never publish to GitHub.' };
export class ProbeModel extends WorkerEntrypoint<ProbeEnv> {
  async deliverCheckpoint(receipt: string, body: string) { await this.env.LEDGER.getByName('budget').deliverCheckpoint(receipt, body); }
  async checkpoint(token: string, body: string) { await this.env.LEDGER.getByName('budget').checkpoint(token, body); }
  async completed(token: string) {
    const ledger = this.env.LEDGER.getByName('budget');
    const job = await ledger.session(token); if (!job) throw new Error('invalid_review_session');
    return await ledger.runOutput(job.id) !== null;
  }
  async respond(token: string, body: string) {
    const ledger = this.env.LEDGER.getByName('budget');
    if (!await ledger.session(token)) return new Response('Forbidden', { status: 403 });
    const count = await this.env.STATS.getByName('probe-v4').increment('modelTurns');
    const request = JSON.parse(body);
    let events;
    const response = { id: `resp_probe_${count}`, object: 'response', created_at: 1, status: 'completed', model: 'gpt-6-astra', usage: { input_tokens: 10, output_tokens: 10, total_tokens: 20 } };
    if (!request.input.some((v: { type: string }) => v.type === 'custom_tool_call_output')) {
      const call = { type: 'custom_tool_call', id: 'ctc_probe', call_id: 'call_probe', name: 'exec', namespace: 'functions', input: `text(await tools.exec_command(${JSON.stringify({ cmd: `python -c "import os; assert os.getuid() == 65534; assert not os.access('/tmp/request.json', os.R_OK); print('receipt protected')"`, yield_time_ms: 1000 })}));` };
      events = [
        { type: 'response.output_item.added', output_index: 0, item: { ...call, input: '' } },
        { type: 'response.custom_tool_call_input.delta', item_id: call.id, output_index: 0, delta: call.input },
        { type: 'response.custom_tool_call_input.done', item_id: call.id, output_index: 0, input: call.input },
        { type: 'response.output_item.done', output_index: 0, item: call },
        { type: 'response.completed', response: { ...response, output: [call] } },
      ];
    } else {
      const text = JSON.stringify(fixture);
      const part = { type: 'output_text', text, annotations: [] };
      const item = { type: 'message', id: 'msg_probe', role: 'assistant', status: 'completed', content: [part] };
      events = [
        { type: 'response.created', response: { ...response, status: 'in_progress', output: [] } },
        { type: 'response.output_item.added', output_index: 0, item: { ...item, status: 'in_progress', content: [] } },
        { type: 'response.content_part.added', item_id: item.id, output_index: 0, content_index: 0, part: { ...part, text: '' } },
        { type: 'response.output_text.delta', item_id: item.id, output_index: 0, content_index: 0, delta: text },
        { type: 'response.output_text.done', item_id: item.id, output_index: 0, content_index: 0, text },
        { type: 'response.content_part.done', item_id: item.id, output_index: 0, content_index: 0, part },
        { type: 'response.output_item.done', output_index: 0, item },
        { type: 'response.completed', response: { ...response, output: [item] } },
      ];
    }
    return new Response(events.map(event => 'data: ' + JSON.stringify(event) + '\n\n').join(''), { headers: { 'Content-Type': 'text/event-stream' } });
  }
}
export class RecoveryProbe extends WorkflowEntrypoint<ProbeEnv, { replay?: boolean }> {
  async run(_event: WorkflowEvent<{ replay?: boolean }>, step: WorkflowStep) {
    const ledger = this.env.LEDGER.getByName('budget');
    const job: Job = { id: 'recovery-probe-v4', pr: 999999, head, base, scope: '', status: 'running', result: null, created: 1, notified: 0 };
    await ledger.register(job); await ledger.claimReviewSlot(job.id);
    const wrapped = {
      do: (async (name: string, config: unknown, callback: unknown) => {
        const value = await Reflect.apply(step.do, step, [name, config, callback]);
        if (name.startsWith('inspect reviewer process') && value === 'complete') {
          await this.env.STATS.getByName('probe-v4').increment('recoveryFaults');
          throw new Error('injected lost workflow acknowledgement');
        }
        return value;
      }) as WorkflowStep['do'],
      sleep: (name: string, duration: Parameters<WorkflowStep["sleep"]>[1]) => step.sleep(name, duration),
    };
    const result = await runReview({
      LEDGER: this.env.LEDGER,
      PUBLISHER: { read: async (path: string) => {
        if (path === '/pulls/999999') return JSON.stringify({ number: 999999, head: { sha: head }, base: { sha: base }, state: 'open', draft: false, title: 'Synthetic infrastructure probe', body: '', user: { login: 'probe' } });
        if (path.startsWith('/compare/')) return JSON.stringify({ merge_base_commit: { sha: base } });
        return '[]';
      } },
      RUNNER: {
        start: async (...args) => { await this.env.STATS.getByName('probe-v4').increment('launches'); await this.env.RUNNER.start(...args); },
        poll: async (...args) => { const complete = await this.env.RUNNER.poll(...args); if (complete) throw new Error('injected lost completion acknowledgement'); return false; },
        cleanup: async (...args) => { await this.env.RUNNER.cleanup(...args); throw new Error('injected lost cleanup acknowledgement'); },
      },
    }, job, wrapped);
    return JSON.stringify({ synthetic: true, verdict: result.verdict, executions: result.execution_records, costs: result.cost_summary, ...await this.env.STATS.getByName('probe-v4').counts() });
  }
}
export default { fetch() { return new Response('Not found', { status: 404 }); } };
