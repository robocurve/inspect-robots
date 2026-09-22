import { WorkerEntrypoint } from 'cloudflare:workers';
import { getSandbox, Sandbox, ContainerProxy } from '@cloudflare/sandbox';
import { Buffer } from 'node:buffer';
import { boundedText, SHA } from './common';
export { ContainerProxy };

export class ReviewSandbox extends Sandbox<RunnerEnv> {
  enableInternet = false;
  allowedHosts = ['review-model.local'];
  private launching?: Promise<void>;
  async launch(request: string): Promise<void> {
    // The durable claim prevents a lost start acknowledgement from launching twice.
    // The in-memory promise only coalesces concurrent calls; it is not the claim.
    if (!this.launching) this.launching = this.launchOnce(request).finally(() => { this.launching = undefined; });
    await this.launching;
  }
  private async launchOnce(request: string): Promise<void> {
    if (await this.ctx.storage.get('launch-claimed')) {
      if (!await this.getProcess('codex-review')) throw new Error('review_launch_uncertain');
      return;
    }
    const input = JSON.parse(request);
    await this.armDeadline();
    await this.writeFile('/tmp/head.tar.gz', await archive(input.head), { encoding: 'base64' });
    await this.writeFile('/tmp/base.tar.gz', await archive(input.base), { encoding: 'base64' });
    await this.writeFile('/tmp/request.json', request);
    const protectedFile = await this.exec('chmod 600 /tmp/request.json');
    if (!protectedFile.success) throw new Error('sandbox_execution_failed');
    await this.ctx.storage.put('launch-claimed', true);
    await this.startProcess('/opt/review-env/bin/python /opt/codex-review.py > /tmp/review-output.json', { processId: 'codex-review', autoCleanup: false, timeout: 21 * 60_000 });
  }
  async armDeadline(): Promise<void> {
    await this.schedule(22 * 60, 'expireReview');
  }
  async expireReview(): Promise<void> { await this.destroy(); }
}
ReviewSandbox.outboundByHost = {
  'review-model.local': async (request, env) => {
    // The client capability is carried in a private environment-backed header,
    // never in command arguments inherited/readable by repository processes.
    const url = new URL(request.url);
    if (request.method === 'POST' && url.pathname === '/responses') {
      const token = request.headers.get('X-Review-Token') ?? '';
      if (!/^[a-f0-9]{64}$/.test(token)) return new Response('Not allowed', { status: 403 });
      return env.MODEL.respond(token, await boundedText(request, 3_000_000));
    }
    const match = /^\/([a-f0-9]{64})\/(checkpoint)$/.exec(url.pathname);
    if (request.method !== 'POST' || !match) return new Response('Not allowed', { status: 403 });
    await env.MODEL.deliverCheckpoint(match[1], await boundedText(request, 1_500_000));
    return new Response('Saved');
  }
};

export async function archive(sha: string): Promise<string> {
  if (!SHA.test(sha)) throw new Error('invalid_review_request');
  const response = await fetch(new Request(`https://codeload.github.com/robocurve/inspect-robots/tar.gz/${sha}`, { redirect: 'manual', signal: AbortSignal.timeout(30000) }));
  if (!response.ok || !response.body) throw new Error('archive_unavailable');
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    length += part.value.length;
    if (length > 10_000_000) { await reader.cancel(); throw new Error('archive_too_large'); }
    chunks.push(part.value);
  }
  return Buffer.concat(chunks).toString('base64');
}
export class CodeRunner extends WorkerEntrypoint<RunnerEnv> {
  private box(id: string) {
    if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error('invalid_review_request');
    return getSandbox(this.env.SANDBOX, id, { sleepAfter: '30s', keepAlive: true, enableDefaultSession: false });
  }
  async start(head: string, base: string, context: string, policy: string, schema: string, token: string, sandbox: string, checkpointToken: string): Promise<void> {
    if (!/^[a-f0-9]{64}$/.test(checkpointToken) || !SHA.test(head) || !SHA.test(base) || !/^[a-f0-9]{64}$/.test(token) || context.length > 2_000_000 || policy.length > 30000 || schema.length > 30000) throw new Error('invalid_review_request');
    if (await this.env.MODEL.completed(token)) return;
    await this.box(sandbox).launch(JSON.stringify({ head, base, context, policy, schema, token, checkpointToken }));
  }
  async poll(sandbox: string, token: string): Promise<boolean> {
    if (await this.env.MODEL.completed(token)) return true;
    const box = this.box(sandbox);
    const process = await box.getProcess('codex-review');
    if (!process || ['starting', 'running'].includes(process.status)) return false;
    if (process.exitCode !== 0) throw new Error('sandbox_execution_failed');
    const file = await box.readFile('/tmp/review-output.json', { encoding: 'utf-8' });
    await this.env.MODEL.checkpoint(token, file.content);
    return true;
  }
  async cleanup(sandbox: string): Promise<void> { await this.box(sandbox).destroy(); }
}
export default { fetch() { return new Response('Not found', { status: 404 }); } };
