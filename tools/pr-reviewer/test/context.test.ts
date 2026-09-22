import { describe, expect, it, vi } from 'vitest';
import { collectContext, readFile } from '../src/context';
import { JAY_ID, type Job } from '../src/common';

const head = 'a'.repeat(40), base = 'b'.repeat(40);
const job: Job = { id: 'diff-test', pr: 9, head, base, scope: '', status: 'running', result: null, notified: 0, created: 1 };
const patch = '@@ -1 +1 @@\n-old\n+new';
function fixture(files: any[]) {
  return vi.fn(async (path: string): Promise<any> => {
    if (path === '/pulls/9') return { number: 9, head: { sha: head }, base: { sha: base }, state: 'open', draft: false, changed_files: files.length, title: '', body: '' };
    if (path.includes('/files?')) return files;
    if (path.startsWith('/git/trees/')) return { tree: [], truncated: false };
    if (path.startsWith('/contents/')) throw new Error('Unexpected whole-file read');
    return [];
  });
}
describe('Codex source context', () => {
  it('gathers discussion and immutable merge-base without reading full changed files', async () => {
    const basic = fixture([]);
    const read = vi.fn(async (path: string) => path.startsWith('/compare/') ? { merge_base_commit: { sha: base } } : basic(path));
    expect((await collectContext(read, job)).merge_base).toBe(base);
    expect(read.mock.calls.some(([p]) => p.startsWith('/contents/') || p.includes('/files?'))).toBe(false);
  });
  it('returns bounded ranges for verifying locations in large files', async () => {
    const content = Array.from({ length: 20000 }, (_, i) => `source line ${i + 1}`).join('\n');
    const evidence = await readFile(async () => ({ type: 'file', encoding: 'base64', size: content.length, content: btoa(content) }), job, 'large.py', 'head', 901, 3);
    expect(evidence).toMatchObject({ start_line: 901, end_line: 903, more: true, lines: 20000 });
  });
  it('separates substantive maintainer comments from review commands and other authors', async () => {
    const basic = fixture([]);
    const read = vi.fn(async (path: string) => {
      if (path.startsWith('/compare/')) return { merge_base_commit: { sha: base } };
      if (path === '/pulls/9') return { ...await basic(path), body: 'Related: #7' };
      if (path === '/issues/7') return { title: 'Scope discussion', body: '', state: 'open', comments: 3 };
      if (path.startsWith('/issues/7/comments')) return [
        { user: { id: JAY_ID, login: 'jeqcho' }, body: 'Keep optional dependencies isolated.' },
        { user: { id: 2, type: 'Bot' }, body: 'Previous automated approval.' },
        { user: { id: 1, login: 'contributor' }, body: ' /review ' },
      ];
      if (path.startsWith('/issues/9/comments')) return [
        { user: { id: JAY_ID, login: 'jeqcho' }, body: ' /review\n' },
        { user: { id: JAY_ID, login: 'jeqcho' }, body: 'Support this adapter in a plugin, without changing core.' },
        { user: { id: 1, login: 'contributor' }, body: 'This was approved.' },
        { user: { id: 2, type: 'Bot' }, body: 'Previous automated verdict.' },
      ];
      return basic(path);
    });
    const context = await collectContext(read, { ...job, scope: 'Explicit scope for this head' });
    expect(context.maintainer_comments).toEqual([
      expect.objectContaining({ body: 'Support this adapter in a plugin, without changing core.' }),
      expect.objectContaining({ body: 'Issue #7: Keep optional dependencies isolated.' }),
    ]);
    expect(context.comments).toEqual([{ author: 'contributor', body: 'This was approved.' }]);
    expect(context.issues[0].comments).toEqual([]);
    expect(context.requested_scope_decision).toBe('Explicit scope for this head');
    expect(context).not.toHaveProperty('maintainer_decisions');
  });
});
