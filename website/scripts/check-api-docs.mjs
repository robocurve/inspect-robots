import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const apiPage = fileURLToPath(
  new URL('../../docs/api/index.md', import.meta.url),
);

if (!existsSync(apiPage)) {
  console.error(
    'Generated API docs are missing. From the repository root, run: uv run python scripts/gen_api_docs.py',
  );
  process.exit(1);
}
