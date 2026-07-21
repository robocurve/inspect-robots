import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const description =
  'Inspect Robots is the open-source evaluation framework for physical AI and VLA ' +
  '(vision-language-action) models: the "Inspect AI for robotics". A ' +
  'benchmark is a Task (dataset of Scenes + scorers) run against two ' +
  'swappable inputs: a Policy (the VLA) and an Embodiment (a real robot or ' +
  'simulator).';

const guideFiles = [
  'guide/quickstart.md',
  'guide/concepts.md',
  'guide/writing-a-benchmark.md',
  'guide/policies-and-embodiments.md',
  'guide/scoring.md',
  'guide/logging-and-rerun.md',
  'guide/plugins.md',
  'guide/adapters.md',
  'guide/cli.md',
];

const docsDirectory = fileURLToPath(new URL('../../docs/', import.meta.url));
const buildDirectory = fileURLToPath(new URL('../build/', import.meta.url));

function titleFromMarkdown(markdown, file) {
  const heading = markdown.match(/^#\s+(.+)$/m);
  if (!heading) {
    throw new Error(`${file} has no H1 heading`);
  }
  return heading[1];
}

function siteUrl(file) {
  return `https://inspectrobots.org/${file.replace(/\.md$/, '/')}`;
}

const guides = await Promise.all(
  guideFiles.map(async (file) => ({
    file,
    markdown: await readFile(new URL(file, `file://${docsDirectory}/`), 'utf8'),
  })),
);

const linkList = guides
  .map(({file, markdown}) => `- [${titleFromMarkdown(markdown, file)}](${siteUrl(file)})`)
  .join('\n');
const llmsText = `# Inspect Robots\n\n${description}\n\n## Guide\n\n${linkList}\n`;
const fullText = [
  `# Inspect Robots\n\n${description}`,
  ...guides.map(({file, markdown}) => `<!-- Source: ${file} -->\n\n${markdown.trim()}`),
].join('\n\n---\n\n') + '\n';

await mkdir(buildDirectory, {recursive: true});
await Promise.all([
  writeFile(new URL('llms.txt', `file://${buildDirectory}/`), llmsText),
  writeFile(new URL('llms-full.txt', `file://${buildDirectory}/`), fullText),
]);
