import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';

import styles from './index.module.css';

const quickstart = `from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

task = Task(
    name="cubepick-reach",
    scenes=[Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(5)],
    scorer=success_at_end(),
    max_steps=80,
)

# The two swappable inputs: a policy (VLA) and an embodiment (robot/sim).
(log,) = eval(task, ScriptedPolicy(), CubePickEmbodiment())
print(log.status, log.results.metrics)   # success {'success_at_end': 1.0}`;

const features = [
  {
    title: 'Real-world first',
    body: 'Interfaces assume real-robot reality: human-in-the-loop reset, no privileged success oracle, and a wall-clock control rate. Simulators just offer more (seeding, privileged success, rendering) via opt-in capabilities.',
  },
  {
    title: 'Reproducible',
    body: (
      <>
        Every run yields an immutable, schema-versioned <code>EvalLog</code>{' '}
        with the resolved config, git revision, and package versions. It is
        re-readable across releases and re-scorable offline.
      </>
    ),
  },
  {
    title: 'Light core',
    body: 'Depends only on NumPy. Rerun and simulator or VLA backends are optional extras and separately installable plugins.',
  },
  {
    title: 'Safe unattended',
    body: 'An explicit error taxonomy separates record and continue from halt and require a human, so a faulted robot never auto-advances overnight.',
  },
  {
    title: 'Rerun visualization',
    body: (
      <>
        Stream camera images, 3D poses, joint and action time-series, and
        success markers to a <code>.rrd</code> recording. Logging is
        non-blocking: a slow viewer connection drops camera frames first
        instead of delaying the robot control loop.
      </>
    ),
  },
  {
    title: 'Pluggable',
    body: (
      <>
        Backends ship as separate packages: the first-party plugins below, and
        rig plugins like <code>inspect-robots-yam</code>. Entry points make
        them appear in <code>inspect-robots list</code> automatically.
      </>
    ),
  },
];

const plugins = [
  {
    name: 'ros',
    description: (
      <>
        Run evals on ROS 1 or ROS 2 arms through rosbridge, with no ROS
        installation on the eval machine (<code>--embodiment ros</code>).
      </>
    ),
    href: 'https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-ros',
  },
  {
    name: 'isaacsim',
    description: (
      <>
        Run evals against an Isaac Lab simulation (
        <code>--embodiment isaacsim</code>).
      </>
    ),
    href: 'https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-isaacsim',
  },
  {
    name: 'xpolicylab',
    description: (
      <>
        Drive any XPolicyLab-served policy. One adapter puts its zoo of 40+
        VLAs (π0/π0.5, GR00T, OpenVLA-OFT, RDT-1B, SmolVLA, ACT, …) behind{' '}
        <code>--policy xpolicylab -P url=ws://gpu-box:19000</code>.
      </>
    ),
    href: 'https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-xpolicylab',
  },
  {
    name: 'agent',
    description: (
      <>
        Let a frontier LLM (Claude, GPT, anything behind an OpenAI-compatible
        API) drive any embodiment through tool calls, as a first-class policy.
        The same <code>--policy agent</code> runs ad-hoc instructions and
        scores on registered tasks next to fine-tuned VLAs.
      </>
    ),
    href: 'https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent',
  },
  {
    name: 'capx',
    description: (
      <>
        Evaluate CaP-X-style code-as-policy agents against a joint-space
        embodiment. Model-generated Python calls separately served SAM3,
        Contact-GraspNet, and Pyroki helpers, then queues approver-checked
        joint targets behind <code>--policy capx</code>.
      </>
    ),
    href: 'https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-capx',
  },
];

function GitHubMark(): ReactNode {
  return (
    <svg
      viewBox="0 0 16 16"
      width="18"
      height="18"
      aria-hidden="true"
      fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

function Hero(): ReactNode {
  return (
    <header className={styles.hero}>
      <div className={`container ${styles.heroInner}`}>
        <div>
          <Heading as="h1" className={styles.heroTitle}>
            Inspect Robots
          </Heading>
          <p className={styles.tagline}>
            An open-source evaluation framework for benchmarking AI and robots
            in the physical world
          </p>
          <p className={styles.heroCopy}>
            Define a robotics benchmark once, then run any policy against any
            compatible embodiment (a real robot or a simulator) with
            reproducible logs and first-class Rerun visualization.
          </p>
          <div className={styles.heroActions}>
            <Link className={styles.ctaPrimary} to="/guide/quickstart/">
              Get started
            </Link>
            <Link
              className={styles.ctaOutline}
              href="https://github.com/robocurve/inspect-robots">
              <GitHubMark /> GitHub
            </Link>
          </div>
        </div>
        <img
          className={styles.heroLogo}
          src="/img/inspect-robots-logo.svg"
          alt="Inspect Robots logo, a robot inspecting a dot through a magnifying lens"
        />
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Inspect Robots"
      description="An open-source evaluation framework for benchmarking AI and robots in the physical world">
      <Hero />
      <main className={styles.landing}>
        <section className={styles.section}>
          <div className="container">
            <p className={styles.eyebrow}>The framework</p>
            <Heading as="h2">One framework, two swappable inputs</Heading>
            <p className={styles.sectionLead}>
              LLM evals have a single swappable input: the model. Robotics evals
              have two, and Inspect Robots makes both first-class and orthogonal.
            </p>
            <div className={styles.inputGrid}>
              <article className={styles.inputCard}>
                <span className={styles.cardLabel}>The VLA brain</span>
                <Heading as="h3">Policy</Heading>
                <p>
                  Maps an observation and a language instruction to an action
                  chunk, a horizon of actions executed open-loop as π0, ACT, and
                  diffusion policies do.
                </p>
              </article>
              <article className={styles.inputCard}>
                <span className={styles.cardLabel}>The robot or sim</span>
                <Heading as="h3">Embodiment</Heading>
                <p>
                  Produces observations, executes actions, and owns the action
                  and observation spaces and control rate. Real robots come
                  first; simulators are a stricter special case.
                </p>
              </article>
            </div>
            <p className={styles.afterCards}>
              A Task, a dataset of Scenes plus scorers, is defined independently
              of both. Before any rollout, Inspect Robots verifies the policy and
              embodiment are compatible and fails fast if they are not.
            </p>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <div className={styles.sectionHeadingRow}>
              <div>
                <p className={styles.eyebrow}>Try it</p>
                <Heading as="h2">Quickstart</Heading>
                <p>No hardware or simulator required. The CubePick mock world exercises the whole stack.</p>
              </div>
              <Link to="/guide/quickstart/">Read the full quickstart</Link>
            </div>
            <CodeBlock language="python">{quickstart}</CodeBlock>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <p className={styles.eyebrow}>Design</p>
            <Heading as="h2">Why Inspect Robots</Heading>
            <div className={styles.featureGrid}>
              {features.map((feature) => (
                <article className={styles.featureCard} key={feature.title}>
                  <Heading as="h3">{feature.title}</Heading>
                  <p>{feature.body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <p className={styles.eyebrow}>For Inspect AI users</p>
            <Heading as="h2">How it maps to Inspect AI</Heading>
            <p>
              If you know{' '}
              <Link to="https://inspect.aisi.org.uk/">Inspect AI</Link>, you
              already know Inspect Robots.
            </p>
            <div className="table-responsive">
              <table>
                <thead>
                  <tr>
                    <th>Inspect AI</th>
                    <th>Inspect Robots</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><code>Model</code></td><td><code>Policy</code> (VLA) + <code>Embodiment</code> (two inputs)</td></tr>
                  <tr><td><code>Task = dataset + solver + scorer</code></td><td><code>Task = scenes + controller + scorer</code></td></tr>
                  <tr><td><code>Sample</code></td><td><code>Scene</code></td></tr>
                  <tr><td><code>Solver</code> chain</td><td><code>Controller</code> middleware (chunking, ensembling, smoothing)</td></tr>
                  <tr><td><code>eval()</code> → <code>EvalLog</code></td><td><code>eval()</code> → <code>EvalLog</code></td></tr>
                  <tr><td><code>@task</code> / <code>@solver</code> / <code>@scorer</code> + registry</td><td><code>@task</code> / <code>@policy</code> / <code>@embodiment</code> / <code>@scorer</code> + entry points</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className="container">
            <p className={styles.eyebrow}>Ecosystem</p>
            <Heading as="h2">First-party plugins</Heading>
            <p className={styles.sectionLead}>
              Both halves of an eval, the body and the brain, have ready-made
              adapters shipped from this repository as separate packages.
            </p>
            <div className={styles.pluginStrip}>
              {plugins.map((plugin) => (
                <Link className={styles.pluginCard} key={plugin.name} to={plugin.href}>
                  <code>inspect-robots-{plugin.name}</code>
                  <span>{plugin.description}</span>
                </Link>
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
