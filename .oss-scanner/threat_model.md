# Threat model

Inspect Robots is a Python framework for evaluating robot policies (VLA models,
LLM agents, scripted controllers) on real robots and simulators. A `Policy` maps
observations to actions, an `Embodiment` executes them, and `eval()` runs the
closed loop and writes an `EvalLog`. The core package (`src/inspect_robots/`) is
NumPy plus the standard library. First-party plugins live in `plugins/*`.

The framework can command physical hardware, so a bug that lets an attacker
influence the actions sent to a robot is a physical-safety issue, not only a
software one.

## Where untrusted input enters

Treat all of these as attacker-controlled:

- **Remote policy servers:** `plugins/inspect-robots-xpolicylab` (msgpack and
  msgpack-numpy over websockets), `plugins/inspect-robots-capx` (HTTP perception
  and IK servers, `np.load` of returned arrays). Users often point these at
  shared lab or cloud servers they do not control.
- **LLM responses:** `plugins/inspect-robots-agent` (OpenAI-compatible and
  Anthropic wire formats) and `src/inspect_robots/_chatwire.py` (VLM grader,
  `taskgen`, `summarize`). Model output can be steered by prompt injection via
  camera images, task text, or operator messages.
- **Code-as-policy:** `plugins/inspect-robots-capx` executes model-generated
  Python in-process (`_sandbox.py`) with the evaluator's privileges. This is
  documented behaviour, not a security boundary: users running untrusted models
  must isolate the whole evaluator in a container or VM. Do not report imports,
  file, network, or process access from generated code as vulnerabilities. In
  scope: robot motion that generated code can produce without passing through
  the approver chain, and bugs in how model output is parsed before execution.
- **Robot middleware:** `plugins/inspect-robots-ros` speaks rosbridge JSON over
  websockets. A malicious or compromised rosbridge server is in scope.
- **Eval logs and frame directories:** `inspect-robots view`, `inspect`,
  `summarize`, and `video` read logs and `.npy`/PNG frames that are routinely
  shared between people. `_html.py` renders them into HTML reports that are
  opened in browsers, so script injection (XSS) through any log field is in scope.
- **The live-view server:** `inspect-robots view --serve` runs a
  `ThreadingHTTPServer` (`cli.py`). It binds to 127.0.0.1 by default but is
  documented with `--host 0.0.0.0` on headless rigs, so treat it as reachable
  from the local network. Path traversal or serving files outside the log
  directory is in scope.
- **Config and `.env` files:** `defaults.py` (INI config) and `_dotenv.py`.
  Lower priority: these normally come from the user themself.

## Components that matter most

1. The safety gate: `approver.py` (`ClampApprover`, `DeltaLimitApprover`,
   `ChainApprover`) and the non-finite-action check in `rollout.py`. Any input
   that makes an action reach `Embodiment.step()` outside the configured bounds,
   bypasses the approver chain, or turns a halting error (`SafetyAbort`,
   `EmbodimentFault`) into a continuing one is the most important class of bug.
2. Deserialisation of network responses in the plugins listed above.
3. HTML report rendering (`_html.py`) and the `--serve` HTTP server.
4. Subprocess use in `_video.py` (ffmpeg): argument or filename injection.

## Less important / out of scope

- `docs/` (Markdown guides only; there is no documentation site).
- `plugins/inspect-robots-isaacsim` and `plugins/inspect-robots-voice` (local
  simulator and local microphone/speaker); report only clear remote-input issues.
- Anything that requires the attacker to already control the user's Python
  environment, installed plugins (entry points run code by design), or config
  file.
- Denial of service against the user's own eval run, unless it defeats a
  safety halt.
- Leaking API keys the user passed to their own chosen provider endpoint.
- Host access by CaP-X generated code (see above): it runs with full process
  privileges by design.

## How to exercise it

- `tests/` runs the full stack against the dependency-free `CubePick` mock world
  (`src/inspect_robots/mock/`); no hardware or simulator is needed.
- Each plugin has its own `plugins/<name>/tests/` with mocked servers.
- `examples/quickstart.py` is a minimal end-to-end eval.
- `inspect-robots view <log> -o out.html` renders a report from a saved log.

## Severity

- **Critical:** remote code execution from a policy server, LLM response,
  rosbridge server, or shared log; or any input that sends an unapproved or
  out-of-bounds action to a robot.
- **High:** arbitrary file read or write through `--serve` or log/frame paths;
  stored XSS in HTML reports.
- **Medium:** crashes or hangs that skip a safety halt or leave the robot
  without a clean stop; information disclosure of local files to a remote
  server.
- **Low:** everything else.

## Report format

Please include a proof of concept that runs against the mock world or a plugin's
mocked server, and a minimal patch with a regression test (the core has a 100%
coverage gate).
