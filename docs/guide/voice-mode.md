# Voice operator input

Voice mode keeps a microphone open during an attended evaluation and sends accepted speech
through the same operator-message path as typed console feedback. Each accepted utterance is
echoed as `voice: <text>`, saved in the evaluation log with `source: "voice"`, and delivered at
the policy's next inference when the policy accepts operator messages.

Voice mode is available for both `run` and `eval-set`. It requires an interactive terminal and
does not work with `--no-prompt`.

## Install

Install the first-party voice plugin on the machine connected to the microphone:

```bash
pip install inspect-robots-voice
```

The plugin uses `sounddevice`, which needs the PortAudio system library. pip installs the
Python binding but not the library itself on Linux, so install it once per machine:

```bash
sudo apt install libportaudio2    # Debian, Ubuntu
sudo dnf install portaudio        # Fedora
brew install portaudio            # macOS
```

Model weights may be downloaded when voice input starts for the first time.

## Run

Pass `--voice` to enable the microphone for the whole evaluation:

```bash
inspect-robots run --task my-task --policy agent --embodiment my-robot --voice
inspect-robots eval-set 'my-benchmark/*' --policy agent --embodiment my-robot --voice
```

If the plugin is not installed, the CLI exits with an installation hint. Voice startup also
exits before the first trial when the model cannot load or the requested microphone cannot be
opened. A failure after startup disables only voice input. Typed console input continues to
work.

## Configure

Repeat `-V key=value` to configure the voice plugin. Values use the same scalar coercion as
`-P` and `-E`.

| Key | Default | Backend | Meaning |
| --- | --- | --- | --- |
| `model` | `parakeet-tdt-0.6b-v3` | both | Parakeet alias or `nemo-` name, or a faster-whisper model size or local path |
| `device` | system default | both | sounddevice input index or case-insensitive name substring |
| `language` | `none` | whisper | explicit transcription language; Parakeet auto-detects language |
| `compute` | `auto` | whisper | CTranslate2 compute type |
| `asr_device` | `cpu` | whisper | where Whisper runs: `cpu` (default, no CUDA needed), `cuda`, or `auto` (needs the CUDA runtime libraries) |

Parakeet TDT 0.6B v3 is the default. Its int8 ONNX weights download once from the Hugging Face
hub on first use and use about 640 MB. To use Whisper instead, pass `-V model=small` or another
faster-whisper model name or local path. Explicit `language`, `compute`, and non-CPU `asr_device`
values require a Whisper model.

Parakeet TDT 0.6B v3 weights are provided by NVIDIA under the CC-BY-4.0 license.

For example:

```bash
inspect-robots run --task my-task --policy agent --embodiment my-robot \
    --voice -V model=medium -V device="USB Microphone" -V compute=int8
```

An integer device value selects a sounddevice index. A string selects the single input device
whose name contains that value, ignoring case. Missing or ambiguous names fail with the
available input-device table. `-V` without `--voice` is an error, which helps catch flags copied
onto a non-voice invocation.

## Silence filtering

The plugin does not send every audio block to the policy. A local adaptive energy gate opens
after at least 100 ms above the learned noise threshold, prepends 300 ms of audio, and closes
after 700 ms below the threshold. An utterance is force-closed at 30 seconds so a stuck-open
gate cannot grow without bound. The noise estimate adapts only while speech is not open.

Each closed candidate shorter than 0.4 seconds is rejected before transcription. Both backends
also reject blank text and exact silence hallucinations such as `Thank you.`, `you`, and
`Thanks for watching!`.

Parakeet relies on the energy gate and its transducer decoder, then applies a coarse garbage
check that rejects mean token log probability below -2.5 when token probabilities are available.
Whisper also runs its bundled Silero VAD and rejects segments with no-speech probability above
0.6 or average log probability below -1.0. Rejected candidates disappear silently.

## Feedback-only behavior

Spoken input can add context, but it cannot end an episode or record a verdict. Episode end,
`/y`, `/n`, `/p`, and post-trial scoring remain keyboard actions. This limits a
mistranscription's effect to one extra feedback message instead of allowing it to terminate or
score a trial.

A session-aware embodiment can keep the operator channel active for a policy that does not
accept feedback. In that end-only mode, voice notes are echoed and saved to the log but are not
delivered to the policy. The CLI prints a notice so the operator knows the model cannot hear
them.
