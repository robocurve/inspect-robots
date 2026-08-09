# inspect-robots-voice

Local spoken operator feedback for attended
[Inspect Robots](https://github.com/robocurve/inspect-robots) evaluations. The plugin keeps the
microphone open during a run, transcribes accepted utterances locally, and sends them through the
same operator-message channel as typed console feedback. Parakeet TDT 0.6B v3 is the default;
faster-whisper models remain selectable by model name or local path.

Voice input is feedback-only. Spoken words cannot end a trial or record a verdict. Silence and
low-confidence transcription candidates produce no message.

## Install

```bash
pip install inspect-robots-voice
```

PortAudio must be available for `sounddevice` on the machine connected to the microphone:
`sudo apt install libportaudio2` (Debian and Ubuntu), `sudo dnf install portaudio` (Fedora),
or `brew install portaudio` (macOS).

## Run

```bash
inspect-robots run --task my-task --policy agent --embodiment my-robot --voice
```

Voice mode requires an attended terminal. Repeat `-V key=value` to configure it:

| Key | Default | Backend | Meaning |
| --- | --- | --- | --- |
| `model` | `parakeet-tdt-0.6b-v3` | both | Parakeet alias or `nemo-` name, or a faster-whisper model size or local path |
| `device` | system default | both | sounddevice input index or case-insensitive name substring |
| `language` | `none` | whisper | explicit transcription language (Whisper uses `en` when unset); Parakeet auto-detects |
| `compute` | `auto` | whisper | CTranslate2 compute type |
| `asr_device` | `cpu` | whisper | `cpu`, `cuda`, or `auto` |

The default int8 Parakeet weights download once from the Hugging Face hub on first use and use
about 640 MB. Pass `-V model=small` to restore the previous Whisper default. Explicit `language`,
`compute`, and non-CPU `asr_device` values require a Whisper model.

Parakeet TDT 0.6B v3 weights are provided by NVIDIA under the CC-BY-4.0 license.

For example:

```bash
inspect-robots run --task my-task --policy agent --embodiment my-robot \
    --voice -V model=medium -V device="USB Microphone" -V compute=int8
```

The plugin can also narrate streamed policy notes with `run --speak`. Speech defaults to
`interrupt`, which cuts off superseded narration. Pass `-S mode=blocking` to wait boundedly for
each previous note, or `-S mode=queue` to retain bounded drop-oldest queueing.

## Filtering

Audio is segmented locally with an adaptive energy gate. Parakeet applies duration, blank-text,
coarse mean-token-confidence, and known silence-hallucination checks. Whisper keeps its bundled
VAD plus duration, confidence, no-speech, and known silence-hallucination checks. Rejected
candidates disappear silently so the policy receives only accepted operator utterances.

Transcription runs in a worker thread. The rollout loop never waits for speech recognition, and a
voice pipeline failure disables voice input without disabling typed console feedback.
