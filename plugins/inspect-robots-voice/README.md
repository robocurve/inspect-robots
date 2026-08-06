# inspect-robots-voice

Local spoken operator feedback for attended
[Inspect Robots](https://github.com/robocurve/inspect-robots) evaluations. The plugin keeps the
microphone open during a run, transcribes accepted utterances with faster-whisper, and sends them
through the same operator-message channel as typed console feedback.

Voice input is feedback-only. Spoken words cannot end a trial or record a verdict. Silence and
likely Whisper hallucinations produce no message.

## Install

```bash
pip install inspect-robots-voice
```

PortAudio must be available for `sounddevice` on the machine connected to the microphone.

## Run

```bash
inspect-robots run --task my-task --policy agent --embodiment my-robot --voice
```

Voice mode requires an attended terminal. Repeat `-V key=value` to configure it:

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `small` | faster-whisper model size or local model path |
| `device` | system default | sounddevice input index or case-insensitive name substring |
| `language` | `en` | transcription language |
| `compute` | `auto` | CTranslate2 compute type |

For example:

```bash
inspect-robots run --task my-task --policy agent --embodiment my-robot \
    --voice -V model=medium -V device="USB Microphone" -V compute=int8
```

## Filtering

Audio is segmented locally with an adaptive energy gate. Each candidate then passes
faster-whisper's bundled VAD plus duration, confidence, no-speech, and known silence-hallucination
checks. Rejected candidates disappear silently so the policy receives only accepted operator
utterances.

Transcription runs in a worker thread. The rollout loop never waits for speech recognition, and a
voice pipeline failure disables voice input without disabling typed console feedback.
