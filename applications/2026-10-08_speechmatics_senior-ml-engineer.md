# Speechmatics — Senior Machine Learning Engineer

- **Date:** 2026-10-08
- **Link:** https://wellfound.com/jobs/4797168-senior-machine-learning-engineer
- **Company:** Speechmatics (51–200 employees, Voice AI / speech recognition)
- **Location:** London · Full time · Visa sponsorship not available · Relocation not allowed
- **Posted:** 5 days ago
- **Hiring contact:** David Field
- **Status:** Application form filled in; screenshot is at the form before "Send application" — send not yet confirmed
- **Proof:** `proof/2026-10-08_speechmatics_senior-ml-engineer.png`

## Role summary
R&D on the Modelling Team: training large self-supervised speech models and deploying them. Wants transformer knowledge (GQA, KV-caching), distributed training, inference optimisation (dynamic batching, flash attention, speculative decoding); publications/open source/technical writing preferred.

## Why a good fit (message to hiring contact)
I build a dictation tool I use every day, a Whisper Flow-style pipeline: streaming speech-to-text, voice activity detection, barge-in, and text-to-speech. I keep the audio segments from my own sessions and I am fine-tuning Parakeet on them so it handles my voice, my corrections and the way I actually speak, instead of a generic checkpoint. Around that I have written a Whisper-style log-mel front-end in NumPy, with tests for shape, determinism, silence and gain, and I serve Silero VAD through ONNX Runtime so the inference path does not import PyTorch.

The reason I want this team is the same reason I built the tool. I want speech systems that hold up for people who struggle to read, write and be understood, including breaking language down far enough that the model is useful rather than just accurate on a benchmark. I have not trained Speechmatics-scale self-supervised models, and I have not done distributed training. I do train models properly, with walk-forward validation and held-out data, and I care about the serving side: latency, state across chunks, and a contract other people can rely on. Speechmatics is the modelling team I want to learn that from.

## Other form answers
- How did you hear about us: Search Engine / Our Website
- Authorised to work in the country (UK): Yes
- Interview adjustments: I have dyslexia. I am fine with a spoken interview. If there is a written exercise, I will use my usual assistive tools, and a little extra reading time is enough.
- (Phone and LinkedIn were filled in on the form; omitted here.)

## Notes
- The answer is candid about the gaps (no large-scale SSL training, no distributed training), which is honest but this role lists distributed training as a requirement — expect that to come up.
