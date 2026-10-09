# VectorStackAI — AI/ML Research Engineer

- **Date:** 2026-10-09
- **Link:** https://wellfound.com/jobs/4662933-ai-ml-research-engineer
- **Company:** VectorStackAI (1–10 people, Amsterdam; "Optimizing Generative AI stack end-to-end"; Top 10% of responders). Founder: Shreyas Saxena. https://vectorstack.ai/
- **Comp:** $60k–$120k, no equity
- **Location:** Amsterdam; onsite or remote (everywhere)
- **Posted:** 1 week ago
- **Status:** Applied — Wellfound email "Your application has been submitted!" ("You applied today", expect to hear back in 1–2 weeks)
- **Proof:** `proof/2026-10-09_vectorstackai_ai-ml-research-engineer.png`

## Q1: Tell us about your most impressive accomplishment / demo / research result

I build ranking and representation models, and I judge them on the ordering and the out-of-sample behavior, not on a single accuracy number.

The result I'd point to first is in finance. I trained CatBoost rankers over roughly 6,000 copy-trading wallets and a cross-sectional ranker over about 300 perpetual markets. Validation was point-in-time and walk-forward, scored with rank-IC. The thing I actually care about: raw win rate was inflated, because a trader could open one position and close it a hundred times and every close counted as a win. I treated a round trip as one trade, rebuilt the features, and SHAP showed consistency beating that fake win rate. That's the same discipline a reranker needs — score candidates, order them, and don't trust a metric a single component can game.

The second result is a fine-tuning one, on a Qwen 2.5 7B LoRA (QLoRA, base loaded in 4-bit, base weights frozen, low-rank matrices on the attention projections). I had a pipeline where an off-the-shelf LLM generated descriptions on finance and price data, with point-in-time discipline the whole way so I wasn't leaking the future. I clustered the descriptions semantically and found that certain ones were consistently wrong — not random noise, the same wrong interpretation over and over. So I built a harness first: a second pass that broke the descriptions into questions and evaluated them with a multi-way confusion-matrix breakdown, nested hierarchically by opportunity zone and price condition. That harness defined the error taxonomy — that was the ground truth. Then I fine-tuned against it. On the out-of-sample walk-forward set, 80–90% of the specific error types the harness had identified were gone. I walked the model forward so validation stayed genuinely out-of-sample — that's the difference between "I fine-tuned a model" and "I fine-tuned a model and proved it out-of-sample."

I've also done representation learning where the embedding is the point. I trained an LSTM variational autoencoder to produce a compressed latent embedding of sequence data — encoder LSTM outputs a mean and log-variance, reparameterization trick to sample z, decoder LSTM reconstructs, loss is reconstruction plus KL. I then used that learned embedding downstream in gradient-boosted models, both alongside the raw features and as a replacement for them. It moved the metric slightly better than raw features alone — not a dramatic jump, but a real one, and it confirmed the latent representation was carrying signal the hand-built features weren't. Separately, I've built transformer models with learned embeddings as part of the architecture. I want to be precise: the LSTM VAE is a standalone embedding model, the transformer embeddings were the embedding layer inside a model I was training, not a separately trained retrieval-style embedding model. I haven't yet fine-tuned a text embedding model or a reranker for legal or finance — that's the direction I'm moving, and the ranking work above is the same shape.

Public demo from about a year ago: a Cohere-compatible reranking API on Qwen3-Reranker-0.6B, used to rank UI elements for agent actions — github.com/nodeblackbox/qwen3rerankerapicoherestyle. I served the pretrained model and debugged the causal-LM scoring path, the missing padding token, and the batch-size failure. I didn't fine-tune it. Most of the current ranking work is proprietary.

(Closing line offering a call, with phone number, omitted from this copy.)

## Other fields
- LinkedIn: https://www.linkedin.com/in/anas-nasseur
- GitHub: https://github.com/nodeblackbox

## Q2: Experience fine-tuning LLMs? Share an insight and the result (good or bad)

Yes. The one I'd point to is a Qwen 2.5 7B LoRA — QLoRA specifically, base loaded in 4-bit so it fit on one GPU, base weights frozen, small low-rank matrices on the attention projections, only those trained.

**The setup.** I had a pipeline where an off-the-shelf LLM generated descriptions on finance and price data — "this opportunity is going to do X." I kept point-in-time discipline the whole way so I wasn't leaking the future into the descriptions. Then I clustered the descriptions semantically to see what kinds of calls the model was actually making. What I found was that certain descriptions were consistently wrong — not random noise, the same wrong interpretation over and over. The model couldn't describe the situation properly. It would hallucinate because it didn't actually know what was going on.

**The insight.** The hard part of fine-tuning isn't the training — it's pinning down what is wrong before you touch the model. Not just that something is wrong. I built a harness first: a second pass that broke the descriptions down into questions, then evaluated those answers with a confusion-matrix-style comparison across multiple conditions at once. Not one binary matrix — a multi-way breakdown: did it consistently call up, did it consistently call down, broken out by opportunity zone, broken out by price condition, organized hierarchically like a tree so the conditions nested. That harness is what defined the error taxonomy — that's the ground truth. The fine-tune is only as good as the taxonomy you train against. If you fine-tune against a vague notion of "better," you get a vague improvement. If you fine-tune against a specific, measured, multi-way error taxonomy, you can actually verify the errors are gone.

**The result.** I measured it as per-error-type occurrence rate on the out-of-sample walk-forward set — base model versus fine-tuned, broken out by the same conditions the harness used. The specific error types the harness had identified were the consistent wrong interpretations, and after fine-tuning roughly 80 to 90 percent of those specific mistakes were gone. The descriptions that were consistently wrong before weren't wrong anymore. And the part that makes it real: I walked it forward. I couldn't just fine-tune and call it done — I had to walk the model forward so the validation stayed genuinely out-of-sample.

**A negative result.** I ran GRPO on a trading exit policy — the model picks the exit action, trail / take profit / stop, reward is the realized PnL, which fits because the reward only arrives after the trade closes so supervised labels are hard to write. It didn't work well: the reward signal was too sparse, and the entry model's 43% win rate capped what the exits could do. The lesson was that the training method was fine but the problem framing was wrong — no exit policy can outrun a weak entry, and a sparse terminal reward doesn't give the policy enough gradient to learn from.

**A different modality, same lesson.** The Parakeet fine-tune is current — fine-tuning on audio from my own dictation tool so it handles my voice, my corrections, and how I actually speak, instead of a generic checkpoint. A general model treats self-corrections and interrupted speech as errors; the adaptation is what teaches it those are the signal. And I've seen the failure mode directly: in my final-year project, a generic skin LoRA plus ControlNet on synthetic Stable Diffusion renders produced artefacts, because the adapter wasn't trained on real hand geometry. General adapter, wrong domain, doesn't transfer.

**On the embedding side.** I've trained an LSTM variational autoencoder to produce a compressed latent embedding of sequence data, and used it downstream in gradient-boosted models, alongside raw features and as a replacement; it performed slightly better than raw features alone. That's a standalone embedding model in the representation-learning sense, not a text embedding model. I have not yet fine-tuned a text embedding model or a reranker on legal or finance text. What I have done is ranking — CatBoost rankers on ~6,000 copy-trading wallets and a cross-sectional ranker over ~300 perpetual markets, validated point-in-time walk-forward with rank-IC — the same shape as a reranker: score candidates, order them, don't trust a metric a single component can game.

## Notes
- Answers are candid about gaps (no fine-tuned text embedding model or reranker yet), which suits a role built around embeddings/rerankers but expect follow-up on it.
- Compensation is low ($60k–$120k, no equity) for a research engineer role.
