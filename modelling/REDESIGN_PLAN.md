# Zephyr Model Redesign: All-Station Prediction with Transformer Architecture

## Assessment & Implementation Plan

**Date**: 2026-03-19
**Goal**: Best possible forecasting results for the least effort

---

## 1. Current State

The current system trains a **PatchTST** model on a regional subset of stations (e.g. all stations within 30km of Coronet Peak). Key characteristics:

- **Data**: ~600k rows, 19 stations, 10-minute intervals
- **Input**: 60 timesteps (10 hours) of 10 features (4 weather + 6 temporal)
- **Output**: 6 timesteps (1 hour) of 4 weather variables
- **Multi-station handling**: Each station's time series creates independent sliding windows via `SlidingWindowPanel(unique_id_cols=["station_id"])`. Stations are mixed in training batches but the model sees one station at a time per sample — it does **not** see cross-station spatial information within a single forward pass
- **Regional filtering**: SQL query selects stations within X km of a centre station

### Problems with Current Approach

1. **No spatial modelling**: The model treats each station independently. It cannot learn that wind arriving at Station A predicts conditions at Station B 30 minutes later
2. **Regional subsetting is arbitrary**: The 30km radius is a manual choice. Including all stations would provide more training data and enable the model to learn NZ-wide weather dynamics
3. **Retraining per region**: Each new area of interest requires a new SQL query, new CSV export, and new model — significant operational overhead

---

## 2. Proposed Architecture: All-Station Transformer

### 2.1 Can Transformers Predict Multiple Output Variables?

**Yes, absolutely.** This is not just feasible — it's what modern time series transformers are designed for. The key consideration is whether to use a **channel-independent** or **channel-dependent** approach:

| Approach | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **Channel-Independent** (current PatchTST default) | Each variable is processed independently through the same transformer backbone | Simpler, avoids overfitting on small datasets, proven strong baseline | Cannot capture cross-variable relationships |
| **Channel-Dependent** | Variables interact within the attention mechanism | Captures physical relationships (wind→temperature, gust→average) | Needs more data, risk of overfitting |

**Recommendation**: For weather data where variables are physically coupled (wind speed, gusts, bearing, and temperature all interact), a **channel-dependent approach is better** — and our dataset (600k rows × 19 stations) is large enough to support it.

### 2.2 Architecture Options Evaluated

#### Option A: iTransformer (Recommended)

**What it is**: An "inverted" transformer (ICLR 2024 Spotlight) that treats each variable as a token instead of each timestep. This is a natural fit for multivariate forecasting with cross-variable dependencies.

**Why it's ideal for us**:
- **Channel-dependent by design**: Attention operates across variables, so it naturally captures wind→temperature relationships. The attention weights are also interpretable — you can visualise which variables the model considers correlated
- **Handles multi-station data well**: We can flatten station×variable into the token dimension, so the model sees all 19 stations' variables simultaneously in one forward pass. With 19 stations × 4 variables = 76 tokens, the attention is lightweight
- **State-of-the-art results**: Outperforms PatchTST on most multivariate benchmarks, especially when cross-variable relationships matter
- **Efficient**: Attention is over variables (4-76 tokens) rather than over time patches (many more tokens), so it can actually be more efficient than PatchTST when you have many timesteps but few variables

**Availability**: NOT in tsai. Available via:
- `pip install iTransformer` (standalone package by lucidrains)
- [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib) — the official implementation
- NeuralForecast, GluonTS

Would require replacing the tsai `TSForecaster` with TSLib's training loop, but the data preparation code can be largely reused.

**Input format**: Instead of (batch, 60_timesteps, N_features), iTransformer treats (batch, N_features, 60_timesteps) — each variable-station pair becomes a token with its 60-timestep history as the token embedding.

#### Option B: PatchTST with Channel-Dependent Mode

**What it is**: The architecture we already use, but switched from channel-independent to channel-dependent mode.

**Why consider it**:
- Zero migration effort — just change a config flag
- Well-understood, proven architecture
- But: channel-dependent PatchTST tends to underperform iTransformer on benchmarks

#### Option C: Foundation Models (TimesFM / Chronos-2)

**TimesFM (Google)**: Decoder-only foundation model, 200M-500M params. Fundamentally **univariate** — does not jointly model multiple output variables. Poor fit for our use case.

**Chronos-2 (Amazon)**: More interesting. 120M-param encoder-only model with a novel **group attention** mechanism that alternates between time attention (within each series) and group attention (across all series). This gives it true multivariate capabilities. Zero-shot capable with no training needed. Produces **probabilistic forecasts** (quantiles), which could be more useful for safety-critical paragliding decisions (uncertainty quantification). Open source, Apache 2.0.

**Verdict**: TimesFM is not suitable. Chronos-2 is worth testing as a **zero-shot baseline** — if it performs well out of the box, it eliminates training entirely. However, a domain-specific trained model will likely outperform it on our specific dataset.

#### Option D: TSMixer (MLP-based)

An all-MLP architecture that alternates between time-mixing and feature-mixing layers. Channel-dependent via the feature-mixing component. **2-3x faster** than PatchTST with lower memory. Available in HuggingFace Transformers as `PatchTSMixer`. Good balance of simplicity and performance, but less expressive than transformers for complex spatial-temporal patterns.

#### Option E: TSTPlus (Quick Win via tsai)

Already available in tsai — works with a one-line change (`arch="TSTPlus"`). Channel-dependent unlike PatchTST. Not as modern as iTransformer, but **zero migration effort**. Good as a quick experiment before committing to a larger rewrite.

### 2.3 Why Channel-Dependent Matters for Weather Data

A 2025 analysis using Granger causality confirmed that channel-independent models only dominate benchmarks where cross-channel dependencies are weak. For weather data specifically:

1. **Physical coupling**: Temperature, wind speed, gusts, and bearing are governed by shared atmospheric dynamics. Pressure gradients drive wind, wind advects temperature, gusts are caused by thermal instability
2. **The benchmark illusion**: PatchTST wins on standard benchmarks (ETTh, Weather, Traffic) partly because those datasets have weak inter-channel correlations — NOT the case for our data
3. **Joint distribution matters**: For paragliding safety, you care about combinations (high gust AND shifting bearing), not individual forecasts
4. **Precedent**: State-of-the-art weather models (GraphCast, GenCast, Pangu-Weather) all use shared representations across variables

### 2.3 Recommended Architecture: iTransformer with All-Station Input

**The key insight**: Instead of creating independent windows per station, we reshape the data so that one training sample contains **all 19 stations' observations at a given time window**. The model sees the full spatial picture.

```
Current approach (per-station windows):
  X shape: (num_samples, 60, 10)  — one station's 60 timesteps × 10 features
  y shape: (num_samples, 6, 4)   — one station's 6 future timesteps × 4 variables

Proposed approach (all-station windows):
  X shape: (num_samples, 60, 19×4 + 6)  — 60 timesteps × (76 station-variables + 6 temporal)
  y shape: (num_samples, 6, 19×4)        — 6 future timesteps × 76 station-variables
```

With iTransformer, each of the 76 station-variable pairs becomes a token. The attention mechanism learns which stations influence which — e.g., that windward stations predict leeward conditions.

**Fallback**: If 76 tokens is too many for the dataset size, we can group variables per station (4 variables → 1 station token with 4-dimensional embedding), reducing to 19 tokens.

---

## 3. Training Automation

### 3.1 Karpathy's `autoresearch` Pattern

**What it is**: A design pattern (not a library) where an AI coding agent autonomously runs ML experiments in a loop. Released March 2026, ~43k GitHub stars, MIT license.

**The loop**:
1. Agent reads `program.md` (human-written instructions) and `train.py`
2. Agent forms a hypothesis and modifies `train.py`
3. Agent commits the change to git
4. Training runs for a fixed time budget (e.g. 5 minutes)
5. If the metric improved → keep. If not → `git revert`
6. Results logged to `results.tsv`
7. Repeat indefinitely (~12 experiments/hour)

**How we'd use it**:
- Write a `program.md` that instructs Claude Code to optimise our weather forecasting model
- The agent would experiment with: learning rates, batch sizes, window lengths, attention heads, layer counts, feature engineering, normalisation strategies
- Fixed time budget per experiment (5-10 min depending on GPU)
- Metric to optimise: MAE on validation set (weighted across wind variables)
- All experiments tracked via git commits + TSV log

**Effort to adopt**: Low-medium. We need to:
1. Restructure `train.py` to be a self-contained single file that prints the target metric
2. Write `program.md` with domain-specific instructions
3. Point Claude Code at it and let it run overnight

**Expected value**: High. The agent can test 100+ hyperparameter/architecture combinations overnight — work that would take weeks manually.

### 3.2 SkyPilot for Cloud GPU Training

**What it is**: An open-source framework (UC Berkeley, ~10k stars) for running ML jobs on any cloud with automatic cost optimisation.

**Key features**:
- **Multi-cloud**: Works across AWS, GCP, Azure, Lambda, RunPod, and 15+ other providers
- **Spot instances**: Automatic provisioning of spot/preemptible GPUs with 70-90% cost savings
- **Preemption recovery**: Automatic checkpointing and job restart if a spot instance is reclaimed
- **Simple YAML config**: Define your job once, run anywhere
- **No vendor lock-in**: Switch clouds with a flag change

**Example SkyPilot config for our project**:
```yaml
# sky.yaml
name: zephyr-train

resources:
  accelerators: T4:1   # or A100:1 for faster training
  use_spot: true        # 70-90% cheaper
  cloud: aws            # or gcp, azure, lambda, etc.

workdir: .

setup: |
  pip install uv
  uv sync

run: |
  python modelling/train.py
```

**Launch**: `sky launch sky.yaml` — SkyPilot finds the cheapest available spot GPU across regions and clouds.

**Cost comparison**:
| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Current (local CPU) | None | ~8 hours | Free but slow |
| SkyPilot spot T4 | T4 | ~30 min | ~$0.10-0.15 |
| SkyPilot spot A100 | A100 | ~10 min | ~$0.15-0.25 |
| Lambda Labs (on-demand) | A100 | ~10 min | ~$0.18 |

**Effort to adopt**: Very low. Install SkyPilot, write the YAML, run `sky launch`. Cloud credentials are the main setup step.

### 3.3 Combining autoresearch + SkyPilot

The optimal workflow:

1. **SkyPilot** provisions a cheap spot GPU instance
2. **autoresearch pattern** runs Claude Code on that instance, iterating experiments overnight
3. Each experiment trains for 5-10 minutes on the GPU
4. ~100 experiments run overnight for ~$1-3 total compute cost
5. Morning: review `results.tsv`, pick the best model

This gives us automated hyperparameter search and architecture tuning on cloud GPUs for negligible cost.

---

## 4. Implementation Plan

### Phase 1: Data Pipeline Refactor (Low effort, High impact)

**Goal**: Produce all-station training data instead of regional subsets.

**Changes**:
1. **New SQL query** (`modelling/sql/all_stations.sql`): Select all observations from all stations, no distance filtering
2. **Update `train.py`**: Remove the `get_training_data(station_name, radius)` pattern. Just load all data
3. **Reshape data for all-station input**: Pivot from long format (one row per station-timestamp) to wide format (one row per timestamp, columns for each station's variables)
4. **Handle missing data**: Some stations may have gaps. Use forward-fill then NaN masking

**Estimated effort**: 1-2 hours

### Phase 2: Architecture Experiments (Low-Medium effort, High impact)

**Goal**: Find the best architecture for cross-station/cross-variable modelling.

**Suggested experiment ladder** (each compared on the same validation set):

1. **Baseline**: Current PatchTST (channel-independent) on new all-station data
2. **Quick win**: Switch to `TSTPlus` in tsai (`arch="TSTPlus"`) — channel-dependent, zero migration effort
3. **Zero-shot test**: Try Chronos-2 with no training at all (`pip install chronos-forecasting`) — if competitive, it's the lowest-effort option
4. **Best channel-dependent**: iTransformer via TSLib or lucidrains package — requires replacing tsai training loop but reuses data prep
5. **If compute matters**: TSMixer/PatchTSMixer — 2-3x faster training

**Changes for iTransformer (primary target)**:
1. Install via `pip install iTransformer` or clone [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
2. Replace the tsai `TSForecaster` with TSLib's training loop
3. Adjust input shape to (batch, n_variates, timesteps) format
4. Update evaluation: MAE per station per variable, plus aggregate metrics

**Estimated effort**: 2-4 hours for steps 1-2 (one-line changes), 4-6 hours for step 4 (new training loop)

**Fallback**: TSTPlus in tsai is a one-line change and still an improvement over channel-independent PatchTST.

### Phase 3: SkyPilot Integration (Very low effort, High impact)

**Goal**: Enable cheap cloud GPU training.

**Changes**:
1. Write `sky.yaml` config file
2. Add `skypilot` to dev dependencies
3. Add checkpointing to `train.py` (save model every N epochs for spot instance recovery)
4. Document the `sky launch` workflow

**Estimated effort**: 30 minutes - 1 hour

### Phase 4: autoresearch Setup (Medium effort, Very high impact)

**Goal**: Automate hyperparameter and architecture search.

**Changes**:
1. **Refactor `train.py`** into a self-contained script that:
   - Takes all config via command-line args or a config section at the top of the file
   - Prints a single metric (e.g. `val_mae: 0.xxx`) at the end
   - Runs within a fixed time budget
2. **Write `program.md`** with instructions for the agent:
   - What to optimise (validation MAE, weighted toward wind variables)
   - What it can change (architecture params, learning rate, batch size, window length, feature engineering)
   - What it must not change (data loading, evaluation, metric calculation)
   - Domain context (weather data, station relationships, physical constraints)
3. **Set up the experiment loop**: Configure Claude Code with autonomous permissions, point at `program.md`
4. **Run overnight on SkyPilot GPU**

**Estimated effort**: 3-4 hours for initial setup, then it runs unattended

### Phase 5: Evaluation & Deployment (Low effort)

**Goal**: Validate the best model and deploy.

**Changes**:
1. Compare best autoresearch model against current PatchTST baseline
2. Run on held-out test set (time-based split, not random)
3. Export best model as `.pkl`
4. Update inference code to handle all-station input format

**Estimated effort**: 1-2 hours

---

## 5. Recommended Execution Order

```
Phase 1 (Data Pipeline)     ████░░░░░░  ~2 hrs
Phase 2 (iTransformer)      ████░░░░░░  ~3 hrs
Phase 3 (SkyPilot)          █░░░░░░░░░  ~1 hr
Phase 4 (autoresearch)      ████░░░░░░  ~4 hrs
Phase 5 (Evaluate)          ██░░░░░░░░  ~2 hrs
                            ─────────────────
Total hands-on effort:       ~12 hours
Overnight compute:           ~$2-5 (spot GPUs)
```

Phases 1+2 should be done together (data format and model are coupled). Phase 3 is independent and can be done in parallel. Phase 4 depends on 1+2 being complete.

---

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| iTransformer not in tsai | Medium | Use lucidrains implementation or fall back to PatchTST channel-dependent |
| All-station input too large for memory | Low | Reduce window length, use gradient accumulation, or subsample stations |
| Spot instance preemption during autoresearch | Medium | SkyPilot handles recovery; experiments are short (5 min) so little work lost |
| autoresearch agent makes poor changes | Low | Keep/revert mechanism ensures only improvements survive |
| Worse performance than current model | Low | Keep current model as baseline; only deploy if new model beats it |

---

## 7. Summary of Recommendations

1. **Predict all stations at once** — reshape data from per-station windows to all-station-per-timestep. This is the single highest-impact change: it enables spatial learning
2. **Switch to iTransformer** — purpose-built for multivariate forecasting with cross-variable attention. Minimal code change from current tsai-based pipeline
3. **Use SkyPilot for GPU training** — 30 minutes of setup saves hours of training time and costs pennies via spot instances
4. **Adopt the autoresearch pattern** — let Claude Code run 100+ experiments overnight to find optimal hyperparameters. This replaces weeks of manual tuning
5. **Test Chronos-2 as a zero-shot baseline** — it has true multivariate support and needs no training. If it's competitive, it's the lowest-effort path. But a domain-specific iTransformer will likely beat it
6. **Skip TimesFM** — it's univariate-only for outputs

The total investment is roughly 12 hours of hands-on work plus an overnight compute run. The expected outcome is a model that:
- Predicts all 19 stations simultaneously (vs. one region at a time)
- Captures spatial weather dynamics across stations
- Has automatically-optimised hyperparameters
- Trains on cloud GPUs for under $5
- Eliminates the per-region workflow entirely
