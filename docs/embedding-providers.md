# Embedding providers (Milestone 5)

CubeQL retrieval remains model-independent. The core `cubeql` crate still accepts Arrow vectors
and numeric query vectors, and does not depend on a model runtime, tokenizer, repository walker
or CLI. The new `cubeql-embeddings` crate has no Arrow or CubeCL dependency. The application
layer `cubeql-code` composes providers with code indexing and retrieval.

## Provider contract

```rust
pub trait EmbeddingProvider {
    fn config(&self) -> &EmbeddingConfig;
    fn embed_documents(&mut self, documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError>;
}
```

Document batches must preserve input order and return exactly one vector per text. Queries have
a separate method because some models require different document/query prefixes. Providers must
describe all output-affecting settings in `EmbeddingConfig`: model, pinned revision, dimensions,
normalization, implementation version, pooling, token limit, both prefixes, and artifact hashes.
Custom providers are responsible for truthfully describing their implementation and honoring
that contract; metadata cannot prove how arbitrary external vectors were produced.

`EmbeddedQuery` tags a validated vector with this full configuration. It supports query encoding
through `embed_query(provider, text)` as well as precomputed vectors through `EmbeddedQuery::new`.
`PrecomputedProvider` supplies separate exact-text maps of document and query vectors for offline
workflows and deterministic tests. `HashProvider` retains the old deterministic lexical fixture;
it is not a learned model and hash collisions can produce unrelated hits.

The builder validates output counts, dimensions, finite values and declared L2 normalization.
Zero vectors are allowed and retain the engine's cosine-zero semantics. A provider that changes
its configuration during encoding is rejected. The implementation cannot detect a provider
silently shuffling otherwise valid vectors; input-order preservation is part of its contract.

## Local reference: MiniLM

Enable `cubeql-code/local-embeddings` (or `cubeql-embeddings/minilm`) explicitly. Default builds
and consumers of the core engine do not acquire these ML dependencies.

The reference is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Runtime dependencies are pinned to Candle 0.11.0 and
tokenizers 0.22.2. It uses CPU inference, FP32, a pinned uncased
tokenizer, right truncation to 256 tokens including special tokens, attention-mask-aware mean
pooling, and L2 normalization. Output width is 384. Attention padding is excluded both inside
BERT and during pooling, so mixing short/long inputs in a batch does not change their meaning.
The reference limits batches to 32 inputs; the CLI defaults to 8.

The model is a **general English sentence model**, not a code-specialized encoder or a multilingual
code-search solution. Natural-language/code smoke tests validate a few documented examples, not
repository-wide relevance. Long functions are still indexed whole but only their first 256 tokens
contribute to the vector. Model-specific chunking, code-oriented models, hybrid exact search and
a labeled retrieval benchmark remain follow-up work.

Sources: [model card and pooling recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[pinned model files](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41),
[Candle BERT implementation](https://docs.rs/candle-transformers/0.11.0/src/candle_transformers/models/bert.rs.html).

## CLI: download, index, query

From the CubeQL checkout, download the three public artifacts once (~91 MB):

```bash
cargo run -p cubeql-code --features local-embeddings -- model download \
  --directory /tmp/cubeql-minilm
```

The downloader uses a fixed revision and checks SHA-256 and size limits for every file. Valid
existing files are reused. Conflicting files are never replaced; each verified download is
published via a same-directory temporary file with no-clobber persistence. Network errors can
leave already completed model files, which a subsequent download reuses. Loading/encoding never
access the network, so a missing file fails explicitly instead of silently downloading it.

Start with the tiny checked-in example (three documented Rust functions):

```bash
cargo run -p cubeql-code --features local-embeddings -- index \
  crates/cubeql-code/tests/fixtures/semantic-demo \
  --embedding minilm --model-dir /tmp/cubeql-minilm --batch-size 8 \
  --output /tmp/demo-minilm-v1.arrow
```

Query using the same local artifacts (cosine, default top-5):

```bash
cargo run -p cubeql-code --features local-embeddings -- query \
  /tmp/demo-minilm-v1.arrow "Where do we read configuration from a file?" \
  --model-dir /tmp/cubeql-minilm --top-k 5
```

The CLI selects a provider from the saved configuration and verifies its full identity before
uploading vectors. There is no fallback to token hashing if model loading fails. Plain-text
output contains file path, inclusive line range, symbol and cosine score, not a relevance
probability. `query` is the minimal provider workflow, not the full filtering/JSON/benchmark CLI
planned for Milestone 6. It currently supports the built-in hash and MiniLM providers only.

On Apple hardware, choose native Metal for **retrieval**, while MiniLM inference remains on CPU:

```bash
cargo run -p cubeql-code --no-default-features --features metal,local-embeddings -- query \
  /tmp/demo-minilm-v1.arrow "Read configuration settings from disk" \
  --model-dir /tmp/cubeql-minilm
```

Changing the retrieval backend does not change the embedding contract. CPU and Metal can load
the same file. No hosted embedding API or upload of repository contents is involved.

Replace the fixture directory with a real repository or source subdirectory when ready. Full
repository encoding can be expensive on the CPU, especially in debug builds; use
`cargo run --release` for throughput measurements and start with a small directory. No full-CubeCL
model encoding throughput or repository-wide relevance claim is made by this milestone.

## Incremental indexing and persistence

```bash
cargo run -p cubeql-code --features local-embeddings -- index \
  crates/cubeql-code/tests/fixtures/semantic-demo \
  --embedding minilm --model-dir /tmp/cubeql-minilm \
  --previous /tmp/demo-minilm-v1.arrow --output /tmp/demo-minilm-v2.arrow
```

Only unmatched chunk sources are batched for encoding. Reused and new vectors are assembled
in the current chunk order (old row IDs are not stable); duplicate source occurrences are accounted for
separately. Unchanged repositories perform zero document embedding calls, although the CLI
currently loads the model and scans/parses the repository again and rewrites the full snapshot.

The `cubeql.code.embedding_config` schema metadata key stores the serialized full contract.
`cubeql.code.embedding_id` is its versioned BLAKE3 fingerprint, and the dimension column metadata
must agree. The code reader checks that these agree and validates declared vector normalization.
Older callback-only indexes are still readable by `CodeIndex`/`VectorIndex` and the legacy hash
example, but have no verified provider contract: the new guarded query and provider-update paths
reject them with a rebuild instruction. Rebuild from source into a **new** output path; a token-hash
index cannot be converted into a learned-model index by changing its metadata.

The old `build_repository` callback API remains available for raw/precomputed workflows. It does
not invent a provider contract or offer model-aware querying. New applications should prefer
`BuildOptions::for_embedding`, `build_repository_with_provider` and `CodeSearchIndex`.

## Library composition

```rust
use cubeql::Metric;
use cubeql_code::{BuildOptions, CodeSearchIndex, build_repository_with_provider};
use cubeql_embeddings::{EmbeddingProvider, MiniLm}; // feature: minilm

let mut model = MiniLm::from_dir("/tmp/cubeql-minilm")?;
let options = BuildOptions::for_embedding(model.config())?;
let built = build_repository_with_provider("../cubecl", &options, None, &mut model, 8)?;
built.index.save_ipc("/tmp/new-minilm-index.arrow")?;
let search = CodeSearchIndex::new(&cubecl::Device::default(), built.index)?;
let hits = search.query_text(&mut model, "Read configuration settings", 5, Metric::Cosine)?;
for hit in hits {
    println!("{}", search.code().chunks()[hit.row].symbol);
}
Ok::<(), Box<dyn std::error::Error>>(())
```

`CodeSearchIndex` owns both the code snapshot and the resident `VectorIndex`, preventing accidental
row/metadata mismatches. It rejects a wrong provider before query inference, and rejects an
incompatible tagged query before device scoring, even if dimensions match. It exposes all three
existing metrics. Core `VectorIndex` intentionally remains numeric/model-agnostic; callers bypassing
this application wrapper must enforce their own model compatibility. Model metadata is supplied
by providers/callers, not a security attestation.

## Tests

Default tests are offline and do not require a model download:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Explicit local-model tests require downloaded files and never download from inside the test:

```bash
CUBEQL_MINILM_DIR=/tmp/cubeql-minilm cargo test -p cubeql-embeddings --features minilm \
  --test providers local_minilm -- --ignored
CUBEQL_MINILM_DIR=/tmp/cubeql-minilm cargo test -p cubeql-code --features local-embeddings \
  --test providers real_local_model -- --ignored
```

These check normalization, short/long batch padding, empty/truncated inputs, basic sentence
similarity, full IPC persistence, and natural-language queries against a tiny documented code
fixture. They are not a substitute for the next milestone's relevance benchmark. Deterministic
precomputed providers cover batch order, incremental reuse, incompatible configs, bad dimensions,
NaNs, failed batches, tampered metadata and model-aware queries without ML dependencies.
