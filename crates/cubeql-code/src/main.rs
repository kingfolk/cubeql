use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use cubeql_code::{
    BuildOptions, BuildResult, CodeIndex, CodeSearchIndex, VectorPrecision,
    benchmark::{self, Dataset, RunOptions},
    build_repository_with_provider,
    evaluation::{EvaluationOptions, QuerySet},
    hybrid_evaluation::{evaluate, evaluate_with_candidates},
    prepare_repository,
    reporting::{CodeFilter, RetrievalResponse, SearchTimings, SnapshotInfo, elapsed_ms},
    retrieval::{RetrievalOptions, SearchStrategy, SnapshotTextIndex, materialize_hits},
};
use cubeql_embeddings::{
    BundleProvider, EmbeddingConfig, EmbeddingError, EmbeddingInput, EmbeddingInputs,
    EmbeddingProvider, EncoderProcessConfig, HashProvider, ProcessProvider, VectorBundle,
    embed_identified_query, embed_query,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(
    version,
    about = "Build, search and evaluate portable Arrow code indexes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum ProviderKind {
    Hash,
    Minilm,
    Precomputed,
    Process,
}

#[derive(Clone, Copy, Default, ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Clone, Copy, ValueEnum)]
enum Strategy {
    Semantic,
    Lexical,
    Symbol,
    Hybrid,
}
impl From<Strategy> for SearchStrategy {
    fn from(value: Strategy) -> Self {
        match value {
            Strategy::Semantic => Self::Semantic,
            Strategy::Lexical => Self::Lexical,
            Strategy::Symbol => Self::Symbol,
            Strategy::Hybrid => Self::Hybrid,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    Fp16,
    Fp32,
}
impl From<Precision> for VectorPrecision {
    fn from(value: Precision) -> Self {
        match value {
            Precision::Fp16 => Self::Fp16,
            Precision::Fp32 => Self::Fp32,
        }
    }
}

#[derive(Args)]
struct Output {
    #[arg(long, value_enum, default_value = "text")]
    format: OutputFormat,
}

#[derive(Subcommand)]
enum Command {
    /// Export exact source/query inputs for an external local encoder.
    Vectors {
        #[command(subcommand)]
        command: VectorCommand,
    },
    /// Index source files. Existing output files are never overwritten.
    Index {
        root: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        previous: Option<PathBuf>,
        /// Hash is an offline lexical fixture, not semantic search.
        #[arg(long, value_enum, default_value = "hash")]
        embedding: ProviderKind,
        #[arg(long, conflicts_with_all = ["encoder_config", "vectors"])]
        model_dir: Option<PathBuf>,
        /// Import an offline ID/content-bound vector bundle; never performs inference.
        #[arg(long, conflicts_with_all = ["encoder_config", "model_dir"])]
        vectors: Option<PathBuf>,
        /// Launch an explicitly configured resident local encoder; never downloads models.
        #[arg(long, conflicts_with_all = ["vectors", "model_dir"])]
        encoder_config: Option<PathBuf>,
        /// Hash defaults to 128; MiniLM has a fixed width of 384.
        #[arg(long)]
        dimensions: Option<usize>,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 80)]
        fallback_lines: usize,
        #[arg(long, default_value_t = 2 * 1024 * 1024)]
        max_file_bytes: u64,
        /// Defaults to FP32, or preserves the previous snapshot's precision.
        #[arg(long, value_enum)]
        precision: Option<Precision>,
        #[command(flatten)]
        display: Output,
    },
    /// Reuse a snapshot's root, chunking options, provider contract and vector precision.
    Update {
        index: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Override the saved root if the repository was moved.
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long, conflicts_with_all = ["encoder_config", "vectors"])]
        model_dir: Option<PathBuf>,
        /// Import an offline ID/content-bound vector bundle; never performs inference.
        #[arg(long, conflicts_with_all = ["encoder_config", "model_dir"])]
        vectors: Option<PathBuf>,
        /// Launch an explicitly configured resident local encoder; never downloads models.
        #[arg(long, conflicts_with_all = ["vectors", "model_dir"])]
        encoder_config: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[command(flatten)]
        display: Output,
    },
    /// Inspect host metadata without loading an embedding model or initializing a device.
    Inspect {
        index: PathBuf,
        #[command(flatten)]
        display: Output,
    },
    /// Search saved code using vector, lexical, syntax-symbol or fused rankings.
    #[command(visible_alias = "query")]
    Search {
        index: PathBuf,
        text: String,
        /// Disambiguate identical query text associated with multiple imported IDs.
        #[arg(long, requires = "vectors")]
        query_id: Option<String>,
        #[arg(long, conflicts_with_all = ["encoder_config", "vectors"])]
        model_dir: Option<PathBuf>,
        /// Import an offline ID/content-bound vector bundle; never performs inference.
        #[arg(long, conflicts_with_all = ["encoder_config", "model_dir"])]
        vectors: Option<PathBuf>,
        /// Launch an explicitly configured resident local encoder; never downloads models.
        #[arg(long, conflicts_with_all = ["vectors", "model_dir"])]
        encoder_config: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, value_enum, default_value = "semantic")]
        strategy: Strategy,
        /// Candidates per source. Defaults to max(50, top-k), before overlap deduplication.
        #[arg(long)]
        candidate_k: Option<usize>,
        /// Keep highly overlapping source ranges instead of suppressing them.
        #[arg(long)]
        no_dedup: bool,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        path_prefix: Option<String>,
        /// Include source from the saved snapshot, not the current working tree.
        #[arg(long)]
        include_source: bool,
        #[command(flatten)]
        display: Output,
    },
    /// Compare semantic, literal and experimental RRF rankings on labeled queries.
    Evaluate {
        index: PathBuf,
        queries: PathBuf,
        #[arg(long, conflicts_with_all = ["encoder_config", "vectors"])]
        model_dir: Option<PathBuf>,
        /// Import an offline ID/content-bound vector bundle; never performs inference.
        #[arg(long, conflicts_with_all = ["encoder_config", "model_dir"])]
        vectors: Option<PathBuf>,
        /// Launch an explicitly configured resident local encoder; never downloads models.
        #[arg(long, conflicts_with_all = ["vectors", "model_dir"])]
        encoder_config: Option<PathBuf>,
        #[arg(long, default_value_t = 50)]
        candidate_k: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Derive FP16 storage from this original FP32 snapshot; encode queries only once.
        #[arg(long)]
        compare_fp16: bool,
        /// Include complete retained candidate windows (JSON output only).
        #[arg(long, requires = "format")]
        export_candidates: bool,
        #[command(flatten)]
        display: Output,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Import external corpus/query/qrels data and export native document rankings.
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommand,
    },
}

#[derive(Subcommand)]
enum VectorCommand {
    /// Export new/changed repository chunks and optional JSONL queries. No model is loaded.
    Prepare {
        root: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        previous: Option<PathBuf>,
        #[arg(long)]
        queries: Option<PathBuf>,
        #[arg(long, conflicts_with = "queries")]
        query: Option<String>,
        #[arg(long, default_value = "query")]
        query_id: String,
        #[arg(long)]
        fallback_lines: Option<usize>,
        #[arg(long)]
        max_file_bytes: Option<u64>,
    },
    /// Export a query-only input JSON; model inference runs separately.
    Query {
        text: String,
        #[arg(long, default_value = "query")]
        id: String,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum BenchmarkCommand {
    /// Export canonical document/query inputs for external encoding; qrels are not included.
    Prepare {
        dataset: PathBuf,
        #[arg(long, default_value = "test")]
        split: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Validate all IDs/judgments and fingerprint inputs without loading a model or device.
    Validate {
        dataset: PathBuf,
        #[arg(long, default_value = "test")]
        split: String,
    },
    /// Exact cosine retrieval; preserve document units/IDs and leave scoring to the evaluator.
    Run {
        dataset: PathBuf,
        #[arg(long, default_value = "test")]
        split: String,
        /// Must be a new directory. Failures leave diagnostics; retries use another directory.
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "hash")]
        embedding: ProviderKind,
        #[arg(long, conflicts_with_all = ["encoder_config", "vectors"])]
        model_dir: Option<PathBuf>,
        /// Import an offline ID/content-bound vector bundle; never performs inference.
        #[arg(long, conflicts_with_all = ["encoder_config", "model_dir"])]
        vectors: Option<PathBuf>,
        /// Launch an explicitly configured resident local encoder; never downloads models.
        #[arg(long, conflicts_with_all = ["vectors", "model_dir"])]
        encoder_config: Option<PathBuf>,
        #[arg(long)]
        dimensions: Option<usize>,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 100)]
        top_k: usize,
        #[arg(long, default_value = "cubeql-semantic")]
        run_tag: String,
        /// Optional caller-supplied provenance, not a build-attested revision.
        #[arg(long)]
        engine_revision: Option<String>,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download the pinned MiniLM files (~91 MB); never replace conflicting files.
    Download {
        #[arg(long)]
        directory: PathBuf,
    },
}

fn provider(
    kind: ProviderKind,
    dimensions: Option<usize>,
    directory: Option<&Path>,
    vectors: Option<&Path>,
    encoder_config: Option<&Path>,
) -> Result<Box<dyn EmbeddingProvider>> {
    if vectors.is_some() && !matches!(kind, ProviderKind::Precomputed) {
        return Err("--vectors requires --embedding precomputed".into());
    }
    if encoder_config.is_some() && !matches!(kind, ProviderKind::Process) {
        return Err("--encoder-config requires --embedding process".into());
    }
    match kind {
        ProviderKind::Process => {
            if directory.is_some() || vectors.is_some() {
                return Err("--encoder-config conflicts with --model-dir and --vectors".into());
            }
            let options = EncoderProcessConfig::load(
                encoder_config.ok_or("process requires --encoder-config")?,
            )?;
            let provider = ProcessProvider::spawn(&options, None)?;
            if dimensions.is_some_and(|d| d != provider.config().dimensions) {
                return Err("--dimensions disagrees with encoder configuration".into());
            }
            Ok(Box::new(provider))
        }
        ProviderKind::Precomputed => {
            if directory.is_some() {
                return Err("--model-dir cannot be used with imported vectors".into());
            }
            let bundle = VectorBundle::load(vectors.ok_or("precomputed requires --vectors")?)?;
            if dimensions.is_some_and(|d| d != bundle.config().dimensions) {
                return Err("--dimensions disagrees with vector bundle".into());
            }
            Ok(Box::new(BundleProvider::new(bundle)))
        }
        ProviderKind::Hash => {
            if directory.is_some() {
                return Err("--model-dir is only used with MiniLM".into());
            }
            eprintln!("Lexical fixture only: token hashes are not semantic embeddings.");
            Ok(Box::new(HashProvider::new(dimensions.unwrap_or(128))?))
        }
        ProviderKind::Minilm => {
            if dimensions.is_some_and(|d| d != 384) {
                return Err("MiniLM dimensions are fixed at 384".into());
            }
            #[cfg(feature = "local-embeddings")]
            {
                let directory =
                    directory.ok_or("MiniLM requires --model-dir; run model download first")?;
                Ok(Box::new(cubeql_embeddings::MiniLm::from_dir(directory)?))
            }
            #[cfg(not(feature = "local-embeddings"))]
            {
                let _ = directory;
                Err("rebuild with --features local-embeddings to enable MiniLM".into())
            }
        }
    }
}

fn saved_provider(
    code: &CodeIndex,
    directory: Option<&Path>,
    vectors: Option<&Path>,
    encoder_config: Option<&Path>,
) -> Result<Box<dyn EmbeddingProvider>> {
    let config = code
        .embedding_config()
        .ok_or("legacy index has no provider contract; rebuild with index")?;
    if let Some(path) = encoder_config {
        if directory.is_some() || vectors.is_some() {
            return Err("--encoder-config conflicts with --model-dir and --vectors".into());
        }
        let options = EncoderProcessConfig::load(path)?;
        return Ok(Box::new(ProcessProvider::spawn(&options, Some(config))?));
    }
    if let Some(path) = vectors {
        let provider = provider(
            ProviderKind::Precomputed,
            Some(config.dimensions),
            directory,
            Some(path),
            None,
        )?;
        config.ensure_compatible(provider.config())?;
        return Ok(provider);
    }
    let kind = match config.model.as_str() {
        "cubeql-hash-tokens" => ProviderKind::Hash,
        "sentence-transformers/all-MiniLM-L6-v2" => ProviderKind::Minilm,
        _ => {
            if directory.is_some() {
                return Err("external models require --vectors rather than --model-dir".into());
            }
            // A no-op/removal-only update needs no new embeddings. Any attempted inference
            // still fails explicitly; unknown models never fall back to token hashing.
            return Ok(Box::new(ImportedOnly(config.clone())));
        }
    };
    let provider = provider(kind, Some(config.dimensions), directory, None, None)?;
    config.ensure_compatible(provider.config())?;
    Ok(provider)
}

struct ImportedOnly(EmbeddingConfig);
impl EmbeddingProvider for ImportedOnly {
    fn config(&self) -> &EmbeddingConfig {
        &self.0
    }
    fn embed_documents(
        &mut self,
        _: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(EmbeddingError::Invalid("new source needs external embeddings: supply --encoder-config, or run vectors prepare, encode, then update --vectors".into()))
    }
    fn embed_query(&mut self, _: &str) -> std::result::Result<Vec<f32>, EmbeddingError> {
        Err(EmbeddingError::Invalid("this CLI does not implement a built-in encoder for this model; external query embedding required: supply --encoder-config, or run vectors query, encode with the same model configuration, then search --vectors".into()))
    }
}

fn write_inputs(output: &Path, inputs: &EmbeddingInputs) -> Result<()> {
    inputs.validate()?;
    benchmark::write_json(output, inputs)?;
    println!(
        "documents={} queries={} output={}",
        inputs.documents.len(),
        inputs.queries.len(),
        output.display()
    );
    Ok(())
}

fn new_output(path: &Path) -> Result<()> {
    if path.exists() {
        return Err("output already exists; choose a new snapshot path".into());
    }
    Ok(())
}

fn print_build(result: BuildResult, output: &Path, format: OutputFormat) -> Result<()> {
    result.index.save_ipc(output)?; // create_new in the writer also protects against races.
    for warning in &result.report.warnings {
        eprintln!("warning: {warning}");
    }
    match format {
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "format_version": 1, "output": output, "report": result.report,
                "snapshot": SnapshotInfo::new(&result.index)?
            }))?
        ),
        OutputFormat::Text => {
            let r = result.report;
            println!(
                "files={} chunks={} embedded={} reused={} added={} changed={} removed={}",
                r.files, r.chunks, r.embedded, r.reused, r.added, r.changed, r.removed
            );
            println!(
                "embedding={} dimensions={} precision={:?} output={}",
                result
                    .index
                    .embedding_config()
                    .map(|c| c.model.as_str())
                    .unwrap_or(result.index.embedding_id()),
                result.index.dimensions(),
                result.index.precision(),
                output.display()
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Vectors { command } => match command {
            VectorCommand::Query { text, id, output } => write_inputs(
                &output,
                &EmbeddingInputs {
                    format_version: 1,
                    documents: Vec::new(),
                    queries: vec![EmbeddingInput::new(id, text)],
                },
            )?,
            VectorCommand::Prepare {
                root,
                output,
                previous,
                queries,
                query,
                query_id,
                fallback_lines,
                max_file_bytes,
            } => {
                let previous = previous.map(CodeIndex::load_ipc).transpose()?;
                let mut options = previous
                    .as_ref()
                    .map(CodeIndex::build_options)
                    .transpose()?
                    .unwrap_or_default();
                if let Some(lines) = fallback_lines {
                    options.fallback_lines = lines;
                }
                if let Some(bytes) = max_file_bytes {
                    options.max_file_bytes = bytes;
                }
                let plan = prepare_repository(root, &options)?;
                let mut query_inputs = Vec::new();
                if let Some(path) = queries {
                    use std::io::BufRead;
                    for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
                        let query: benchmark::Query = serde_json::from_str(&line?)?;
                        query_inputs.push(EmbeddingInput::new(query.id, query.text));
                    }
                }
                if let Some(text) = query {
                    query_inputs.push(EmbeddingInput::new(query_id, text));
                }
                write_inputs(
                    &output,
                    &EmbeddingInputs {
                        format_version: 1,
                        documents: plan.pending_inputs(previous.as_ref())?,
                        queries: query_inputs,
                    },
                )?;
            }
        },
        Command::Benchmark { command } => match command {
            BenchmarkCommand::Prepare {
                dataset,
                split,
                output,
            } => {
                write_inputs(&output, &Dataset::load(dataset, &split)?.embedding_inputs())?;
            }
            BenchmarkCommand::Validate { dataset, split } => {
                let dataset = Dataset::load(dataset, &split)?;
                println!("{}", serde_json::to_string_pretty(dataset.summary())?);
            }
            BenchmarkCommand::Run {
                dataset,
                split,
                output,
                embedding,
                model_dir,
                vectors,
                encoder_config,
                dimensions,
                batch_size,
                top_k,
                run_tag,
                engine_revision,
            } => {
                let dataset = Dataset::load(dataset, &split)?;
                let options = RunOptions {
                    top_k,
                    batch_size,
                    run_tag,
                };
                options.validate()?;
                std::fs::create_dir(&output)?;
                let outcome = (|| -> Result<()> {
                    benchmark::write_json(&output.join("dataset.json"), dataset.summary())?;
                    benchmark::write_json(&output.join("qrels.json"), dataset.qrels())?;
                    benchmark::write_json(
                        &output.join("request.json"),
                        &serde_json::json!({
                            "options": &options,
                            "embedding": match embedding { ProviderKind::Hash => "hash", ProviderKind::Minilm => "minilm", ProviderKind::Precomputed => "precomputed", ProviderKind::Process => "process" },
                            "model_dir": &model_dir,
                            "vectors": &vectors,
                            "encoder_config": &encoder_config,
                            "dimensions": dimensions,
                            "split": &split,
                            "engine_revision": engine_revision.as_deref()
                        }),
                    )?;
                    let start = Instant::now();
                    let mut provider = provider(
                        embedding,
                        dimensions,
                        model_dir.as_deref(),
                        vectors.as_deref(),
                        encoder_config.as_deref(),
                    )?;
                    let model_load_ms = elapsed_ms(start);
                    // Capture binary/model provenance before a potentially long encoding run,
                    // and retain it even if document/query inference subsequently fails.
                    let mut executable_hash = blake3::Hasher::new();
                    executable_hash
                        .update_reader(std::fs::File::open(std::env::current_exe()?)?)?;
                    benchmark::write_json(
                        &output.join("preparation.json"),
                        &serde_json::json!({
                            "model_load_ms": model_load_ms,
                            "cubeql_code_version": env!("CARGO_PKG_VERSION"),
                            "embedding": provider.config(),
                            "executable_blake3": executable_hash.finalize().to_hex().to_string(),
                            "engine_revision": engine_revision,
                            "engine_revision_source": "optional-caller-supplied-not-build-attested"
                        }),
                    )?;
                    let run = benchmark::run(
                        &dataset,
                        &cubecl::Device::default(),
                        provider.as_mut(),
                        options,
                    )?;
                    run.save(&output)?;
                    println!(
                        "{}",
                        serde_json::to_string(
                            &serde_json::json!({"status":"complete","output":output,"queries":run.report.queries.len()})
                        )?
                    );
                    Ok(())
                })();
                if let Err(error) = outcome {
                    if let Err(record_error) = benchmark::write_json(
                        &output.join("failure.json"),
                        &serde_json::json!({"status":"failed","error":error.to_string()}),
                    ) {
                        eprintln!("could not save benchmark failure: {record_error}");
                    }
                    return Err(error);
                }
            }
        },
        Command::Index {
            root,
            output,
            previous,
            embedding,
            model_dir,
            vectors,
            encoder_config,
            dimensions,
            batch_size,
            fallback_lines,
            max_file_bytes,
            precision,
            display,
        } => {
            new_output(&output)?;
            let previous = previous.map(CodeIndex::load_ipc).transpose()?;
            let precision = precision.map(VectorPrecision::from).unwrap_or_else(|| {
                previous
                    .as_ref()
                    .map_or(VectorPrecision::Fp32, CodeIndex::precision)
            });
            if previous
                .as_ref()
                .is_some_and(|p| p.precision() != precision)
            {
                return Err("changing precision with --previous is not supported; rebuild without --previous".into());
            }
            let mut provider = provider(
                embedding,
                dimensions,
                model_dir.as_deref(),
                vectors.as_deref(),
                encoder_config.as_deref(),
            )?;
            let options = BuildOptions {
                fallback_lines,
                max_file_bytes,
                ..BuildOptions::for_embedding(provider.config())?
            };
            let mut result = build_repository_with_provider(
                root,
                &options,
                previous.as_ref(),
                provider.as_mut(),
                batch_size,
            )?;
            result.index = result.index.with_precision(precision)?;
            print_build(result, &output, display.format)?;
        }
        Command::Update {
            index,
            output,
            root,
            model_dir,
            vectors,
            encoder_config,
            batch_size,
            display,
        } => {
            new_output(&output)?;
            let previous = CodeIndex::load_ipc(index)?;
            let root = root
                .or_else(|| previous.root().map(PathBuf::from))
                .ok_or("snapshot has no saved root; supply --root")?;
            let mut provider = saved_provider(
                &previous,
                model_dir.as_deref(),
                vectors.as_deref(),
                encoder_config.as_deref(),
            )?;
            let result = build_repository_with_provider(
                root,
                &previous.build_options()?,
                Some(&previous),
                provider.as_mut(),
                batch_size,
            )?;
            print_build(result, &output, display.format)?;
        }
        Command::Inspect { index, display } => {
            let start = Instant::now();
            let code = CodeIndex::load_ipc(&index)?;
            let read_ms = elapsed_ms(start);
            let snapshot = SnapshotInfo::new(&code)?;
            match display.format {
                OutputFormat::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "format_version": 1, "snapshot": snapshot, "input_file_bytes": std::fs::metadata(index)?.len(), "read_ms": read_ms
                    }))?
                ),
                OutputFormat::Text => {
                    println!(
                        "chunks={} files={} dimensions={} precision={:?}",
                        snapshot.chunks, snapshot.files, snapshot.dimensions, snapshot.precision
                    );
                    println!(
                        "root={}\nembedding={}\ncorpus={}",
                        snapshot.root.as_deref().unwrap_or("unknown"),
                        snapshot.embedding_id,
                        snapshot.corpus_fingerprint
                    );
                    println!(
                        "vector_bytes={} arrow_array_bytes={} ipc_bytes={}",
                        snapshot.vector_storage_bytes,
                        snapshot.arrow_array_memory_bytes,
                        snapshot.ipc_bytes
                    );
                }
            }
        }
        Command::Search {
            index,
            text,
            query_id,
            model_dir,
            vectors,
            encoder_config,
            top_k,
            strategy,
            candidate_k,
            no_dedup,
            language,
            path_prefix,
            include_source,
            display,
        } => {
            let strategy = SearchStrategy::from(strategy);
            let options = RetrievalOptions {
                strategy,
                top_k,
                candidate_k: candidate_k.unwrap_or(50.max(top_k)),
                deduplicate: !no_dedup,
            };
            options.validate()?;
            if text.trim().is_empty() {
                return Err("search text must not be empty".into());
            }
            let mut timings = SearchTimings::default();
            let start = Instant::now();
            let code = CodeIndex::load_ipc(index)?;
            timings.read_ms = elapsed_ms(start);
            let precision = code.precision();
            let start = Instant::now();
            let needs_semantic = strategy.needs_embedding() && top_k > 0;
            let mut provider = if needs_semantic {
                Some(saved_provider(
                    &code,
                    model_dir.as_deref(),
                    vectors.as_deref(),
                    encoder_config.as_deref(),
                )?)
            } else {
                None
            };
            timings.model_load_ms = if needs_semantic {
                elapsed_ms(start)
            } else {
                0.0
            };
            let start = Instant::now();
            let query = provider
                .as_mut()
                .map(|p| match &query_id {
                    Some(id) => embed_identified_query(p.as_mut(), &EmbeddingInput::new(id, &text)),
                    None => embed_query(p.as_mut(), &text),
                })
                .transpose()?;
            timings.embedding_ms = if needs_semantic {
                elapsed_ms(start)
            } else {
                0.0
            };
            let device = needs_semantic.then(cubecl::Device::default);
            let start = Instant::now();
            let (search, host) = if let Some(device) = &device {
                (Some(CodeSearchIndex::new(device, code)?), None)
            } else {
                (None, Some(code))
            };
            timings.upload_ms = if needs_semantic {
                elapsed_ms(start)
            } else {
                0.0
            };
            let code = search
                .as_ref()
                .map(CodeSearchIndex::code)
                .or(host.as_ref())
                .unwrap();
            let start = Instant::now();
            let text_index = SnapshotTextIndex::new(code);
            text_index.warm_up(strategy);
            timings.text_index_ms = elapsed_ms(start);
            let filter = CodeFilter {
                language,
                path_prefix,
            };
            let start = Instant::now();
            let selected = text_index.prepare(&filter, search.as_ref())?;
            timings.filter_ms = elapsed_ms(start);
            let start = Instant::now();
            let result = selected.query(&text, query.as_ref(), options)?;
            timings.retrieval_ms = elapsed_ms(start);
            timings.scoring_topk_readback_ms = result.timings.semantic_ms;
            timings.lexical_ms = result.timings.lexical_ms;
            timings.symbol_ms = result.timings.symbol_ms;
            timings.fusion_ms = result.timings.fusion_ms;
            timings.dedup_ms = result.timings.dedup_ms;
            let start = Instant::now();
            let hits = materialize_hits(code, &result.hits, include_source)?;
            timings.materialization_ms = elapsed_ms(start);
            let response = RetrievalResponse {
                format_version: 2,
                query: text,
                strategy,
                metric: strategy.score_kind(),
                backend: device.map_or_else(|| "Host".into(), |d| format!("{d:?}")),
                precision,
                hits,
                timings,
            };
            match display.format {
                OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&response)?),
                OutputFormat::Text => {
                    for hit in response.hits {
                        let sources = hit
                            .provenance
                            .iter()
                            .map(|p| format!("{:?}@{}", p.source, p.rank))
                            .collect::<Vec<_>>()
                            .join(",");
                        let hit = hit.code;
                        println!(
                            "{}:{}-{} {} score={:.6} sources={}",
                            hit.path, hit.start_line, hit.end_line, hit.symbol, hit.score, sources
                        );
                        if let Some(source) = hit.source {
                            println!("{source}");
                        }
                    }
                }
            }
        }
        Command::Evaluate {
            index,
            queries,
            model_dir,
            vectors,
            encoder_config,
            candidate_k,
            repeats,
            compare_fp16,
            export_candidates,
            display,
        } => {
            if export_candidates && !matches!(display.format, OutputFormat::Json) {
                return Err("--export-candidates requires --format json".into());
            }
            let start = Instant::now();
            let code = CodeIndex::load_ipc(&index)?;
            let read_ms = elapsed_ms(start);
            let dataset: QuerySet = serde_json::from_reader(std::fs::File::open(queries)?)?;
            dataset.resolve(&code)?;
            let start = Instant::now();
            let mut provider = saved_provider(
                &code,
                model_dir.as_deref(),
                vectors.as_deref(),
                encoder_config.as_deref(),
            )?;
            let model_load_ms = elapsed_ms(start);
            let options = EvaluationOptions {
                candidate_k,
                repeats,
                compare_fp16,
            };
            let report = if export_candidates {
                evaluate_with_candidates(
                    code,
                    &cubecl::Device::default(),
                    provider.as_mut(),
                    &dataset,
                    options,
                )?
            } else {
                evaluate(
                    code,
                    &cubecl::Device::default(),
                    provider.as_mut(),
                    &dataset,
                    options,
                )?
            };
            match display.format {
                OutputFormat::Json => println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "format_version": 1, "read_ms": read_ms, "model_load_ms": model_load_ms,
                        "input_file_bytes": std::fs::metadata(index)?.len(), "evaluation": report
                    }))?
                ),
                OutputFormat::Text => {
                    println!(
                        "dataset={} backend={} queries={} read_ms={:.3} model_load_ms={:.3} embedding_ms={:.3}",
                        report.dataset.name,
                        report.backend,
                        report.dataset.queries.len(),
                        read_ms,
                        model_load_ms,
                        report.embedding_total_ms
                    );
                    for variant in report.variants {
                        println!(
                            "precision={:?} chunks={} vector_bytes={} ipc_bytes={}",
                            variant.snapshot.precision,
                            variant.snapshot.chunks,
                            variant.snapshot.vector_storage_bytes,
                            variant.snapshot.ipc_bytes
                        );
                        for (name, summary) in variant.overall {
                            println!(
                                "{name}: recall@5={:.4} recall@10={:.4} mrr@{}={:.4}",
                                summary.recall_at_5, summary.recall_at_10, candidate_k, summary.mrr
                            );
                        }
                    }
                    println!(
                        "Use --format json for per-query/category rankings, timings and memory estimates."
                    );
                }
            }
        }
        Command::Model {
            command: ModelCommand::Download { directory },
        } => {
            #[cfg(feature = "local-embeddings")]
            {
                eprintln!(
                    "Downloading/verifying pinned MiniLM artifacts in {}",
                    directory.display()
                );
                cubeql_embeddings::download_minilm(&directory)?;
                println!(
                    "model={} revision={} directory={}",
                    cubeql_embeddings::MINILM_MODEL,
                    cubeql_embeddings::MINILM_REVISION,
                    directory.display()
                );
            }
            #[cfg(not(feature = "local-embeddings"))]
            {
                let _ = directory;
                return Err("rebuild with --features local-embeddings to download MiniLM".into());
            }
        }
    }
    Ok(())
}
