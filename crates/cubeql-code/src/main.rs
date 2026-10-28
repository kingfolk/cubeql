use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use cubeql_code::{
    BuildOptions, BuildResult, CodeIndex, CodeSearchIndex, VectorPrecision,
    build_repository_with_provider,
    evaluation::{EvaluationOptions, QuerySet},
    hybrid_evaluation::evaluate,
    reporting::{CodeFilter, RetrievalResponse, SearchTimings, SnapshotInfo, elapsed_ms},
    retrieval::{RetrievalOptions, SearchStrategy, SnapshotTextIndex, materialize_hits},
};
use cubeql_embeddings::{EmbeddingProvider, HashProvider, embed_query};

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
        #[arg(long)]
        model_dir: Option<PathBuf>,
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
        #[arg(long)]
        model_dir: Option<PathBuf>,
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
        #[arg(long)]
        model_dir: Option<PathBuf>,
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
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 50)]
        candidate_k: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Derive FP16 storage from this original FP32 snapshot; encode queries only once.
        #[arg(long)]
        compare_fp16: bool,
        #[command(flatten)]
        display: Output,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
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
) -> Result<Box<dyn EmbeddingProvider>> {
    match kind {
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
) -> Result<Box<dyn EmbeddingProvider>> {
    let config = code
        .embedding_config()
        .ok_or("legacy index has no provider contract; rebuild with index")?;
    let kind = match config.model.as_str() {
        "cubeql-hash-tokens" => ProviderKind::Hash,
        "sentence-transformers/all-MiniLM-L6-v2" => ProviderKind::Minilm,
        _ => {
            return Err(
                "this CLI does not implement the index's embedding provider; use the library"
                    .into(),
            );
        }
    };
    let provider = provider(kind, Some(config.dimensions), directory)?;
    config.ensure_compatible(provider.config())?;
    Ok(provider)
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
        Command::Index {
            root,
            output,
            previous,
            embedding,
            model_dir,
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
            let mut provider = provider(embedding, dimensions, model_dir.as_deref())?;
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
            batch_size,
            display,
        } => {
            new_output(&output)?;
            let previous = CodeIndex::load_ipc(index)?;
            let root = root
                .or_else(|| previous.root().map(PathBuf::from))
                .ok_or("snapshot has no saved root; supply --root")?;
            let mut provider = saved_provider(&previous, model_dir.as_deref())?;
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
            model_dir,
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
                Some(saved_provider(&code, model_dir.as_deref())?)
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
                .map(|p| embed_query(p.as_mut(), &text))
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
            candidate_k,
            repeats,
            compare_fp16,
            display,
        } => {
            let start = Instant::now();
            let code = CodeIndex::load_ipc(&index)?;
            let read_ms = elapsed_ms(start);
            let dataset: QuerySet = serde_json::from_reader(std::fs::File::open(queries)?)?;
            dataset.resolve(&code)?;
            let start = Instant::now();
            let mut provider = saved_provider(&code, model_dir.as_deref())?;
            let model_load_ms = elapsed_ms(start);
            let report = evaluate(
                code,
                &cubecl::Device::default(),
                provider.as_mut(),
                &dataset,
                EvaluationOptions {
                    candidate_k,
                    repeats,
                    compare_fp16,
                },
            )?;
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
