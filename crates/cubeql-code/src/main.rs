use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use cubeql::Metric;
use cubeql_code::{
    BuildOptions, BuildResult, CodeIndex, CodeSearchIndex, VectorPrecision,
    build_repository_with_provider,
    evaluation::{EvaluationOptions, QuerySet, evaluate},
    reporting::{CodeFilter, SearchResponse, SearchTimings, SnapshotInfo, elapsed_ms, materialize},
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
    /// Embed text, apply metadata filters, return cosine top-k source locations.
    #[command(visible_alias = "query")]
    Search {
        index: PathBuf,
        text: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
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
            language,
            path_prefix,
            include_source,
            display,
        } => {
            let mut timings = SearchTimings::default();
            let start = Instant::now();
            let code = CodeIndex::load_ipc(index)?;
            timings.read_ms = elapsed_ms(start);
            let precision = code.precision();
            let start = Instant::now();
            let mut provider = saved_provider(&code, model_dir.as_deref())?;
            timings.model_load_ms = elapsed_ms(start);
            let start = Instant::now();
            let query = embed_query(provider.as_mut(), &text)?;
            timings.embedding_ms = elapsed_ms(start);
            let device = cubecl::Device::default();
            let start = Instant::now();
            let search = CodeSearchIndex::new(&device, code)?;
            timings.upload_ms = elapsed_ms(start);
            let filter = CodeFilter {
                language,
                path_prefix,
            };
            let predicates = filter.predicates();
            let start = Instant::now();
            let selected = search.prepare_filter(&predicates)?;
            timings.filter_ms = elapsed_ms(start);
            let start = Instant::now();
            let hits = selected.query(&query, top_k, Metric::Cosine)?;
            timings.scoring_topk_readback_ms = elapsed_ms(start);
            let start = Instant::now();
            let hits = materialize(search.code(), &hits, include_source)?;
            timings.materialization_ms = elapsed_ms(start);
            let response = SearchResponse {
                format_version: 1,
                query: text,
                metric: "cosine",
                backend: format!("{device:?}"),
                precision,
                hits,
                timings,
            };
            match display.format {
                OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&response)?),
                OutputFormat::Text => {
                    for hit in response.hits {
                        println!(
                            "{}:{}-{} {} score={:.6}",
                            hit.path, hit.start_line, hit.end_line, hit.symbol, hit.score
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
