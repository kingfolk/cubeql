use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use cubeql::Metric;
use cubeql_code::{BuildOptions, CodeIndex, CodeSearchIndex, build_repository_with_provider};
use cubeql_embeddings::{EmbeddingProvider, HashProvider};

#[derive(Parser)]
#[command(
    version,
    about = "Build and query code indexes with pluggable embedding providers"
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

#[derive(Subcommand)]
enum Command {
    /// Index source files; optionally reuse unchanged embeddings from a previous snapshot.
    Index {
        root: PathBuf,
        /// A new output file. Existing files are never overwritten.
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        previous: Option<PathBuf>,
        /// Hash is an offline lexical fixture, not semantic search.
        #[arg(long, value_enum, default_value = "hash")]
        embedding: ProviderKind,
        /// Local pinned MiniLM artifacts, downloaded separately with `model download`.
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
    },
    /// Embed text with the index's provider and return cosine top-k code locations.
    Query {
        index: PathBuf,
        text: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Manage optional local reference model artifacts.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download the pinned MiniLM files (~91 MB); never replace existing conflicting files.
    Download {
        #[arg(long)]
        directory: PathBuf,
    },
}

fn provider(
    kind: ProviderKind,
    dimensions: Option<usize>,
    directory: Option<&Path>,
) -> Result<Box<dyn EmbeddingProvider>, Box<dyn std::error::Error>> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        } => {
            if output.exists() {
                return Err("output already exists; choose a new snapshot path".into());
            }
            if batch_size == 0 {
                return Err("--batch-size must be positive".into());
            }
            let mut provider = provider(embedding, dimensions, model_dir.as_deref())?;
            let options = BuildOptions {
                fallback_lines,
                max_file_bytes,
                ..BuildOptions::for_embedding(provider.config())?
            };
            let previous = previous.map(CodeIndex::load_ipc).transpose()?;
            let result = build_repository_with_provider(
                root,
                &options,
                previous.as_ref(),
                provider.as_mut(),
                batch_size,
            )?;
            result.index.save_ipc(&output)?;
            let r = result.report;
            println!(
                "files={} chunks={} embedded={} reused={} added={} changed={} removed={}",
                r.files, r.chunks, r.embedded, r.reused, r.added, r.changed, r.removed
            );
            println!(
                "embedding={} dimensions={} output={}",
                provider.config().model,
                result.index.dimensions(),
                output.display()
            );
            for warning in r.warnings {
                eprintln!("warning: {warning}");
            }
        }
        Command::Query {
            index,
            text,
            model_dir,
            top_k,
        } => {
            let code = CodeIndex::load_ipc(index)?;
            let config = code
                .embedding_config()
                .ok_or("legacy index has no provider contract; rebuild with index")?;
            let kind = match config.model.as_str() {
                "cubeql-hash-tokens" => ProviderKind::Hash,
                "sentence-transformers/all-MiniLM-L6-v2" => ProviderKind::Minilm,
                _ => return Err(
                    "this CLI does not implement the index's embedding provider; use the library"
                        .into(),
                ),
            };
            let mut provider = provider(kind, Some(config.dimensions), model_dir.as_deref())?;
            config.ensure_compatible(provider.config())?;
            let search = CodeSearchIndex::new(&cubecl::Device::default(), code)?;
            for hit in search.query_text(provider.as_mut(), &text, top_k, Metric::Cosine)? {
                let chunk = &search.code().chunks()[hit.row];
                println!(
                    "{}:{}-{} {} score={:.6}",
                    chunk.path, chunk.start_line, chunk.end_line, chunk.symbol, hit.score
                );
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
