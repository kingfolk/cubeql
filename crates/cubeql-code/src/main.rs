use std::path::PathBuf;

use clap::{Parser, Subcommand};
use cubeql_code::{BuildOptions, CodeIndex, build_repository, hash_embedding};

#[derive(Parser)]
#[command(
    version,
    about = "Build Arrow code indexes (lexical fixture embeddings, not semantic search)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
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
        #[arg(long, default_value_t = 128)]
        dimensions: usize,
        #[arg(long, default_value_t = 80)]
        fallback_lines: usize,
        #[arg(long, default_value_t = 2 * 1024 * 1024)]
        max_file_bytes: u64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Cli { command } = Cli::parse();
    match command {
        Command::Index {
            root,
            output,
            previous,
            dimensions,
            fallback_lines,
            max_file_bytes,
        } => {
            // Catch the common mistake before scanning or performing embedding work. The actual
            // save still uses create_new to protect against races between this check and writing.
            if output.exists() {
                return Err("output already exists; choose a new snapshot path".into());
            }
            let options = BuildOptions {
                dimensions,
                fallback_lines,
                max_file_bytes,
                ..Default::default()
            };
            let previous = previous.map(CodeIndex::load_ipc).transpose()?;
            let result = build_repository(root, &options, previous.as_ref(), |text| {
                hash_embedding(text, dimensions)
            })?;
            result.index.save_ipc(&output)?;
            let r = result.report;
            println!(
                "files={} chunks={} embedded={} reused={} added={} changed={} removed={}",
                r.files, r.chunks, r.embedded, r.reused, r.added, r.changed, r.removed
            );
            println!(
                "embedding={} dimensions={} output={}",
                result.index.embedding_id(),
                dimensions,
                output.display()
            );
            for warning in r.warnings {
                eprintln!("warning: {warning}");
            }
        }
    }
    Ok(())
}
