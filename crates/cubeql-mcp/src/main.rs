use std::path::PathBuf;

use clap::Parser;
use cubeql_mcp::contract::Strategy;
use cubeql_mcp::{CodeServer, ServerOptions};
use rmcp::{ServiceExt, transport::stdio};

#[derive(Parser)]
#[command(version, about = "Snapshot-local CubeQL code retrieval over MCP stdio")]
struct Cli {
    /// Initial immutable Arrow code snapshot.
    index: PathBuf,
    /// Enable explicit reload_snapshot from this fixed operator-selected path (512 MiB input limit).
    #[arg(long)]
    reload_index: Option<PathBuf>,
    /// Existing pinned MiniLM directory; no automatic downloads.
    #[arg(long, conflicts_with_all = ["encoder_config", "text_only"])]
    model_dir: Option<PathBuf>,
    /// Explicit resident local encoder launch profile; no automatic model downloads.
    #[arg(long, conflicts_with_all = ["model_dir", "text_only"])]
    encoder_config: Option<PathBuf>,
    /// Load neither a model nor a device; only lexical/symbol strategies are supported.
    #[arg(long)]
    text_only: bool,
    /// Explicit directory boundary for check_code. Never inferred from snapshot metadata.
    #[arg(long)]
    workspace_root: Option<PathBuf>,
    /// Restrict search_code to exactly this strategy (useful for controlled evaluations).
    #[arg(long, value_enum)]
    search_strategy: Option<Strategy>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    let server = CodeServer::start(ServerOptions {
        index: args.index,
        reload_index: args.reload_index,
        model_dir: args.model_dir,
        encoder_config: args.encoder_config,
        text_only: args.text_only,
        workspace_root: args.workspace_root,
        search_strategy: args.search_strategy,
    })
    .await?;
    // stdout belongs exclusively to MCP JSON-RPC. Diagnostics and traces go to stderr.
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
