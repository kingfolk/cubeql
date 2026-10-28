use std::path::PathBuf;

use clap::Parser;
use cubeql_mcp::{CodeServer, ServerOptions};
use rmcp::{ServiceExt, transport::stdio};

#[derive(Parser)]
#[command(version, about = "Read-only CubeQL code retrieval over MCP stdio")]
struct Cli {
    /// Immutable Arrow code snapshot. Restart to load a new snapshot.
    index: PathBuf,
    /// Existing pinned MiniLM directory; no automatic downloads.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Load neither a model nor a device; only lexical/symbol strategies are supported.
    #[arg(long)]
    text_only: bool,
    /// Explicit directory boundary for check_code. Never inferred from snapshot metadata.
    #[arg(long)]
    workspace_root: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    let server = CodeServer::start(ServerOptions {
        index: args.index,
        model_dir: args.model_dir,
        text_only: args.text_only,
        workspace_root: args.workspace_root,
    })
    .await?;
    // stdout belongs exclusively to MCP JSON-RPC. Diagnostics and traces go to stderr.
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
