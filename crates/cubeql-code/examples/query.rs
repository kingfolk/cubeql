//! A lexical-fixture query example, not the future semantic-search CLI.
use cubeql::{IndexQueryOptions, Metric, VectorIndex};
use cubeql_code::{CodeIndex, HASH_EMBEDDING_ID, hash_embedding};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: query <index.arrow> <query text>")?;
    let text = args
        .next()
        .ok_or("usage: query <index.arrow> <query text>")?;
    if args.next().is_some() {
        return Err("quote query text as a single argument".into());
    }
    let code = CodeIndex::load_ipc(path)?;
    if code.embedding_id() != HASH_EMBEDDING_ID {
        return Err("this example only queries the built-in token-hash fixture embeddings".into());
    }
    eprintln!(
        "Lexical fixture only: hash collisions can produce irrelevant results; this is not semantic search."
    );
    let query = hash_embedding(&text, code.dimensions())?;
    let index = VectorIndex::new(
        &cubecl::Device::default(),
        code.batch().clone(),
        "embedding",
    )?;
    for hit in index.query_with_metric(&query, IndexQueryOptions::new(5), Metric::Cosine)? {
        let chunk = &code.chunks()[hit.row];
        println!(
            "{}:{}-{} {} score={:.6}",
            chunk.path, chunk.start_line, chunk.end_line, chunk.symbol, hit.score
        );
    }
    Ok(())
}
