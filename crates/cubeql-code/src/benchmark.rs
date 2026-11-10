//! Standard external-document benchmark adapter. IDs remain opaque; no source parsing,
//! rechunking, symbol synthesis, relevance filtering or local quality metric is applied.
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use cubecl::Device;
use cubeql::{IndexOptions, IndexQueryOptions, Metric, VectorIndex, write_ipc_batch};
use cubeql_embeddings::{
    EmbeddingConfig, EmbeddingInput, EmbeddingInputs, EmbeddingProvider, embed_identified_query,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{CodeError, PersistenceError, invalid, reporting::elapsed_ms};

pub const DOCUMENT_TEXT_POLICY: &str = "title-newline-text-v1";

#[derive(Clone, Debug, Deserialize)]
pub struct Document {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub text: String,
}

impl Document {
    /// Only content is embedded, never IDs or relevance judgments. Empty/missing titles
    /// add no separator; otherwise the exact title and text are joined by one newline.
    pub fn embedding_text(&self) -> String {
        match self.title.as_deref().filter(|s| !s.is_empty()) {
            Some(title) => format!("{title}\n{}", self.text),
            None => self.text.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Query {
    #[serde(rename = "_id")]
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DatasetSummary {
    pub format: &'static str,
    pub documents: usize,
    pub queries: usize,
    pub judgments: usize,
    pub judged_queries: usize,
    /// No qrels is not a declaration that the repository contains no answer.
    pub unjudged_queries: usize,
    pub grade_counts: BTreeMap<i32, usize>,
    /// Hashes cover the exact bytes parsed, including original line endings.
    pub input_blake3: BTreeMap<String, String>,
}

pub struct Dataset {
    documents: Vec<Document>,
    queries: Vec<Query>,
    qrels: BTreeMap<String, BTreeMap<String, i32>>,
    summary: DatasetSummary,
}

impl Dataset {
    /// Load canonical BEIR-style JSONL and a three-column, headered qrels TSV.
    /// `split` is one filename component, not an arbitrary path.
    pub fn load(root: impl AsRef<Path>, split: &str) -> Result<Self, CodeError> {
        if split.is_empty()
            || !split
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(invalid(
                "benchmark split must contain only ASCII letters, digits, _ or -",
            ));
        }
        let root = root.as_ref();
        let (documents, corpus_hash) = jsonl::<Document>(&root.join("corpus.jsonl"))?;
        let (queries, queries_hash) = jsonl::<Query>(&root.join("queries.jsonl"))?;
        if documents.is_empty() || queries.is_empty() {
            return Err(invalid("benchmark corpus and queries must be non-empty"));
        }
        let doc_ids = unique_ids(documents.iter().map(|d| d.id.as_str()), "document")?;
        let query_ids = unique_ids(queries.iter().map(|q| q.id.as_str()), "query")?;
        if queries.iter().any(|q| q.text.trim().is_empty()) {
            return Err(invalid("benchmark query text must not be empty"));
        }
        let qrels_name = format!("qrels/{split}.tsv");
        let mut qrels = BTreeMap::<String, BTreeMap<String, i32>>::new();
        let mut grades = BTreeMap::new();
        let mut qrels_lines = 0;
        let qrels_hash = lines(&root.join(&qrels_name), |line, bytes| {
            qrels_lines = line;
            let value = std::str::from_utf8(bytes)
                .map_err(|_| invalid("qrels must be UTF-8"))?
                .trim_end_matches(['\r', '\n']);
            if line == 1 {
                if value != "query-id\tcorpus-id\tscore" {
                    return Err(invalid(
                        "expected qrels header: query-id<TAB>corpus-id<TAB>score",
                    ));
                }
                return Ok(());
            }
            let fields = value.split('\t').collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err(invalid("expected exactly three qrels TSV columns"));
            }
            let (query, document) = (fields[0], fields[1]);
            if !query_ids.contains(query) || !doc_ids.contains(document) {
                return Err(invalid(format!(
                    "qrels references unknown query/document ID: {query} / {document}"
                )));
            }
            let grade = fields[2]
                .parse::<i32>()
                .map_err(|_| invalid("qrels grade must be an integer"))?;
            if qrels
                .entry(query.into())
                .or_default()
                .insert(document.into(), grade)
                .is_some()
            {
                return Err(invalid(format!(
                    "duplicate qrels pair: {query} / {document}"
                )));
            }
            *grades.entry(grade).or_insert(0) += 1;
            Ok(())
        })?;
        if qrels_lines == 0 {
            return Err(invalid(
                "qrels TSV must have a header (zero judgments is allowed)",
            ));
        }
        let summary = DatasetSummary {
            format: "beir-jsonl-tsv-v1",
            documents: documents.len(),
            queries: queries.len(),
            judgments: qrels.values().map(BTreeMap::len).sum(),
            judged_queries: qrels.len(),
            unjudged_queries: queries.len() - qrels.len(),
            grade_counts: grades,
            input_blake3: BTreeMap::from([
                ("corpus.jsonl".into(), corpus_hash),
                ("queries.jsonl".into(), queries_hash),
                (qrels_name, qrels_hash),
            ]),
        };
        Ok(Self {
            documents,
            queries,
            qrels,
            summary,
        })
    }

    pub fn embedding_inputs(&self) -> EmbeddingInputs {
        EmbeddingInputs {
            format_version: 1,
            documents: self
                .documents
                .iter()
                .map(|d| EmbeddingInput::new(&d.id, d.embedding_text()))
                .collect(),
            queries: self
                .queries
                .iter()
                .map(|q| EmbeddingInput::new(&q.id, &q.text))
                .collect(),
        }
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }
    pub fn queries(&self) -> &[Query] {
        &self.queries
    }
    pub fn qrels(&self) -> &BTreeMap<String, BTreeMap<String, i32>> {
        &self.qrels
    }
    pub fn summary(&self) -> &DatasetSummary {
        &self.summary
    }
}

fn token(value: &str) -> bool {
    !value.is_empty() && !value.chars().any(|c| c.is_whitespace() || c.is_control())
}

fn unique_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    kind: &str,
) -> Result<HashSet<&'a str>, CodeError> {
    let mut seen = HashSet::new();
    for id in ids {
        if !token(id) || !seen.insert(id) {
            return Err(invalid(format!(
                "invalid or duplicate {kind} ID: {id:?}; IDs must be nonempty, whitespace/control-free strings"
            )));
        }
    }
    Ok(seen)
}

fn lines(
    path: &Path,
    mut visit: impl FnMut(usize, &[u8]) -> Result<(), CodeError>,
) -> Result<String, CodeError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = Vec::new();
    let mut number = 0;
    loop {
        buffer.clear();
        if reader.read_until(b'\n', &mut buffer)? == 0 {
            break;
        }
        number += 1;
        hasher.update(&buffer);
        visit(number, &buffer).map_err(|e| invalid(format!("{}:{number}: {e}", path.display())))?;
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn jsonl<T: DeserializeOwned>(path: &Path) -> Result<(Vec<T>, String), CodeError> {
    let mut values = Vec::new();
    let hash = lines(path, |_, bytes| {
        values.push(serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?);
        Ok(())
    })?;
    Ok((values, hash))
}

#[derive(Clone, Debug, Serialize)]
pub struct RunOptions {
    pub top_k: usize,
    pub batch_size: usize,
    pub run_tag: String,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            top_k: 100,
            batch_size: 8,
            run_tag: "cubeql-semantic".into(),
        }
    }
}

impl RunOptions {
    pub fn validate(&self) -> Result<(), CodeError> {
        if self.top_k == 0 || self.batch_size == 0 || !token(&self.run_tag) {
            return Err(invalid(
                "benchmark top_k and batch_size must be positive; run_tag must be one nonempty token",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct RankedDocument {
    pub document_id: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct QueryRanking {
    pub query_id: String,
    /// Engine order: descending cosine; equal scores use ascending corpus row ID.
    pub hits: Vec<RankedDocument>,
    pub embedding_ms: f64,
    pub retrieval_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub format_version: u32,
    pub status: &'static str,
    pub dataset: DatasetSummary,
    pub embedding: EmbeddingConfig,
    pub embedding_is_hash_fixture: bool,
    pub embedding_execution: String,
    pub backend: String,
    pub precision: &'static str,
    pub metric: &'static str,
    pub document_text_policy: &'static str,
    pub query_selection: &'static str,
    pub relevance_scoring: &'static str,
    pub options: RunOptions,
    pub document_embedding_ms: f64,
    pub index_setup_upload_ms: f64,
    pub queries: Vec<QueryRanking>,
}

pub struct BenchmarkRun {
    batch: RecordBatch,
    pub report: RunReport,
}

impl BenchmarkRun {
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Write into a caller-created run directory. Every file uses create_new, including
    /// the final completion report; existing evidence is never replaced on retry.
    pub fn save(&self, directory: impl AsRef<Path>) -> Result<(), CodeError> {
        let directory = directory.as_ref();
        let mut writer = BufWriter::new(new_file(&directory.join("results.json"))?);
        let results = self
            .report
            .queries
            .iter()
            .map(|q| {
                (
                    q.query_id.as_str(),
                    q.hits
                        .iter()
                        .map(|h| (h.document_id.as_str(), h.score))
                        .collect::<BTreeMap<_, _>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        serde_json::to_writer(&mut writer, &results).map_err(|e| invalid(e.to_string()))?;
        writer.flush()?;
        let mut writer = BufWriter::new(new_file(&directory.join("run.trec"))?);
        for query in &self.report.queries {
            for (i, hit) in query.hits.iter().enumerate() {
                writeln!(
                    writer,
                    "{} Q0 {} {} {} {}",
                    query.query_id,
                    hit.document_id,
                    i + 1,
                    hit.score,
                    self.report.options.run_tag
                )?;
            }
        }
        writer.flush()?;
        let mut writer = BufWriter::new(new_file(&directory.join("index.arrow"))?);
        write_ipc_batch(
            &self.batch,
            "embedding",
            IndexOptions::default(),
            &mut writer,
        )?;
        writer.flush()?;
        write_json(&directory.join("report.json"), &self.report)?;
        Ok(())
    }
}

pub fn new_file(path: &Path) -> Result<File, CodeError> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<(), CodeError> {
    let mut writer = BufWriter::new(new_file(path)?);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|e| invalid(e.to_string()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Execute every input query, regardless of whether/how it is judged. Qrels are only
/// validation/provenance data and never select documents, queries, vectors or candidates.
pub fn run(
    dataset: &Dataset,
    device: &Device,
    provider: &mut dyn EmbeddingProvider,
    options: RunOptions,
) -> Result<BenchmarkRun, CodeError> {
    options.validate()?;
    let config = provider.config().clone();
    config.validate()?;
    let expected = dataset.embedding_inputs();
    provider.validate_document_inputs(&expected.documents)?;
    provider.validate_query_inputs(&expected.queries)?;
    let mut values = Vec::new();
    let start = Instant::now();
    for documents in dataset.documents.chunks(options.batch_size) {
        let inputs = documents
            .iter()
            .map(|d| EmbeddingInput::new(&d.id, d.embedding_text()))
            .collect::<Vec<_>>();
        let vectors = provider.embed_documents_with_ids(&inputs)?;
        config.ensure_compatible(provider.config())?;
        config.validate_vectors(&vectors, inputs.len())?;
        values.extend(vectors.into_iter().flatten());
    }
    let document_embedding_ms = elapsed_ms(start);
    let vectors: ArrayRef = Arc::new(FixedSizeListArray::try_new_with_length(
        Arc::new(Field::new("item", DataType::Float32, false)),
        config.dimensions as i32,
        Arc::new(Float32Array::from(values)),
        None,
        dataset.documents.len(),
    )?);
    let strings = |get: fn(&Document) -> &str| -> ArrayRef {
        Arc::new(StringArray::from(
            dataset.documents.iter().map(get).collect::<Vec<_>>(),
        ))
    };
    let batch = RecordBatch::try_from_iter(vec![
        ("document_id", strings(|d| &d.id)),
        ("title", strings(|d| d.title.as_deref().unwrap_or(""))),
        ("text", strings(|d| &d.text)),
        ("embedding", vectors),
    ])?;
    // The persisted index contains only corpus provenance, never queries or judgments.
    let metadata = HashMap::from([
        ("cubeql.benchmark.version".into(), "1".into()),
        (
            "cubeql.benchmark.corpus_blake3".into(),
            dataset.summary.input_blake3["corpus.jsonl"].clone(),
        ),
        (
            "cubeql.benchmark.document_text_policy".into(),
            DOCUMENT_TEXT_POLICY.into(),
        ),
        (
            "cubeql.benchmark.embedding_config".into(),
            serde_json::to_string(&config).map_err(|e| invalid(e.to_string()))?,
        ),
    ]);
    let schema = Arc::new(Schema::new_with_metadata(
        batch.schema().fields().clone(),
        metadata,
    ));
    let batch = batch.with_schema(schema)?;
    let start = Instant::now();
    let index =
        VectorIndex::new(device, batch.clone(), "embedding").map_err(PersistenceError::from)?;
    let index_setup_upload_ms = elapsed_ms(start);
    let mut rankings = Vec::with_capacity(dataset.queries.len());
    for query in &dataset.queries {
        let start = Instant::now();
        let vector =
            embed_identified_query(provider, &EmbeddingInput::new(&query.id, &query.text))?;
        config.ensure_compatible(vector.config())?;
        let embedding_ms = elapsed_ms(start);
        let start = Instant::now();
        let hits = index
            .query_with_metric(
                vector.vector(),
                IndexQueryOptions::new(options.top_k),
                Metric::Cosine,
            )
            .map_err(PersistenceError::from)?;
        let retrieval_ms = elapsed_ms(start);
        let hits = hits
            .into_iter()
            .map(|hit| {
                if !hit.score.is_finite() {
                    return Err(invalid("non-finite benchmark score cannot be exported"));
                }
                Ok(RankedDocument {
                    document_id: dataset.documents[hit.row].id.clone(),
                    score: hit.score,
                })
            })
            .collect::<Result<Vec<_>, CodeError>>()?;
        rankings.push(QueryRanking {
            query_id: query.id.clone(),
            hits,
            embedding_ms,
            retrieval_ms,
        });
    }
    Ok(BenchmarkRun {
        batch,
        report: RunReport {
            format_version: 1,
            status: "complete",
            dataset: dataset.summary.clone(),
            embedding_is_hash_fixture: config.model == "cubeql-hash-tokens",
            embedding_execution: provider.execution_kind().into(),
            embedding: config,
            backend: format!("{device:?}"),
            precision: "fp32",
            metric: "cosine",
            document_text_policy: DOCUMENT_TEXT_POLICY,
            query_selection: "all-input-queries",
            relevance_scoring: "external-evaluator",
            options,
            document_embedding_ms,
            index_setup_upload_ms,
            queries: rankings,
        },
    })
}
