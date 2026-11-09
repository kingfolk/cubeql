//! Validated, offline import of precomputed embedding vectors.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{EmbeddedQuery, EmbeddingConfig, EmbeddingError, EmbeddingProvider};

/// One explicitly identified text included in a vector bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingInput {
    pub id: String,
    pub text: String,
    pub content_hash: String,
}

impl EmbeddingInput {
    /// Build an input and bind it to the BLAKE3 hash of its UTF-8 text bytes.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let content_hash = blake3::hash(text.as_bytes()).to_hex().to_string();
        Self {
            id: id.into(),
            text,
            content_hash,
        }
    }

    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.id.is_empty()
            || self
                .id
                .chars()
                .any(|character| character.is_whitespace() || character.is_control())
        {
            return Err(invalid(
                "input ID must be nonempty and contain no whitespace or control characters",
            ));
        }
        if !valid_sha256(&self.content_hash)
            || blake3::hash(self.text.as_bytes()).to_hex().as_str() != self.content_hash
        {
            return Err(invalid(format!(
                "content hash does not match input {:?}",
                self.id
            )));
        }
        Ok(())
    }
}

/// Input rows paired by position with the matrices in a vector bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingInputs {
    pub format_version: u32,
    pub documents: Vec<EmbeddingInput>,
    pub queries: Vec<EmbeddingInput>,
}

impl EmbeddingInputs {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let bytes = fs::read(path)?;
        let inputs: Self = serde_json::from_slice(&bytes)?;
        inputs.validate()?;
        Ok(inputs)
    }

    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.format_version != 1 {
            return Err(invalid(format!(
                "unsupported embedding inputs format version {}",
                self.format_version
            )));
        }
        if self.documents.is_empty() && self.queries.is_empty() {
            return Err(invalid("embedding inputs must contain a document or query"));
        }
        validate_input_role("document", &self.documents, false)?;
        validate_input_role("query", &self.queries, true)?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    format_version: u32,
    dtype: String,
    config: EmbeddingConfig,
    inputs: ManifestFile,
    documents: ManifestFile,
    queries: ManifestFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    file: String,
    sha256: String,
}

/// An in-memory, validated set of vectors and the inputs that identify their rows.
pub struct VectorBundle {
    config: EmbeddingConfig,
    inputs: EmbeddingInputs,
    documents: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    document_rows: HashMap<String, usize>,
    query_rows: HashMap<String, usize>,
}

/// An embedding provider backed entirely by a previously validated vector bundle.
pub struct BundleProvider {
    bundle: VectorBundle,
}

impl BundleProvider {
    pub fn new(bundle: VectorBundle) -> Self {
        Self { bundle }
    }

    pub fn bundle(&self) -> &VectorBundle {
        &self.bundle
    }

    pub fn into_bundle(self) -> VectorBundle {
        self.bundle
    }
}

impl EmbeddingProvider for BundleProvider {
    fn config(&self) -> &EmbeddingConfig {
        self.bundle.config()
    }

    fn execution_kind(&self) -> &'static str {
        "precomputed-vector-lookup"
    }
    fn requires_identified_inputs(&self) -> bool {
        true
    }

    fn validate_document_inputs(&self, inputs: &[EmbeddingInput]) -> Result<(), EmbeddingError> {
        self.bundle.ensure_documents(inputs)
    }

    fn validate_query_inputs(&self, inputs: &[EmbeddingInput]) -> Result<(), EmbeddingError> {
        self.bundle.ensure_queries(inputs)
    }

    fn embed_documents(&mut self, _documents: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(invalid(
            "bundle document lookup requires identified inputs; use embed_documents_with_ids",
        ))
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut matches = self
            .bundle
            .inputs
            .queries
            .iter()
            .filter(|input| input.text == query);
        let input = matches
            .next()
            .ok_or(EmbeddingError::Missing { kind: "query" })?;
        if matches.next().is_some() {
            return Err(invalid(
                "query text matches multiple bundle IDs; use embed_query_with_id",
            ));
        }
        Ok(self.bundle.query(input)?.vector().to_vec())
    }

    fn embed_documents_with_ids(
        &mut self,
        documents: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        documents
            .iter()
            .map(|input| self.bundle.document(input))
            .collect()
    }

    fn embed_query_with_id(&mut self, query: &EmbeddingInput) -> Result<Vec<f32>, EmbeddingError> {
        Ok(self.bundle.query(query)?.vector().to_vec())
    }
}

impl VectorBundle {
    /// Read and validate a manifest and all files named by it.
    pub fn load(manifest: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let manifest_path = manifest.as_ref();
        let manifest_bytes = fs::read(manifest_path)?;
        let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.format_version != 1 || manifest.dtype != "float32_le" {
            return Err(invalid("unsupported vector bundle version or dtype"));
        }
        manifest.config.validate()?;
        let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let inputs_bytes = read_verified(base, &manifest.inputs)?;
        let documents_bytes = read_verified(base, &manifest.documents)?;
        let queries_bytes = read_verified(base, &manifest.queries)?;
        let inputs: EmbeddingInputs = serde_json::from_slice(&inputs_bytes)?;
        inputs.validate()?;
        let documents = parse_matrix(
            &documents_bytes,
            inputs.documents.len(),
            manifest.config.dimensions,
            &manifest.config,
        )?;
        let queries = parse_matrix(
            &queries_bytes,
            inputs.queries.len(),
            manifest.config.dimensions,
            &manifest.config,
        )?;
        Ok(Self {
            config: manifest.config,
            document_rows: inputs
                .documents
                .iter()
                .enumerate()
                .map(|(row, input)| (input.id.clone(), row))
                .collect(),
            query_rows: inputs
                .queries
                .iter()
                .enumerate()
                .map(|(row, input)| (input.id.clone(), row))
                .collect(),
            inputs,
            documents,
            queries,
        })
    }

    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    pub fn inputs(&self) -> &EmbeddingInputs {
        &self.inputs
    }

    pub fn document(&self, input: &EmbeddingInput) -> Result<Vec<f32>, EmbeddingError> {
        let row = find_input(
            &self.inputs.documents,
            &self.document_rows,
            input,
            "document",
        )?;
        Ok(self.documents[row].clone())
    }

    pub fn query(&self, input: &EmbeddingInput) -> Result<EmbeddedQuery, EmbeddingError> {
        let row = find_input(&self.inputs.queries, &self.query_rows, input, "query")?;
        EmbeddedQuery::new(self.config.clone(), self.queries[row].clone())
    }

    /// Require caller-provided inputs to exactly match the ordered document rows.
    pub fn ensure_documents(&self, expected: &[EmbeddingInput]) -> Result<(), EmbeddingError> {
        ensure_role("document", &self.inputs.documents, expected)
    }

    /// Require caller-provided inputs to exactly match the ordered query rows.
    pub fn ensure_queries(&self, expected: &[EmbeddingInput]) -> Result<(), EmbeddingError> {
        ensure_role("query", &self.inputs.queries, expected)
    }

    /// Retrieve a document by ID while also checking its exact text and content hash.
    pub fn document_by_id(&self, id: &str, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let input = self
            .inputs
            .documents
            .iter()
            .find(|input| input.id == id)
            .ok_or(EmbeddingError::Missing { kind: "document" })?;
        ensure_same_content(input, text)?;
        self.document(input)
    }

    /// Retrieve a query by ID while also checking its exact text and content hash.
    pub fn query_by_id(&self, id: &str, text: &str) -> Result<EmbeddedQuery, EmbeddingError> {
        let input = self
            .inputs
            .queries
            .iter()
            .find(|input| input.id == id)
            .ok_or(EmbeddingError::Missing { kind: "query" })?;
        ensure_same_content(input, text)?;
        self.query(input)
    }
}

fn validate_input_role(
    role: &str,
    inputs: &[EmbeddingInput],
    require_text: bool,
) -> Result<(), EmbeddingError> {
    let mut ids = HashSet::with_capacity(inputs.len());
    for input in inputs {
        input.validate()?;
        if require_text && input.text.trim().is_empty() {
            return Err(invalid("query text must be nonempty"));
        }
        if !ids.insert(&input.id) {
            return Err(invalid(format!("duplicate {role} ID {:?}", input.id)));
        }
    }
    Ok(())
}

fn ensure_role(
    role: &str,
    actual: &[EmbeddingInput],
    expected: &[EmbeddingInput],
) -> Result<(), EmbeddingError> {
    validate_input_role(role, expected, role == "query")?;
    if actual != expected {
        return Err(invalid(format!(
            "{role} inputs do not match bundle rows in order"
        )));
    }
    Ok(())
}

fn find_input(
    rows: &[EmbeddingInput],
    index: &HashMap<String, usize>,
    requested: &EmbeddingInput,
    role: &'static str,
) -> Result<usize, EmbeddingError> {
    requested.validate()?;
    let row = index
        .get(&requested.id)
        .copied()
        .ok_or(EmbeddingError::Missing { kind: role })?;
    if rows[row] != *requested {
        return Err(invalid(format!(
            "{role} ID {:?} has different content",
            requested.id
        )));
    }
    Ok(row)
}

fn ensure_same_content(input: &EmbeddingInput, text: &str) -> Result<(), EmbeddingError> {
    if input.text != text || blake3::hash(text.as_bytes()).to_hex().as_str() != input.content_hash {
        return Err(invalid(format!(
            "input ID {:?} has different content",
            input.id
        )));
    }
    Ok(())
}

fn read_verified(base: &Path, entry: &ManifestFile) -> Result<Vec<u8>, EmbeddingError> {
    if !valid_sha256(&entry.sha256) {
        return Err(invalid(
            "manifest SHA-256 must be 64 hexadecimal characters",
        ));
    }
    let path = Path::new(&entry.file);
    if entry.file.is_empty()
        || entry.file == "."
        || entry.file == ".."
        || entry.file.contains('/')
        || entry.file.contains('\\')
        || path.components().count() != 1
        || !matches!(
            path.components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err(invalid("bundle filenames must be single path components"));
    }
    let bytes = fs::read(base.join(path))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if !digest.eq_ignore_ascii_case(&entry.sha256) {
        return Err(invalid(format!("SHA-256 mismatch for {:?}", entry.file)));
    }
    Ok(bytes)
}

fn parse_matrix(
    bytes: &[u8],
    rows: usize,
    dimensions: usize,
    config: &EmbeddingConfig,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let value_count = rows
        .checked_mul(dimensions)
        .ok_or_else(|| invalid("vector matrix size overflows"))?;
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| invalid("vector matrix byte size overflows"))?;
    if bytes.len() != byte_count {
        return Err(invalid(format!(
            "expected {byte_count} matrix bytes for {rows} rows and {dimensions} dimensions, got {}",
            bytes.len()
        )));
    }
    let mut vectors = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut vector = Vec::with_capacity(dimensions);
        for col in 0..dimensions {
            let offset = (row * dimensions + col) * 4;
            vector.push(f32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("matrix length was checked"),
            ));
        }
        config.validate_vector(&vector)?;
        vectors.push(vector);
    }
    Ok(vectors)
}

fn valid_sha256(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn invalid(message: impl Into<String>) -> EmbeddingError {
    EmbeddingError::Invalid(message.into())
}
