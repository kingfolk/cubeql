use std::path::Path;

use cubecl::Device;
use cubeql::{
    Filter, FilterMode, IndexQueryOptions, Metric, PreparedFilter, SearchHit, VectorIndex,
};
use cubeql_embeddings::{EmbeddedQuery, EmbeddingConfig, EmbeddingProvider, embed_query};

use crate::{
    BuildOptions, BuildResult, CodeError, CodeIndex, EMBEDDING_CONFIG, build_repository_batched,
    invalid,
};

/// Only unmatched documents are sent to the provider, in ordered batches. The provider's
/// complete vector-space contract must match both options and any previous snapshot.
pub fn build_repository_with_provider(
    root: impl AsRef<Path>,
    options: &BuildOptions,
    previous: Option<&CodeIndex>,
    provider: &mut dyn EmbeddingProvider,
    batch_size: usize,
) -> Result<BuildResult, CodeError> {
    let config = provider.config().clone();
    config.validate()?;
    if options.dimensions != config.dimensions || options.embedding_id != config.fingerprint()? {
        return Err(invalid(
            "use BuildOptions::for_embedding with the selected provider",
        ));
    }
    if let Some(previous) = previous {
        let old = previous.embedding_config().ok_or_else(|| {
            invalid("legacy indexes have no provider contract; rebuild without --previous")
        })?;
        config.ensure_compatible(old)?;
    }
    let mut result =
        build_repository_batched(root.as_ref(), options, previous, batch_size, |texts| {
            let vectors = provider.embed_documents(texts)?;
            config.ensure_compatible(provider.config())?;
            config.validate_vectors(&vectors, texts.len())?;
            Ok(vectors)
        })?;
    result.index.batch.schema_metadata_mut().insert(
        EMBEDDING_CONFIG,
        serde_json::to_string(&config).map_err(cubeql_embeddings::EmbeddingError::from)?,
    );
    result.index.embedding_config = Some(config);
    Ok(result)
}

/// Application-level guard around the generic vector engine. It binds source metadata,
/// resident vectors and their embedding contract to the same snapshot. Core CubeQL remains
/// model-agnostic; callers using raw VectorIndex are responsible for their own model checks.
pub struct CodeSearchIndex {
    code: CodeIndex,
    vectors: VectorIndex,
    config: EmbeddingConfig,
}

impl CodeSearchIndex {
    pub fn new(device: &Device, code: CodeIndex) -> Result<Self, CodeError> {
        let config = code.embedding_config().cloned().ok_or_else(|| {
            invalid("legacy index has no embedding contract; rebuild with a provider")
        })?;
        let vectors = VectorIndex::new(device, code.batch().clone(), "embedding")
            .map_err(crate::PersistenceError::from)?;
        Ok(Self {
            code,
            vectors,
            config,
        })
    }

    pub fn code(&self) -> &CodeIndex {
        &self.code
    }

    /// Prepare a reusable host-evaluated metadata selection. Empty filters avoid allocating
    /// an all-rows selection. Returned row IDs always refer to this source snapshot.
    pub fn prepare_filter(
        &self,
        filters: &[Filter<'_>],
    ) -> Result<PreparedCodeSearch<'_>, CodeError> {
        let selection = if filters.is_empty() {
            None
        } else {
            Some(
                self.vectors
                    .prepare_filter(filters, FilterMode::BeforeScoring)
                    .map_err(crate::PersistenceError::from)?,
            )
        };
        Ok(PreparedCodeSearch {
            index: self,
            selection,
        })
    }

    pub fn query(
        &self,
        query: &EmbeddedQuery,
        top_k: usize,
        metric: Metric,
    ) -> Result<Vec<SearchHit>, CodeError> {
        self.config.ensure_compatible(query.config())?;
        Ok(self
            .vectors
            .query_with_metric(query.vector(), IndexQueryOptions::new(top_k), metric)
            .map_err(crate::PersistenceError::from)?)
    }

    pub fn query_text(
        &self,
        provider: &mut dyn EmbeddingProvider,
        text: &str,
        top_k: usize,
        metric: Metric,
    ) -> Result<Vec<SearchHit>, CodeError> {
        // Reject wrong models before paying for inference or executing a device query.
        self.config.ensure_compatible(provider.config())?;
        self.query(&embed_query(provider, text)?, top_k, metric)
    }
}

pub struct PreparedCodeSearch<'a> {
    index: &'a CodeSearchIndex,
    selection: Option<PreparedFilter<'a>>,
}

impl PreparedCodeSearch<'_> {
    pub fn query(
        &self,
        query: &EmbeddedQuery,
        top_k: usize,
        metric: Metric,
    ) -> Result<Vec<SearchHit>, CodeError> {
        self.index.config.ensure_compatible(query.config())?;
        match &self.selection {
            Some(selected) => Ok(selected
                .query(query.vector(), top_k, metric)
                .map_err(crate::PersistenceError::from)?),
            None => self.index.query(query, top_k, metric),
        }
    }
}
