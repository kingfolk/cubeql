use proc_macro2::Span;
use quote::ToTokens;
use syn::{Item, spanned::Spanned};

/// One source unit. Line numbers are inclusive and one-based; paths are repository-relative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeChunk {
    pub path: String,
    pub language: String,
    pub symbol: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
    /// BLAKE3 of the exact UTF-8 `source`, not its position or filesystem timestamp.
    pub content_hash: String,
    pub source: String,
}

pub(crate) fn language(path: &std::path::Path) -> Option<&'static str> {
    Some(match path.extension()?.to_str()? {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "swift" => "swift",
        "rb" => "ruby",
        "sh" | "bash" | "zsh" => "shell",
        "sql" => "sql",
        "toml" => "toml",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "md" | "mdx" | "txt" => "text",
        _ => return None,
    })
}

pub(crate) fn chunk_file(
    path: &str,
    language: &str,
    source: &str,
    fallback_lines: usize,
) -> (Vec<CodeChunk>, bool) {
    let mut collector = Collector {
        path,
        language,
        source,
        parsed_offset: 0,
        line_starts: std::iter::once(0)
            .chain(source.match_indices('\n').map(|(byte, _)| byte + 1))
            .collect(),
        chunks: Vec::new(),
    };
    if language == "rust"
        && let Ok(file) = syn::parse_file(source)
    {
        // syn strips an optional BOM and shebang. Its span ranges refer to the remaining
        // input, whereas the persisted source and line map refer to the original file.
        collector.parsed_offset = usize::from(source.starts_with('\u{feff}'))
            * '\u{feff}'.len_utf8()
            + file.shebang.as_ref().map_or(0, String::len);
        collector.items(&file.items, "");
        // Comment-only or otherwise item-free files are still useful source documents.
        if !collector.chunks.is_empty() || source.trim().is_empty() {
            return (collector.chunks, false);
        }
    }
    for (block, lines) in source
        .split_inclusive('\n')
        .collect::<Vec<_>>()
        .chunks(fallback_lines)
        .enumerate()
    {
        let text = lines.concat();
        if text.trim().is_empty() {
            continue;
        }
        let start = block * fallback_lines + 1;
        collector.chunks.push(CodeChunk {
            path: path.into(),
            language: language.into(),
            symbol: format!("lines:{start}-{}", start + lines.len() - 1),
            kind: "lines".into(),
            start_line: start as u32,
            end_line: (start + lines.len() - 1) as u32,
            content_hash: blake3::hash(text.as_bytes()).to_hex().to_string(),
            source: text,
        });
    }
    (collector.chunks, language == "rust")
}

struct Collector<'a> {
    path: &'a str,
    language: &'a str,
    source: &'a str,
    parsed_offset: usize,
    line_starts: Vec<usize>,
    chunks: Vec<CodeChunk>,
}

impl Collector<'_> {
    fn emit(&mut self, span: Span, kind: &str, symbol: &str) {
        self.emit_range(span.byte_range(), kind, symbol);
    }

    fn header(&mut self, span: Span, open: Span, kind: &str, symbol: &str) {
        // Include attributes/visibility/generics but stop at the opening brace. Child bodies
        // become separate chunks, so changing a method does not invalidate its container.
        self.emit_range(span.byte_range().start..open.byte_range().end, kind, symbol);
    }

    fn emit_range(&mut self, range: std::ops::Range<usize>, kind: &str, symbol: &str) {
        let range = range.start + self.parsed_offset..range.end + self.parsed_offset;
        let text = &self.source[range.clone()];
        self.chunks.push(CodeChunk {
            path: self.path.into(),
            language: self.language.into(),
            symbol: symbol.into(),
            kind: kind.into(),
            start_line: self
                .line_starts
                .partition_point(|byte| *byte <= range.start) as u32,
            end_line: self.line_starts.partition_point(|byte| *byte < range.end) as u32,
            content_hash: blake3::hash(text.as_bytes()).to_hex().to_string(),
            source: text.into(),
        });
    }

    fn items(&mut self, items: &[Item], scope: &str) {
        for item in items {
            let named = |name: String| {
                if scope.is_empty() {
                    name
                } else {
                    format!("{scope}::{name}")
                }
            };
            match item {
                Item::Fn(v) => self.emit(v.span(), "function", &named(v.sig.ident.to_string())),
                Item::Struct(v) => self.emit(v.span(), "struct", &named(v.ident.to_string())),
                Item::Enum(v) => self.emit(v.span(), "enum", &named(v.ident.to_string())),
                Item::Union(v) => self.emit(v.span(), "union", &named(v.ident.to_string())),
                Item::Type(v) => self.emit(v.span(), "type", &named(v.ident.to_string())),
                Item::Const(v) => self.emit(v.span(), "const", &named(v.ident.to_string())),
                Item::Static(v) => self.emit(v.span(), "static", &named(v.ident.to_string())),
                Item::Mod(v) => {
                    let symbol = named(v.ident.to_string());
                    if let Some((brace, items)) = &v.content {
                        self.header(v.span(), brace.span.open(), "module", &symbol);
                        self.items(items, &symbol);
                    } else {
                        self.emit(v.span(), "module", &symbol);
                    }
                }
                Item::Impl(v) => {
                    let ty = v.self_ty.to_token_stream().to_string();
                    let name = if let Some((_, tr, _)) = &v.trait_ {
                        format!("<{ty} as {}>", tr.to_token_stream())
                    } else {
                        ty
                    };
                    let symbol = named(name);
                    self.header(v.span(), v.brace_token.span.open(), "impl", &symbol);
                    for child in &v.items {
                        let (kind, name) = match child {
                            syn::ImplItem::Fn(f) => ("method", f.sig.ident.to_string()),
                            syn::ImplItem::Const(c) => ("const", c.ident.to_string()),
                            syn::ImplItem::Type(t) => ("type", t.ident.to_string()),
                            _ => ("impl_item", "item".into()),
                        };
                        self.emit(child.span(), kind, &format!("{symbol}::{name}"));
                    }
                }
                Item::Trait(v) => {
                    let symbol = named(v.ident.to_string());
                    self.header(v.span(), v.brace_token.span.open(), "trait", &symbol);
                    for child in &v.items {
                        let (kind, name) = match child {
                            syn::TraitItem::Fn(f) => ("trait_method", f.sig.ident.to_string()),
                            syn::TraitItem::Const(c) => ("const", c.ident.to_string()),
                            syn::TraitItem::Type(t) => ("type", t.ident.to_string()),
                            _ => ("trait_item", "item".into()),
                        };
                        self.emit(child.span(), kind, &format!("{symbol}::{name}"));
                    }
                }
                Item::Macro(v) => {
                    let name = v
                        .ident
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| v.mac.path.to_token_stream().to_string());
                    self.emit(v.span(), "macro", &named(name));
                }
                _ => self.emit(item.span(), "item", &named("item".into())),
            }
        }
    }
}
