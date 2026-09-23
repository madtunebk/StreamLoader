//! Deterministic prefix-based grouping of tensors into logical transformer
//! blocks. Purely a naming/grouping label: it says nothing about execution
//! order or GPU residency.

use std::collections::BTreeMap;

use crate::descriptor::TensorDescriptor;

/// Label used for tensors that did not match any configured block family.
pub const UNASSIGNED: &str = "shared/unassigned";

/// One ordered family of blocks, e.g. `transformer_blocks.<n>.*`.
/// `label` is both the match prefix and the prefix used to build block ids
/// (`format!("{label}.{index}")`).
#[derive(Debug, Clone)]
pub struct BlockFamily {
    pub label: String,
}

impl BlockFamily {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

/// Ordered list of block families plus an optional shared name prefix, e.g.
/// `model` for tensors named `model.layers.0.*`. Families are tried in
/// order; the first match wins. This is the "simple documented
/// override/config" the spec asks for instead of a plugin system: build
/// your own `BlockConfig` with whatever families/order your architecture
/// uses.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// Optional shared prefix, matched as a whole leading dotted segment
    /// (or sequence of segments). Only tensors starting with exactly this
    /// prefix have it considered; tensors that don't start with it are
    /// matched against families using their full original name, so an
    /// optional prefix never silently swallows unrelated name portions.
    pub model_prefix: Option<String>,
    pub families: Vec<BlockFamily>,
}

impl BlockConfig {
    pub fn new(families: Vec<BlockFamily>) -> Self {
        Self {
            model_prefix: None,
            families,
        }
    }

    pub fn with_model_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.model_prefix = Some(prefix.into());
        self
    }

    /// The four initial conventions named in the spec, tried in this
    /// declared order. Used when no preset/config is given explicitly.
    pub fn generic() -> Self {
        Self::new(vec![
            BlockFamily::new("transformer_blocks"),
            BlockFamily::new("single_transformer_blocks"),
            BlockFamily::new("blocks"),
            BlockFamily::new("layers"),
        ])
    }

    /// FLUX preset: `transformer_blocks` are listed before
    /// `single_transformer_blocks`; within each family, indices sort
    /// numerically (2 before 10), never lexicographically, and never by
    /// scanning for arbitrary digits elsewhere in the tensor name.
    pub fn flux() -> Self {
        Self::new(vec![
            BlockFamily::new("transformer_blocks"),
            BlockFamily::new("single_transformer_blocks"),
        ])
    }

    /// Try to match `name` against the configured families, after
    /// stripping `model_prefix` if configured and present. Returns
    /// `(family_label, index)` on a match.
    fn match_name(&self, name: &str) -> Option<(&str, u64)> {
        let candidate = match &self.model_prefix {
            Some(prefix) => {
                let with_dot = format!("{prefix}.");
                name.strip_prefix(&with_dot).unwrap_or(name)
            }
            None => name,
        };

        let parts: Vec<&str> = candidate.split('.').collect();
        for family in &self.families {
            let family_parts: Vec<&str> = family.label.split('.').collect();
            if parts.len() <= family_parts.len() {
                continue;
            }
            if parts[..family_parts.len()] != family_parts[..] {
                continue;
            }
            let idx_str = parts[family_parts.len()];
            if !idx_str.is_empty()
                && idx_str.bytes().all(|b| b.is_ascii_digit())
                && let Ok(idx) = idx_str.parse::<u64>()
            {
                return Some((family.label.as_str(), idx));
            }
        }
        None
    }

    /// Canonical block id for a matched family/index, e.g.
    /// `transformer_blocks.12`. Deliberately excludes `model_prefix` so ids
    /// are stable regardless of that optional wrapper.
    fn block_id(family: &str, index: u64) -> String {
        format!("{family}.{index}")
    }
}

/// One logical block: its id and the tensors assigned to it, in the order
/// they were encountered in the input descriptor list.
///
/// `model_id` identifies which [`crate::model::Model`] this block's
/// descriptors were built from. It is set to `0` by the free functions in
/// this module (which know nothing about `Model`) and stamped with the
/// real id by `Model::blocks`/`Model::block`. Passing a `Block` to a
/// method on a *different* `Model` than the one that produced it is
/// rejected at that call site precisely because this field won't match —
/// see `Model::block_views`/`copy_block`/`checksum_block`.
#[derive(Debug, Clone)]
pub struct Block {
    pub id: String,
    pub family: String,
    pub index: Option<u64>,
    pub tensors: Vec<TensorDescriptor>,
    pub model_id: u64,
}

/// Group `descriptors` into blocks per `config`. Returns blocks ordered
/// per the spec: families in configured order, indices numeric ascending
/// within a family, then `shared/unassigned` last. A block can span
/// multiple shards and discontiguous byte ranges; this just partitions the
/// descriptor list, it never assumes single-shard/contiguous layout.
pub fn group_blocks(descriptors: &[TensorDescriptor], config: &BlockConfig) -> Vec<Block> {
    // family label -> index -> tensors, using a BTreeMap<u64,_> so indices
    // come out numerically sorted for free without a lexicographic trap.
    let mut by_family: BTreeMap<&str, BTreeMap<u64, Vec<TensorDescriptor>>> = BTreeMap::new();
    let mut unassigned: Vec<TensorDescriptor> = Vec::new();

    for family in &config.families {
        by_family.entry(family.label.as_str()).or_default();
    }

    for d in descriptors {
        match config.match_name(&d.name) {
            Some((family, idx)) => {
                by_family
                    .entry(family)
                    .or_default()
                    .entry(idx)
                    .or_default()
                    .push(d.clone());
            }
            None => unassigned.push(d.clone()),
        }
    }

    let mut out = Vec::new();
    // Iterate families in *configured* order (not BTreeMap's alphabetical
    // order), since e.g. FLUX requires transformer_blocks before
    // single_transformer_blocks regardless of string sort.
    for family in &config.families {
        if let Some(by_index) = by_family.get(family.label.as_str()) {
            for (idx, tensors) in by_index {
                out.push(Block {
                    id: BlockConfig::block_id(&family.label, *idx),
                    family: family.label.clone(),
                    index: Some(*idx),
                    tensors: tensors.clone(),
                    model_id: 0,
                });
            }
        }
    }

    if !unassigned.is_empty() {
        out.push(Block {
            id: UNASSIGNED.to_string(),
            family: UNASSIGNED.to_string(),
            index: None,
            tensors: unassigned,
            model_id: 0,
        });
    }

    out
}

/// Find one block's tensors by exact id (e.g. `transformer_blocks.1`, which
/// must never match `transformer_blocks.10`). Also accepts the literal
/// `shared/unassigned` id.
///
/// Does a single filtering pass over `descriptors` for just the requested
/// id, rather than building every other block first (as calling
/// `group_blocks(..).find(..)` would) — listing/fetching one block should
/// cost proportional to the checkpoint's tensor count, not to the number
/// of blocks times the tensor count.
pub fn find_block(
    descriptors: &[TensorDescriptor],
    config: &BlockConfig,
    block_id: &str,
) -> Option<Block> {
    if block_id == UNASSIGNED {
        let tensors: Vec<TensorDescriptor> = descriptors
            .iter()
            .filter(|d| config.match_name(&d.name).is_none())
            .cloned()
            .collect();
        return if tensors.is_empty() {
            None
        } else {
            Some(Block {
                id: UNASSIGNED.to_string(),
                family: UNASSIGNED.to_string(),
                index: None,
                tensors,
                model_id: 0,
            })
        };
    }

    for family in &config.families {
        let Some(idx_str) = block_id
            .strip_prefix(family.label.as_str())
            .and_then(|rest| rest.strip_prefix('.'))
        else {
            continue;
        };
        let Ok(idx) = idx_str.parse::<u64>() else {
            continue;
        };
        let tensors: Vec<TensorDescriptor> = descriptors
            .iter()
            .filter(|d| config.match_name(&d.name) == Some((family.label.as_str(), idx)))
            .cloned()
            .collect();
        if !tensors.is_empty() {
            return Some(Block {
                id: block_id.to_string(),
                family: family.label.clone(),
                index: Some(idx),
                tensors,
                model_id: 0,
            });
        }
    }
    None
}
