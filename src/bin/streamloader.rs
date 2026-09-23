use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde::Serialize;
use streamloader::{Block, BlockConfig, LoaderError, Model, OpenOptions, TensorDescriptor};

/// Shared help text for the `--trust-root` flag every subcommand exposes.
const TRUST_ROOT_HELP: &str = "Widen the symlink-escape trust boundary for index-referenced \
shard paths to this directory instead of the index file's own directory. Needed for real \
Hugging Face hub caches, which symlink shard files into a shared blobs/ directory outside the \
snapshot dir. Only widen this to a directory you actually trust.";

fn open_model(path: &Path, trust_root: Option<PathBuf>) -> Result<Model, LoaderError> {
    let opts = OpenOptions {
        trusted_root: trust_root,
        ..Default::default()
    };
    Model::open_with(path, &opts)
}

#[derive(Parser)]
#[command(
    name = "streamloader",
    about = "Index and inspect SafeTensors checkpoints without pre-splitting or copying them"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List tensor metadata for a checkpoint.
    Inspect {
        path: PathBuf,
        #[arg(long, help = TRUST_ROOT_HELP)]
        trust_root: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// List logical blocks grouped from tensor names.
    Blocks {
        path: PathBuf,
        #[arg(long, value_enum, default_value = "generic")]
        preset: Preset,
        #[arg(long)]
        model_prefix: Option<String>,
        #[arg(long, help = TRUST_ROOT_HELP)]
        trust_root: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Show every tensor belonging to exactly one block id.
    Block {
        path: PathBuf,
        #[arg(long)]
        id: String,
        #[arg(long, value_enum, default_value = "generic")]
        preset: Preset,
        #[arg(long)]
        model_prefix: Option<String>,
        #[arg(long, help = TRUST_ROOT_HELP)]
        trust_root: Option<PathBuf>,
        /// Also perform an explicit owned copy of the block and report the
        /// resulting buffer layout (distinct from the zero-copy view path).
        #[arg(long)]
        copy: bool,
        #[arg(long)]
        json: bool,
    },
    /// Read and checksum the actual payload bytes of a block or tensor.
    Verify {
        path: PathBuf,
        #[arg(long, conflicts_with = "tensor")]
        block: Option<String>,
        /// Verify one exact tensor by name instead of a whole block --
        /// the selection path for shared/unassigned tensors.
        #[arg(long, conflicts_with = "block")]
        tensor: Option<String>,
        #[arg(long, value_enum, default_value = "generic")]
        preset: Preset,
        #[arg(long)]
        model_prefix: Option<String>,
        #[arg(long, help = TRUST_ROOT_HELP)]
        trust_root: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Preset {
    Flux,
    Generic,
}

fn build_config(preset: Preset, model_prefix: Option<String>) -> BlockConfig {
    let config = match preset {
        Preset::Flux => BlockConfig::flux(),
        Preset::Generic => BlockConfig::generic(),
    };
    match model_prefix {
        Some(p) => config.with_model_prefix(p),
        None => config,
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), LoaderError> {
    match cli.command {
        Command::Inspect {
            path,
            trust_root,
            json,
        } => cmd_inspect(&path, trust_root, json),
        Command::Blocks {
            path,
            preset,
            model_prefix,
            trust_root,
            json,
        } => cmd_blocks(&path, trust_root, build_config(preset, model_prefix), json),
        Command::Block {
            path,
            id,
            preset,
            model_prefix,
            trust_root,
            copy,
            json,
        } => cmd_block(
            &path,
            trust_root,
            &id,
            build_config(preset, model_prefix),
            copy,
            json,
        ),
        Command::Verify {
            path,
            block,
            tensor,
            preset,
            model_prefix,
            trust_root,
            json,
        } => cmd_verify(
            &path,
            trust_root,
            block,
            tensor,
            build_config(preset, model_prefix),
            json,
        ),
    }
}

#[derive(Serialize)]
struct TensorJson<'a> {
    name: &'a str,
    dtype: safetensors::Dtype,
    shape: &'a [usize],
    shard_id: usize,
    shard_path: String,
    file_offset: u64,
    byte_len: u64,
}

fn tensor_json<'a>(model: &'a Model, d: &'a TensorDescriptor) -> TensorJson<'a> {
    let shard_path = model
        .shard_paths()
        .nth(d.shard_id)
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    TensorJson {
        name: &d.name,
        dtype: d.dtype,
        shape: &d.shape,
        shard_id: d.shard_id,
        shard_path,
        file_offset: d.file_offset,
        byte_len: d.byte_len,
    }
}

fn cmd_inspect(path: &Path, trust_root: Option<PathBuf>, json: bool) -> Result<(), LoaderError> {
    let model = open_model(path, trust_root)?;

    if json {
        #[derive(Serialize)]
        struct Out<'a> {
            root_input: String,
            shard_count: usize,
            shard_paths: Vec<String>,
            tensor_count: usize,
            tensors: Vec<TensorJson<'a>>,
        }
        let tensors: Vec<TensorJson> = model
            .descriptors()
            .map(|d| tensor_json(&model, d))
            .collect();
        let out = Out {
            root_input: model.root_input().display().to_string(),
            shard_count: model.shard_paths().count(),
            shard_paths: model
                .shard_paths()
                .map(|p| p.display().to_string())
                .collect(),
            tensor_count: model.tensor_count(),
            tensors,
        };
        println!("{}", serde_json::to_string_pretty(&out).expect("serialize"));
    } else {
        eprintln!("root: {}", model.root_input().display());
        for p in model.shard_paths() {
            eprintln!("shard: {}", p.display());
        }
        println!("{} tensors", model.tensor_count());
        for d in model.descriptors() {
            println!(
                "{}\t{}\t{:?}\tshard={}\toffset={}\tlen={}",
                d.name, d.dtype, d.shape, d.shard_id, d.file_offset, d.byte_len
            );
        }
    }
    Ok(())
}

fn block_json(block: &Block) -> serde_json::Value {
    serde_json::json!({
        "id": block.id,
        "family": block.family,
        "index": block.index,
        "tensor_count": block.tensors.len(),
        "tensor_names": block.tensors.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
    })
}

fn cmd_blocks(
    path: &Path,
    trust_root: Option<PathBuf>,
    config: BlockConfig,
    json: bool,
) -> Result<(), LoaderError> {
    let model = open_model(path, trust_root)?;
    let blocks = model.blocks(&config);

    if json {
        let out: Vec<serde_json::Value> = blocks.iter().map(block_json).collect();
        println!("{}", serde_json::to_string_pretty(&out).expect("serialize"));
    } else {
        for b in &blocks {
            println!("{}\t{} tensors", b.id, b.tensors.len());
        }
    }
    Ok(())
}

fn cmd_block(
    path: &Path,
    trust_root: Option<PathBuf>,
    id: &str,
    config: BlockConfig,
    copy: bool,
    json: bool,
) -> Result<(), LoaderError> {
    let model = open_model(path, trust_root)?;
    let block = model.block(&config, id)?;

    let copied = if copy {
        Some(model.copy_block(&block)?)
    } else {
        None
    };

    if json {
        let mut out = serde_json::json!({
            "id": block.id,
            "family": block.family,
            "index": block.index,
            "tensors": block.tensors.iter().map(|d| tensor_json(&model, d)).collect::<Vec<_>>(),
        });
        if let Some(c) = &copied {
            out["copy"] = serde_json::json!({
                "bytes_copied": c.bytes_copied,
                "layout": c.layout.iter().map(|l| serde_json::json!({
                    "name": l.name, "offset": l.offset, "len": l.len,
                })).collect::<Vec<_>>(),
            });
        }
        println!("{}", serde_json::to_string_pretty(&out).expect("serialize"));
    } else {
        println!("block {} ({} tensors)", block.id, block.tensors.len());
        for d in &block.tensors {
            println!(
                "  {}\t{}\t{:?}\tlen={}",
                d.name, d.dtype, d.shape, d.byte_len
            );
        }
        if let Some(c) = &copied {
            eprintln!("copied {} bytes into an owned buffer", c.bytes_copied);
        }
    }
    Ok(())
}

fn cmd_verify(
    path: &Path,
    trust_root: Option<PathBuf>,
    block_id: Option<String>,
    tensor_name: Option<String>,
    config: BlockConfig,
    json: bool,
) -> Result<(), LoaderError> {
    let model = open_model(path, trust_root)?;

    let (target_desc, checksums) = if let Some(name) = &tensor_name {
        let c = model.checksum_tensor(name)?;
        (format!("tensor:{name}"), vec![c])
    } else if let Some(id) = &block_id {
        let block = model.block(&config, id)?;
        (format!("block:{id}"), model.checksum_block(&block)?)
    } else {
        return Err(LoaderError::InvalidUsage(
            "verify requires either --block <id> or --tensor <name>".to_string(),
        ));
    };

    let total_bytes: u64 = checksums.iter().map(|c| c.bytes).sum();

    if json {
        let out = serde_json::json!({
            "target": target_desc,
            "algorithm": checksums.first().map(|c| c.algorithm).unwrap_or("blake3"),
            "tensors": checksums.iter().map(|c| serde_json::json!({
                "name": c.name, "hex": c.hex, "bytes": c.bytes,
            })).collect::<Vec<_>>(),
            "total_bytes": total_bytes,
        });
        println!("{}", serde_json::to_string_pretty(&out).expect("serialize"));
    } else {
        println!(
            "verified {target_desc}: {} tensor(s), {total_bytes} bytes",
            checksums.len()
        );
        for c in &checksums {
            println!("  {} {} {}", c.algorithm, c.hex, c.name);
        }
    }
    Ok(())
}
