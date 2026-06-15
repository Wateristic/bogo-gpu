use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;
use toml::{from_str, to_string_pretty};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML deserialize error: {0}")]
    TomlDe(#[from] toml::de::Error),
    #[error("TOML serialize error: {0}")]
    TomlSe(#[from] toml::ser::Error),
    #[error("Invalid config")]
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ComputeBackend {
    Gpu,
    Cpu,
}

impl Default for ComputeBackend {
    fn default() -> Self { ComputeBackend::Gpu }
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeBackend::Gpu => write!(f, "GPU"),
            ComputeBackend::Cpu => write!(f, "CPU"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "identity")]
    pub user: UserConfig,
    #[serde(default)]
    pub compute: ComputeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    pub uuid:     String,
    pub nickname: String,
    pub code:     String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    /// Which backend to use.
    #[serde(default)]
    pub backend: ComputeBackend,
    /// GPU arch string passed to the compiler (e.g. "gfx1201", "sm_86").
    #[serde(default = "default_gpu_arch")]
    pub gpu_arch: String,
    /// Number of GPU thread blocks.
    #[serde(default = "default_blocks")]
    pub blocks: u32,
    /// Number of GPU threads per block.
    #[serde(default = "default_threads")]
    pub threads: u32,
    /// Number of CPU threads (0 = use all cores).
    #[serde(default)]
    pub cpu_threads: u32,
}

fn default_gpu_arch()  -> String { "gfx1201".into() }
fn default_blocks()    -> u32    { 256 }
fn default_threads()   -> u32    { 256 }

impl Default for ComputeConfig {
    fn default() -> Self {
        ComputeConfig {
            backend:     ComputeBackend::Gpu,
            gpu_arch:    default_gpu_arch(),
            blocks:      default_blocks(),
            threads:     default_threads(),
            cpu_threads: 0,
        }
    }
}

impl Config {
    pub fn config_dir() -> PathBuf {
        dirs::config_dir()
            .map(|p| p.join("bogo-gpu"))
            .unwrap_or_else(|| PathBuf::from(".config/bogo-gpu"))
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.toml")
    }

    pub fn load() -> Result<Option<Self>, ConfigError> {
        let path = Self::config_path();
        if !path.exists() {
            return Ok(None);
        }
        let contents = std::fs::read_to_string(path)?;
        let config = from_str(&contents)?;
        Ok(Some(config))
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        let dir = Self::config_dir();
        std::fs::create_dir_all(&dir)?;
        let contents = to_string_pretty(self)?;
        std::fs::write(Self::config_path(), contents)?;
        Ok(())
    }
}
