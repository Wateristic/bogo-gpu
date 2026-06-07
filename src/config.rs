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
 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "identity")]
    pub user: UserConfig,
}
 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    pub uuid: String,
    pub nickname: String,
    pub code: String,
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