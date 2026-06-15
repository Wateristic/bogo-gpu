use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub v: u32,
    pub uuid: String,
    pub nickname: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub seed: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub seed: String,
    pub total_done: u64,
    pub best_correct: u32,
    pub best_arr: Vec<u32>,
    pub best_index: u64,
}
