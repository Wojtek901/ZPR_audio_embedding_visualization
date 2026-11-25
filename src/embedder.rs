pub mod audio;
pub mod clap;
pub mod text;

use anyhow::Result;
use ndarray::Array1;

use serde::{Deserialize, Serialize};

pub type Embedding = Array1<f32>;

pub trait EmbeddingModel: Send + Sync{
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed_test(&self, text: &str) -> Result<Embedding>;
    fn embed_audio(&self, audio_bytes: &[u8]) -> Result<Embedding>;
}

