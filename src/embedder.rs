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
    fn embed_text(&self, text: &str) -> Result<Embedding>;
    fn embed_audio(&self, audio_bytes: &[u8]) -> Result<Embedding>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClapVariant {
    General,
    MusicAndSpeech,
}

impl ClapVariant{
    pub fn hf_repo(&self) -> (&'static str, &'static str){
        ("Xenova", "clap-htsat-unfused")
    }
}

pub fn l2_normalize(v: Embedding) -> Embedding{
    let denominator: f32 = v.dot(&v).sqrt();

    if denominator < f32::EPSILON{
        Array1::<f32>::zeros(v.len())
    } else{
        v / denominator
    }
}

pub struct MockEmbedder{
    dim: usize,
}

impl MockEmbedder{
    pub fn new(dim: usize) -> Self{
        Self{ dim: dim}
    }
}

impl EmbeddingModel for MockEmbedder {
    fn name(&self) -> &str {
        "MockEmbedder"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_text(&self, _text: &str) -> Result<Embedding> {
        let mut embedding: Embedding = Array1::linspace(0.0, 10.0, self.dim());
        embedding = l2_normalize(embedding);
        Ok(embedding)
    }

    fn embed_audio(&self, _audio_bytes: &[u8]) -> Result<Embedding> {
        let mut embedding: Embedding = Array1::linspace(10.0, 20.0, self.dim());
        embedding = l2_normalize(embedding);
        Ok(embedding)
    }
}








#[cfg(test)]
mod tests{
    use ndarray::array;

use super::*;

    #[test]
    fn try_l2_normalize_normal(){
        let sample_vec:Array1<f32> = array![6.0, 8.0];
        let normalized_sample_vec = l2_normalize(sample_vec);
        let length_normalized = normalized_sample_vec.dot(&normalized_sample_vec);
        assert!(length_normalized > 1.0 - f32::EPSILON && length_normalized < 1.0 + f32::EPSILON)
    }

    #[test]
    fn try_l2_normalize_zero(){
        let sample_vec:Array1<f32> = array![0.0, 0.0];
        let normalized_sample_vec = l2_normalize(sample_vec);
        let length_normalized = normalized_sample_vec.dot(&normalized_sample_vec);
        assert!(!length_normalized.is_nan())
    }

    #[test]
    fn try_mock_embedder_text(){
        let mock_embedder: MockEmbedder = MockEmbedder { dim: 512 };
        let text_embedding = mock_embedder.embed_text("Kot ma ale.").unwrap();
        let length_normalized = text_embedding.dot(&text_embedding);
        assert!(length_normalized > 1.0 - f32::EPSILON && length_normalized < 1.0 + f32::EPSILON)
    }

    #[test]
    fn try_mock_embedder_audio(){
        let mock_embedder: MockEmbedder = MockEmbedder { dim: 512 };
        let audio_embedding = mock_embedder.embed_audio(&[1, 2]).unwrap();
        let length_normalized = audio_embedding.dot(&audio_embedding);
        assert!(length_normalized > 1.0 - f32::EPSILON && length_normalized < 1.0 + f32::EPSILON)
    }
}