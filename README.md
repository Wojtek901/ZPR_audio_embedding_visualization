# Audio Embedding Explorer (ZPR – Temat nr 11)

Interaktywne narzędzie do multimodalnej analizy podobieństwa tekst–audio oraz wizualizacji przestrzeni embeddingów w 2D, napisane w języku Rust i inspirowane projektem [MTG/music-explore](https://github.com/MTG/music-explore).

Aplikacja działa w architekturze klient–serwer: wielowątkowy backend w Ruście zarządza modelem sieci neuronowej, buforem wektorów i redukcją wymiarowości, natomiast lekki frontend webowy w przeglądarce pozwala na żywo eksplorować wykres, wpisywać zapytania tekstowe, wgrywać własne dane oraz odsłuchiwać wybrane utwory.

---

## Główne funkcjonalności

- **Multimodalne wyszukiwanie (tekst - audio):** Obliczanie podobieństwa kosinusowego w czasie rzeczywistym pomiędzy frazą tekstową w języku naturalnym a zbiorem plików audio przy użyciu modelu **LAION-CLAP** (`clap-htsat-unfused` / `larger_clap_music_and_speech`) uruchamianego przez ONNX Runtime.
- **Redukcja wymiarowości 2D:** Interaktywne rzutowanie przestrzeni 512-wymiarowej na płaszczyznę za pomocą algorytmów **UMAP** oraz **t-SNE** z możliwością zmiany metody i parametrów z poziomu interfejsu.
- **Zarządzanie danymi na żywo:** Dodawanie nowych plików dźwiękowych (`.wav`, `.mp3`, `.flac`, `.ogg`), importowanie gotowych, wyliczonych wcześniej embeddingów oraz usuwanie elementów z aktywnej przestrzeni roboczej bez restartu serwera.
- **Interaktywna wizualizacja i odsłuch:** Wykres punktowy oparty na Plotly.js zintegrowany z odtwarzaczem HTML5 `<audio>` odtwarzającym próbkę po kliknięciu w punkt.

---

## Stos technologiczny

- **Backend & API:** Rust, `axum`, `tokio`, `tower-http`, `serde` / `serde_json`, `csv`
- **Wnioskowanie ML:** `ort` (ONNX Runtime), `hf-hub` (automatyczne pobieranie wag z Hugging Face), `tokenizers`, `ndarray`
- **Przetwarzanie audio:** `symphonia`, `rubato`, `realfft`
- **Redukcja wymiarowości:** `manifolds-rs` (UMAP i t-SNE)
- **Frontend:** HTML5, Vanilla JS, Plotly.js

---

## Struktura projektu

Projekt rozdzielony jest na punkt startowy (`main.rs`) oraz modułową bibliotekę (`lib.rs`), co ułatwia równoległą pracę w zespole i testowanie w CI:

```text
ZPR_audio_embedding_visualization/
├── .github/
│   └── workflows/
│       └── ci.yml            # Potok CI (cargo fmt, clippy, test)
├── Cargo.toml                # Zależności i konfiguracja pakietu
├── README.md                 # Dokumentacja projektu
├── static/
│   └── index.html            # Frontend (HTML + JS + Plotly.js + <audio>)
├── src/
│   ├── main.rs               # Start runtime'u Tokio, inicjalizacja stanu i serwera Axum
│   ├── lib.rs                # Główny moduł biblioteczny spinający pozostałe komponenty
│   ├── state.rs              # Współdzielony stan aplikacji (AppState: bufor embeddingów)
│   ├── similarity.rs         # Obliczanie podobieństwa kosinusowego i rankingu
│   ├── projection.rs         # Wrapper na algorytmy redukcji wymiarów (UMAP / t-SNE)
│   ├── embedder.rs           # Interfejs trait EmbeddingModel oraz eksport podmodułów
│   ├── embedder/
│   │   ├── clap.rs           # Obsługa sesji ONNX (LAION-CLAP) i pobierania wag (hf-hub)
│   │   ├── audio.rs          # Dekodowanie audio, resampling i wyliczanie log-mel spektrogramu
│   │   └── text.rs           # Tokenizacja zapytań tekstowych (RoBERTa BPE)
│   ├── api.rs                # Konfiguracja routera Axum i serwowania plików statycznych
│   └── api/
│       ├── routes.rs         # Handlery endpointów REST API
│       └── dto.rs            # Struktury żądań i odpowiedzi JSON (Request / Response)
└── tests/
    └── api_tests.rs          # Testy integracyjne endpointów REST API
```

---

## Przegląd REST API

| Metoda i ścieżka | Opis |
| :--- | :--- |
| `GET /` | Serwuje interfejs webowy (`static/index.html`). |
| `POST /api/audio` | Przyjmuje pliki audio (`multipart/form-data`), wylicza embeddingi w ONNX, zapisuje w `AppState` i zwraca zaktualizowane współrzędne 2D. |
| `POST /api/embeddings` | Importuje lub podmienia gotowe wektory embeddingów (JSON / CSV) z pominięciem sieci ONNX i przelicza wykres. |
| `DELETE /api/embeddings/{id}` | Usuwa wybrany utwór lub zewnętrzny embedding z aktywnej przestrzeni. |
| `POST /api/search` | Przyjmuje zapytanie tekstowe (`JSON`), liczy embedding tekstu i zwraca ranking `similarity score` dla wszystkich elementów. |
| `POST /api/project` | Przelicza współrzędne 2D dla wybranej metody (`umap` / `tsne`) i parametrów bez ponownego uruchamiania modelu AI. |
| `GET /audio/{plik}` | Strumieniuje wskazany plik dźwiękowy do odtwarzacza `<audio>` w przeglądarce. |

---

## Wymagania i uruchomienie

### Wymagania wstępne
- Aktualny toolchain **Rust** (zalecane `rustup update`, min. Rust 1.92 dla `ort` 2.0-rc / edycja 2024).
- Dostęp do internetu przy **pierwszym uruchomieniu** - program automatycznie pobierze pliki modelu ONNX oraz tokenizer z Hugging Face Hub do lokalnego katalogu cache (`~/.cache/huggingface/`).

### 1. Klonowanie repozytorium
```bash
git clone https://github.com/Wojtek901/ZPR_audio_embedding_visualization.git
cd ZPR_audio_embedding_visualization
```

### 2. Uruchomienie serwera

Tryb deweloperski (szybsza kompilacja):
```bash
cargo run
```

Tryb produkcyjny / demonstracyjny:
```bash
cargo run --release
```

Po uruchomieniu należy otworzyć w przeglądarce adres:
```text
http://localhost:3000
```

---
<!--
## Komendy deweloperskie i CI

Przed otwarciem Pull Requesta upewnij się, że przechodzą lokalnie wszystkie kroki weryfikowane przez GitHub Actions:

```bash
# Sprawdzenie formatowania kodu
cargo fmt --all -- --check

# Analiza statyczna (linter)
cargo clippy --all-targets -- -D warnings

# Uruchomienie wszystkich testów jednostkowych i integracyjnych
cargo test

# Uruchomienie testów konkretnego modułu (np. preprocessing audio)
cargo test audio
```

--- -->

## Autorzy

- **Wojciech Fiedoruk**
- **Paweł Spirdyowicz**