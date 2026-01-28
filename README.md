# ⚽ NLP Football Match Analysis Pipeline

Futbol maçı videolarından otomatik analiz yapan bir NLP (Doğal Dil İşleme) boru hattı. Video dosyalarından ses çıkarır, konuşmayı metne dönüştürür ve çeşitli NLP teknikleri ile analiz eder.

## 🎯 Özellikler

- **Ses Çıkarma**: Video dosyalarından ses çıkarma (ffmpeg)
- **Transkripsiyon**: OpenAI Whisper ile ses→metin dönüşümü
- **Metin Ön İşleme**: Tokenizasyon, lemmatizasyon, gürültü temizleme (spaCy)
- **Bilgi Çıkarma**: Oyuncu, takım, stadyum, hakem tespiti (NER)
- **Olay Algılama**: Gol, faul, kart, ofsayt, sakatlık tespiti
- **Duygu Analizi**: Heyecan seviyesi ve atmosfer analizi
- **Ses Analizi**: Volume seviyesi ile heyecan korelasyonu
- **Özet Oluşturma**: BART modeli ile abstractive summarization
- **Görselleştirme**: Heyecan grafikleri, olay zaman çizelgeleri

## 📁 Proje Yapısı

```
nlpfootbal/
├── main.py                    # Ana orkestrasyon
├── extract_audio.py           # Video → WAV
├── transcribe_audio.py        # Ses → Metin (Whisper)
├── pre_process.py             # Metin ön işleme (spaCy)
├── information_extraction.py  # NER + Olay algılama
├── sentiment_analyzer.py      # Duygu/heyecan analizi
├── audio_volume.py            # Ses seviyesi analizi
├── summarization.py           # Maç özeti (BART)
├── insights.py                # Görselleştirme ve raporlama
├── outputs/                   # Analiz çıktıları
└── requirements.txt           # Python bağımlılıkları
```

## 🚀 Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Python paketlerini yükle
pip install -r requirements.txt

# spaCy modelini indir
python -m spacy download en_core_web_sm
```

### 2. Sistem Bağımlılıkları

```bash
# ffmpeg kurulumu (Ubuntu/Debian)
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 🎮 Kullanım

### Tek Maç Analizi

```python
from main import analyze_match

video_path = "maç_videosu.mp4"
output_dir = analyze_match(video_path)
print(f"Sonuçlar: {output_dir}")
```

### Komut Satırından

```bash
python main.py
```

> `main.py` dosyasındaki `video` değişkenini analiz etmek istediğiniz video yolu ile değiştirin.

## 📊 Çıktılar

Her maç için ayrı bir klasör oluşturulur:

| Dosya | Açıklama |
|-------|----------|
| `*_transcript.txt` | Ham transkript |
| `*_preprocessed.txt` | Temizlenmiş metin |
| `*_events.json` | Tespit edilen olaylar |
| `*_summary.txt` | Yapılandırılmış maç özeti |
| `*_excitement_graph.png` | Heyecan grafiği |
| `*_event_timeline.png` | Olay zaman çizelgesi |
| `*_match_insights.json` | Detaylı istatistikler |
| `*_match_events.csv` | CSV formatında olaylar |

## 🏈 Tespit Edilen Olaylar

| Olay | Pattern Örnekleri |
|------|-------------------|
| ⚽ Gol | "scores!", "it's a goal", "1-0" |
| 🟨 Sarı Kart | "yellow card", "booked" |
| 🟥 Kırmızı Kart | "red card", "sent off" |
| 📐 Ofsayt | "offside", "flag is up" |
| 🔄 Değişiklik | "substitution", "brings on" |
| 🩹 Sakatlık | "injury", "stretcher" |
| 🦵 Faul | "foul", "tackled" |

## 🔧 Pipeline Akışı

```
Video (.mp4)
    ↓
[extract_audio] → Audio (.wav)
    ↓
[transcribe_audio] → Transcript + Timestamps
    ↓
[pre_process] → Cleaned Text
    ↓
[information_extraction] → Entities + Events
    ↓
[sentiment_analyzer] + [audio_volume] → Excitement Analysis
    ↓
[summarization] → Match Summary
    ↓
[insights] → Graphs & Reports
```

## 📦 Teknolojiler

- **spaCy** - NER ve dilbilimsel analiz
- **OpenAI Whisper** - Ses tanıma
- **Transformers (BART)** - Özet oluşturma
- **SciPy/NumPy** - Ses analizi
- **Matplotlib** - Görselleştirme

## 📝 Lisans

MIT License
