# AGENTS.md — voice-studio-kaggle

Đích: **Voice Studio AI** trên Kaggle GPU — notebook cộng đồng "copy&edit = chạy" để *tạo
và sở hữu giọng nói* (Design · Clone · Bank · Script · Export), không chỉ đọc văn bản.

## Nguyên tắc

- **Đơn giản, thực dụng tối đa.** Thêm tính năng = thêm module, không phá cấu trúc.
- **Đúng mục đích:** mọi thay đổi phải phục vụ 1 trong 5 trụ (Design/Clone/Bank/Script/Export).
  Tính năng ngoài phạm vi → ghi vào Roadmap, đừng làm ngay.
- **Voice Bank là TRỤ CỘT** (`studio/voicebank.py`): mọi giọng đều lưu được, export/import
  `.zip` vượt session Kaggle. Không bao giờ để giọng "mất theo phiên".
- **Engine-agnostic:** `render.py` / `voicebank.py` / `app.py` không được phụ thuộc API riêng
  của engine. Chỉ `engines.py` được biết VieNeu/VoxCPM2.
- **Hot-swap:** chỉ 1 engine trong VRAM mỗi lúc (`engines.get_engine`). Thêm engine = đăng ký
  vào `REGISTRY` + kế thừa `BaseEngine` (xem "Cách thêm engine").

## Workflow sửa code

1. Sửa `studio/*.py` (source gốc — **không sửa trực tiếp `.ipynb`**).
2. `python -m tests.smoke` — phải ALL PASS.
3. `python tools/pack_notebook.py` — **bắt buộc** repack (notebook nhúng source bằng base64).
4. `python tools/verify_embed.py && python tools/simulate_notebook.py` — xác minh notebook tự chứa.
5. Copy `notebook/Voice_Studio_AI_Kaggle_GPU.ipynb` lên Kaggle.

> **Quên repack = notebook Kaggle vẫn chạy code cũ.** Đây là lỗi lặp lại nhiều nhất.

## Cách thêm engine mới

```python
class FooEngine(BaseEngine):
    name = "foo"; label = "..."; license = "..."; source = "..."
    def load(self): ...                  # nap model
    def unload(self): self._model=None; super().unload()
    def tts(self, text, *, desc="", ref_path="", ref_text="", preset="", style=""):
        ...  # tra ve (wav_float32_mono, sample_rate)

REGISTRY["foo"] = FooEngine
```
Chỉ cần chuẩn hóa `tts(...)` về `(wav, sr)`. Không sửa `render.py`/`app.py`.
Thêm package cài đặt vào cell install trong `pack_notebook.py`.

## Bất biến kiến trúc (không được phá)

- `CFG` (`studio/config.py`) là **nguồn sự thật duy nhất** cho đường dẫn/tham số sinh âm.
- `VoiceProfile.kind ∈ {preset, clone, design}` — dispatch trong `app.synth_profile` phải bao
  đủ 3 loại. Thêm kind = sửa cả `synth_profile` + UI.
- Đầu ra `tts()` luôn **float32 mono**. `audio.normalize/to_mono` chuẩn hóa trước khi trả.
- Notebook phải **tự chứa**: không `git clone`, không API key cho luồng cơ bản.

## Ma trận rủi ro

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Notebook chạy code cũ sau khi sửa | Quên `pack_notebook.py` | Repack + `verify_embed.py` |
| OOM VRAM khi nạp model | Cài/nạp 2 engine cùng lúc | Giữ hot-swap; VoxCPM2 ~8GB → dùng T4 x2 |
| `pip` xung đột dependency | Cài cả vieneu lẫn voxcpm 1 env | Install cell chỉ cài engine đang `ENGINE` |
| Giọng clone bị robot/nhiễu | Clip bẩn, có nhạc, quá ngắn | Clip 5–15s, sát mic, phòng yên, không nhạc |
| Import lỗi `studio` khi chạy trên Kaggle | Sai `sys.path` / thiếu cell giải nén | Kiểm tra cell 3 (decode base64 → PKG) |
| Không có tiếng Việt chuẩn | Đang dùng VoxCPM2/đa ngôn ngữ | Đổi `ENGINE = "vieneu"` |
| `flash-attn` cài lỗi (VoxCPM2) | Thiếu toolchain CUDA | Bỏ flash-attn (chậm hơn nhưng chạy) |

## Giới hạn đã biết (MVP)

| Có | Chưa có |
|---|---|
| Design + Clone + Style (VoxCPM2) | Chatterbox / Qwen3-TTS-VD (roadmap) |
| Giọng Việt 3 miền + emotion cues (VieNeu) | Streaming thời gian thực |
| Voice Bank export/import `.zip` | Auto-lưu Kaggle Dataset |
| Script nhiều giọng → 1 file | Tự ASR transcript clip clone |
| Gradio 5 tab + share link | Chấm điểm độ giống (ECAPA) |

## Test là bắt buộc

Mọi PR/thay đổi `studio/` phải qua: `python -m tests.smoke` (ALL PASS) + repack + verify.
Engine thật (VieNeu/VoxCPM2) chỉ test được trên Kaggle GPU — phần lõi phải pass ở local.
