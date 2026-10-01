# voice-studio-kaggle

**Voice Studio AI** — notebook Kaggle (GPU) biến GPU miễn phí thành một **studio tạo giọng nói**:
thiết kế giọng từ mô tả, nhân bản giọng từ clip, kho giọng tái sử dụng, dựng podcast/audiobook
nhiều giọng, xuất WAV/MP3. Cộng đồng bấm **Copy & Edit → Run All** là chạy.

> **Tác giả / Author: Elainabaka** — Thiết kế & dựng notebook Voice Studio AI cho cộng đồng.
> © 2026 Elainabaka. Giấy phép MIT (xem mục Credit).

---

## 📖 Hướng dẫn sử dụng

### 🖥️ Cách 1 · Chạy Local (khuyến nghị — nhanh, UI đẹp, hết độ trễ)
1. Clone repo → mở thư mục → **double-click `run.bat`** (Windows) hoặc `./run.sh` (Mac/Linux).
2. Lần đầu tự tạo venv + cài thư viện + tải model (một lần). Các lần sau mở là chạy.
3. Giao diện mở ở `http://127.0.0.1:7860`, tự bật trình duyệt. Giọng lưu **lâu dài** ở `~/.voice_studio`.
4. Muốn chạy tay: `python run.py` (tự venv + cài) hoặc `python -m studio`.

### 🌐 Cách 2 · Chạy trên Kaggle (không cần cài gì)
1. Tải `notebook/Voice_Studio_AI_Kaggle_GPU.ipynb` → vào Kaggle → **New Notebook → Import**.
2. Bật **Settings → Accelerator = GPU (T4 x2)** và **Internet = ON** (bắt buộc để tải model).
3. Bấm **Run All** → chờ cài thư viện + tải model (lần đầu vài phút).
4. Giao diện **tự mở** ngay trong cell cuối (nhúng sẵn, không cần bấm link). Đổi ngôn ngữ **🇻🇳/🇺🇸** ở góc trên.

### B. Dùng các workspace trong Studio
| Tab | Làm gì |
|---|---|
| 🎨 **Thiết kế giọng** | Chọn **giới tính / tuổi / vùng miền / chất giọng / cao độ / nhịp / cảm xúc** → tạo giọng **mới hoàn toàn** (chạy bằng VoxCPM2). |
| 🎙️ **Nhân bản giọng** | Tải clip 5–15 giây + nội dung → nhân bản giọng, chỉnh phong cách. |
| 🖼️ **Thư viện giọng** | Nghe thử giọng mẫu + giọng của bạn, lọc theo loại. **Xuất/Nhập `.zip`** để giữ giọng. |
| 🎧 **Kịch bản** | Dán kịch bản (mỗi dòng `Tên giọng: câu nói`) → xuất podcast/audiobook nhiều giọng. |
| 📦 **Xuất file** | Tải WAV/MP3, hoặc nén toàn bộ ra `.zip`. |
| 📖 **Hướng dẫn** | Bảng hướng dẫn chi tiết ngay trong giao diện. |

### C. Giữ giọng qua session (quan trọng)
Kaggle **xóa file khi hết phiên**. Muốn giữ giọng:
1. Tab 🖼️ **Thư viện giọng** → **Xuất kho giọng → `.zip`** → tải về máy.
2. Lần chạy sau (Run All) → tab 🖼️ **Thư viện giọng** → **Nhập `.zip` vào kho**.

### 🖥️☁️ Chọn Local hay Kaggle? (tab ☁️ Chạy trên Kaggle)
| | **Local** | **Kaggle (offload)** |
|---|---|---|
| Dùng khi | Máy có GPU/CPU, muốn **tương tác** (gõ → nghe liền) | Máy **không có GPU**, hoặc render **audiobook/kịch bản nặng** |
| Độ trễ | Rất thấp (chạy tại chỗ) | Cao (batch: đẩy lên → chờ → tải về) |
| Cách chạy | `run.bat` → gõ chữ nghe liền | Tab ☁️ → đóng gói job → đẩy lên T4 → chờ → tải WAV |

**Offload lên Kaggle** (tab ☁️, chỉ hiện khi chạy local): nhập *kernel slug* + *kịch bản* →
**Đóng gói & đẩy lên GPU** → **Xem trạng thái** → **Tải kết quả về**. Dùng `kaggle kernels push/status/output`.
⚠️ Đây là **batch** — không phải GPU từ xa tương tác. Muốn nghe liền thì dùng Local.

### D. Mẹo chất lượng
- **Clone:** clip 5–15s, sát mic, phòng yên, **không nhạc**, một người, đọc đúng tông bạn muốn.
- **Design:** mô tả càng cụ thể càng tốt (thanh thuộc tính giúp bạn ghép mô tả chuẩn).
- **Giọng Việt chuẩn** → dùng engine `vieneu`. **Design + 30 ngôn ngữ** → dùng `voxcpm`.
- Đổi engine ở cell cấu hình: `ENGINE = "vieneu"` hoặc `"voxcpm"`.

---

## 🗺️ Mind map

```
Voice Studio AI (Kaggle GPU · T4 x2)  —  giao diện tự mở · đổi ngôn ngữ 🇻🇳/🇺🇸
│
├── 🎨 Thiết kế giọng   chọn giới tính/tuổi/accent/cao độ/nhịp/cảm xúc/chất giọng → giọng MỚI
│                       └─ engine: VoxCPM2 (Voice Design)
├── 🎙️ Nhân bản giọng   clip 5–15s → nhân bản + điều khiển phong cách/cảm xúc
│                       └─ engine: VieNeu · VoxCPM2 (controllable/ultimate cloning)
├── 🖼️ Thư viện giọng    giọng mẫu + giọng của bạn · EXPORT-IMPORT .zip  ← TRỤ CỘT
│                       └─ studio/voicebank.py  (vuot qua session Kaggle tam thoi)
├── 🎧 Kịch bản          kịch bản nhiều giọng → podcast / audiobook
│                       └─ studio/render.py  (engine-agnostic)
└── 📦 Xuất file         WAV/MP3 + nén zip  ·  📖 Hướng dẫn (tab riêng)
                        └─ studio/export.py
```

## 💡 Ít người nghĩ tới, nhưng thiếu là hỏng

1. **Kaggle session là tạm thời.** Hết phiên = mất file. Studio cho phép thiết kế/clone giọng
   mà **không lưu được** thì vô dụng với người dùng thật. → `voicebank` export/import `.zip`
   là tính năng "tương lai không thể thiếu" mà chưa notebook TTS nào trên Kaggle làm.
2. **Voice Design là lối thoát đạo đức.** Clone giọng người thật vướng consent; *design* giọng
   tổng hợp từ mô tả thì không. Đây là hướng mọi model đầu bảng (VoxCPM2, Qwen3-VD, Maya1)
   đang chạy.

## 🧱 Kiến trúc (module hóa — hỏng đâu sửa đó, lắp ráp thêm được)

```
voice-studio-kaggle/
├── README.md · AGENTS.md              tài liệu + quy tắc cho AI
├── notebook/
│   └── Voice_Studio_AI_Kaggle_GPU.ipynb   ★ DELIVERABLE (copy&edit = chạy)
├── studio/                            source gốc (bảo trì + test ở đây)
│   ├── config.py    CFG + ENGINE_INFO + ensure_dirs
│   ├── audio.py     đọc/ghi, resample, mono, trim, chuẩn hóa
│   ├── voicebank.py VoiceProfile + bank save/load/export/import   ← TRỤ CỘT
│   ├── render.py    script nhiều giọng → audio (engine-agnostic)
│   ├── export.py    WAV/MP3/zip
│   ├── engines.py   BaseEngine + registry + VieNeu + VoxCPM2 (hot-swap)
│   └── app.py       Gradio Studio 5 tab + điều phối cấp cao
├── tests/smoke.py   kiểm thử phần lõi (không cần GPU)
└── tools/
    ├── pack_notebook.py    nhúng studio/ vào .ipynb (base64) — CHẠY LẠI KHI SỬA CODE
    ├── verify_embed.py     xác minh base64 khớp source
    └── simulate_notebook.py mô phỏng cell thư viện + chạy smoke test
```

**Workflow bảo trì:** sửa `studio/*.py` → `python -m tests.smoke` → `python tools/pack_notebook.py`
→ copy `notebook/*.ipynb` lên Kaggle. *Quên repack = notebook Kaggle vẫn chạy code cũ.*

**Ngân sách VRAM:** VieNeu ~1 GB · VoxCPM2 ~8 GB (fit T4 8 GB, chật — nên để T4 x2).
Chỉ **một** engine được nạp vào VRAM mỗi lúc (hot-swap) để tránh OOM.

## 🧪 Kiểm thử (cho người phát triển)

```bash
python -m tests.smoke              # phần lõi: audio, voicebank, render, export
python tools/verify_embed.py       # base64 trong notebook khớp source
python tools/simulate_notebook.py  # notebook tự chứa + import + chạy được
```

## ⚠️ Đạo đức & consent

Chỉ clone giọng khi bạn có quyền. **Voice Design** (giọng tổng hợp từ mô tả) là lựa chọn an toàn
về pháp lý/đạo đức. Không dùng để mạo danh, lừa đảo, tạo nội dung sai sự thật.

## 📄 Credit & License

**Voice Studio AI** được thiết kế và dựng bởi **Elainabaka** © 2026.
Code trong repo này: **MIT** — dùng được cho mục đích cá nhân lẫn thương mại, giữ nguyên credit.

Dựng trên các model mã nguồn mở của tác giả gốc (**cảm ơn và giữ nguyên credit của họ**):

| Thành phần | Tác giả gốc | License |
|---|---|---|
| VieNeu-TTS (v3-Turbo) | Phạm Nguyễn Ngọc Bảo & cộng sự — [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) | Apache-2.0 |
| VoxCPM2 | OpenBMB — [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | Apache-2.0 |
| Gradio | Gradio | Apache-2.0 |

Model giữ license riêng của tác giả gốc (đều Apache-2.0, dùng thương mại được).

## 🗺️ Roadmap

- [ ] Engine **Chatterbox Multilingual** (MIT, paralinguistic tags) — registry đã sẵn sàng.
- [ ] Engine **Qwen3-TTS-VoiceDesign** (voice design SOTA InstructTTSEval).
- [ ] (Tùy chọn) VieNeu **v3 Nano** làm fallback siêu nhẹ cho máy yếu.
- [ ] Tự ASR transcript clip clone (Whisper) để tăng độ giống.
- [ ] Chấm điểm độ giống giọng (ECAPA cosine) sau mỗi lần clone.
- [ ] (Tùy chọn) lưu bank thẳng lên Kaggle Dataset qua `kaggle` API.
