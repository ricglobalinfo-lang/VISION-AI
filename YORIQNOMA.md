# Lokal AI kamera tizimi — yo‘riqnoma (0 dan ishga tushirish va tekshirish)

Bu hujjat tizimni **bo‘sh kompyuterdan** ishga tushirish, sozlash va to‘g‘ri ishlayotganini tekshirish uchun.

- Loyiha papkasi: `camera-kurbonov`
- UI: `http://127.0.0.1:8080/ai.html`
- Asosiy skript: `start-ai.bat`
- Sozlamalar: `ai/config.py`

---

## 1. Nima kerak (talablar)

### 1.1. Apparat

| Komponent | Tavsiya | Izoh |
|-----------|---------|------|
| CPU | 6+ yadro | FFmpeg + JPEG encode |
| RAM | 16 GB+ (32 GB yaxshi) | 4 kamera + AI |
| GPU | NVIDIA (CUDA) | RTX 5070 / shunga o‘xshash |
| VRAM | 8 GB+ (12 GB ideal) | `yolov8x` + pose + InsightFace |
| Tarmoq | Kameralar bilan bir LAN | RTSP `192.168.1.x` |

GPU bo‘lmasa tizim CPU’da ishlashi mumkin, lekin sekin va sifat pastroq.

### 1.2. Dasturlar

1. **Windows 10/11**
2. **Python 3.12** ([python.org](https://www.python.org/) yoki Microsoft Store)
   - O‘rnatishda **“Add python.exe to PATH”** ni belgilang
3. **NVIDIA Driver** (yangi Game Ready / Studio)
4. **Git** (ixtiyoriy; loyihani nusxa olish uchun)
5. Brauzer: Chrome / Edge

Tekshirish (PowerShell):

```powershell
py -3.12 --version
nvidia-smi
```

`nvidia-smi` GPU nomini va driver versiyasini ko‘rsatishi kerak.

### 1.3. Kameralar

- 4 ta IP kamera (masalan config: `192.168.1.101`, `.102`, `.103`, `.104`)
- RTSP yoqilgan
- Kompyuterdan ping o‘tishi kerak:

```powershell
ping 192.168.1.101
ping 192.168.1.102
ping 192.168.1.103
ping 192.168.1.104
```

---

## 2. Loyihani tayyorlash

### 2.1. Papkani ochish

Loyiha shu yo‘lda bo‘lishi kerak (yoki o‘zingizning yo‘lingiz):

```text
C:\Projects\camera-kurbonov\
  ai\
  www\
  faces\
  data\
  tools\
  start-ai.bat
```

### 2.2. FFmpeg

`tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe` mavjudligini tekshiring.

Agar yo‘q bo‘lsa:

1. FFmpeg Windows build yuklab oling (GPL shared)
2. `tools\` ichiga shu strukturada joylashtiring
3. Yoki tizim PATH’iga `ffmpeg` qo‘shing (`ai/config.py` dagi `FFMPEG` yo‘lini moslang)

Tekshiruv:

```powershell
& ".\tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe" -version
```

---

## 3. Python muhiti (venv) — 0 dan

PowerShell’da loyiha ildiziga o‘ting:

```powershell
cd C:\Projects\camera-kurbonov
```

### 3.1. Virtual environment

```powershell
py -3.12 -m venv ai\.venv
ai\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3.2. PyTorch (CUDA) — majburiy (GPU uchun)

GPU uchun PyTorch’ni **alohida** o‘rnating (rasmiy CUDA 12.8 wheel):

```powershell
ai\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Tekshiruv:

```powershell
ai\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kutilgan: `True` va GPU nomi (masalan `NVIDIA GeForce RTX 5070`).

### 3.3. Qolgan paketlar

```powershell
ai\.venv\Scripts\python.exe -m pip install -r ai\requirements.txt
```

**Muhim:** `onnxruntime-gpu` versiyasi `>=1.16,<1.23` bo‘lishi kerak (1.28 CUDA 13 DLL talab qiladi va InsightFace CPU’ga tushadi).

Tekshiruv:

```powershell
ai\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Ro‘yxatda `CUDAExecutionProvider` bo‘lishi kerak.

### 3.4. (Ixtiyoriy) InsightFace CUDA DLL yo‘li

Worker ishga tushganda torch `lib` papkasini PATH’ga qo‘shadi. Agar yuz hali CPU’da qolsa, worker logida `Applied providers: ['CPUExecutionProvider']` ko‘rinadi — o‘shanda `onnxruntime-gpu` versiyasi va torch CUDA o‘rnatilganini qayta tekshiring.

---

## 4. Sozlamalar (`ai/config.py`)

### 4.1. Kameralar

`CAMERAS` ro‘yxatida har bir kamera uchun:

- `id`, `name`, `ip`
- `rtsp_sub` — yengil oqim (grid / orqa kameralar)
- `rtsp_main` — sifatli oqim (focus kamera)
- `brand`: `hikvision` yoki `oem`

Parol/login o‘zgarganda shu URL’larni yangilang.

### 4.2. Web login

Standart (o‘zgartirish tavsiya etiladi):

- Foydalanuvchi: `admin`
- Parol: `CHANGE_ME_PASSWORD` (`AUTH_PASSWORD`)

Prod’da muhit o‘zgaruvchisi orqali berish mumkin:

```powershell
$env:AUTH_PASSWORD = "YangiParol"
```

### 4.3. AI (hozirgi ishchi sozlamalar)

| Sozlama | Qiymat | Ma’nosi |
|---------|--------|---------|
| `YOLO_MODEL` | `yolov8x.pt` | Asosiy detektor |
| `YOLO_IMGSZ` | 960 | Aniqlik |
| `FIGHT_POSE_MODEL` | `yolov8l-pose.pt` | Urush / axlat pose |
| `FACE_DET_SIZE` | (1600, 1600) | Yuz aniqlash |
| `AI_BG_EVERY_N` | 3 | Orqa kamera AI har 3-tsikl |
| `LITTER_*` | yoqilgan | Axlat A+B |

Birinchi ishga tushirishda Ultralytics kerakli `.pt` fayllarni yuklab olishi mumkin (internet kerak). Loyihada ko‘p vaznlar allaqachon `ai\` ichida bo‘lishi mumkin.

---

## 5. Yuz bazasi (`faces\`)

### 5.1. Strukturа

```text
faces\
  Familiya Ism\
    foto1.jpg
    foto2.jpg
```

- Papka nomi = ekrandagi FIO
- Har odamga 2–5 ta aniq yuz fotosining bo‘lishi yaxshi
- Xira / juda kichik yuzlar kamroq foydali

### 5.2. Enrollment

`start-ai.bat` avtomatik `enroll_faces.py` ni chaqiradi.

Qo‘lda:

```powershell
ai\.venv\Scripts\python.exe ai\enroll_faces.py
```

Natija: `data\face_db.npz`

UI orqali ham: **Odam qo‘shish** (FIO + rasmlar).

---

## 6. Tizimni ishga tushirish

### 6.1. Oddiy usul (tavsiya)

1. Explorer’da loyiha papkasini oching
2. **`start-ai.bat`** ni ikki marta bosing
3. Kuting:
   - eski `worker.py` / `ffmpeg` to‘xtatiladi
   - yuz bazasi yangilanadi
   - brauzer ochiladi
   - konsolda YOLO / Face / Fight / Litter yuklanishi chiqadi
4. Login qiling (`admin` / sozlangan parol)

### 6.2. Qo‘lda (PowerShell)

```powershell
cd C:\Projects\camera-kurbonov

# Eski jarayonlarni to‘xtatish
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -match 'worker\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

ai\.venv\Scripts\python.exe ai\enroll_faces.py
ai\.venv\Scripts\python.exe ai\worker.py
```

Brauzer: `http://127.0.0.1:8080/ai.html`

### 6.3. Birinchi boot logida ko‘rinishi kerak bo‘lgan qatorlar

```text
CUDA: avail=True device=NVIDIA GeForce RTX ...
YOLO warmup OK: yolov8x.pt
Fight pose models: 4 cams OK
Litter detector ON | ... floor=True@0.68 pose=True
Applied providers: ['CUDAExecutionProvider', ...]   # yuz uchun
Face: providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] ...
AI loop: full_all=True batch_yolo=True ... bg_every_n=3 ...
AI UI: http://127.0.0.1:8080/ai.html
Serving with waitress ...
```

Agar `CUDA: avail=False` bo‘lsa — PyTorch CUDA o‘rnatilmagan.  
Agar Face faqat `CPUExecutionProvider` — onnxruntime-gpu / DLL muammosi.

---

## 7. Tekshirish (checklist)

Har bir bandni belgilab chiqing.

### 7.1. Servis va login

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 1 | `http://127.0.0.1:8080/login` ochiladi | Login sahifa |
| 2 | Noto‘g‘ri parol | Xato xabari |
| 3 | To‘g‘ri login | `ai.html` ochiladi |
| 4 | Login’siz `/ai/cameras` | 401 |

### 7.2. Kameralar / efir

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 5 | 4 ta kamera ko‘rinadi | Kamera 1/2/3/4 |
| 6 | Har birida harakat / soat yangilanadi | Muzlab qolmagan |
| 7 | Fokus bitta kameraga | Rasm sifatliroq (main) |
| 8 | **Hammasi** (grid) | 2×2, yengilroq oqim |
| 9 | HUD: FPS | Taxminan 20–30 |
| 10 | HUD: lag | Odatda ~30–150 ms (focus’da ba’zan yuqori) |

API orqali (PowerShell, login session kerak bo‘ladi) yoki UI’dagi status.

Tezkor snapshot:

```text
http://127.0.0.1:8080/ai/snapshot/cam1.jpg
```

(login qilingan brauzer sessiyasida)

### 7.3. YOLO (obyekt / odam)

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 11 | Odam kadrga kirsа | “odam” ramkasi |
| 12 | Stul / noutbuk va hokazo | O‘zbekcha yorliq (agar aniqlansa) |
| 13 | Chalkash (ryukzak/stul) | Juda ko‘p false bo‘lmasligi |

### 7.4. Yuz tanish

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 14 | Bazadagi odam | Yashil ism |
| 15 | Noma’lum odam | Qizil “Noma’lum” + event |
| 16 | `faces\` ga yangi odam qo‘shib restart / UI enroll | Yangi ism chiqadi |
| 17 | Begona bo‘lim | `data\unknown\` da suratlar (sifat yetarli bo‘lsa) |

### 7.5. Urush

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 18 | Ikki odam yonma-yon turadi | **URUSH chiqmasligi** kerak |
| 19 | Haqiqiy agressiv harakat (nazoratli test) | `URUSH` + event + `data\fight\` |

### 7.6. Axlat

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 20 | Stolda tinch turgan stakan | AXLAT bo‘lmasligi (yoki kam) |
| 21 | Odam shisha/stakanni **yerga** tashlaydi | `AXLAT` + event + `data\litter\` |

### 7.7. GPU

```powershell
nvidia-smi
```

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 22 | AI ishlayotganda | GPU util odatda ~50–85% |
| 23 | VRAM | Taxminan 2.5–4+ GB band |
| 24 | `nvidia-smi` da python jarayoni | Ko‘rinadi |

### 7.8. Barqarorlik

| # | Tekshiruv | Kutilgan natija |
|---|-----------|-----------------|
| 25 | 10–15 daqiqa ishlash | Crash yo‘q, kameralar oqadi |
| 26 | Fokusni bir necha marta almashtirish | Stream main/sub almashadi |
| 27 | Brauzer tabini yopib ochish | Qayta ulanadi, “eski kadr” qotib qolmaydi |

---

## 8. Kundalik ishlatish

1. `start-ai.bat`
2. Login
3. Kerakli kamerani fokus qiling
4. Event log’ni kuzating
5. Yangi odam: UI **Odam qo‘shish** yoki `faces\` + restart/enroll
6. To‘xtatish: konsolda `Ctrl+C` yoki oynani yopish (`start-ai.bat` dagi `pause`)

**Eslatma:** bir vaqtda **bitta** `worker.py` ishlasin. Ikkinchi nusxa port 8080 ni band qiladi yoki chalkashtiradi.

---

## 9. Muammolarni bartaraf etish

| Muammo | Sabab / yechim |
|--------|----------------|
| `venv topilmadi` | 3-bo‘lim bo‘yicha venv yarating |
| Port 8080 band | Eski `worker.py` ni o‘ldiring (`start-ai.bat` buni qiladi) |
| Qora ekran / kadr yo‘q | Ping, RTSP URL, kamera paroli, firewall |
| OEM kamera ochilmaydi | FFmpeg yo‘li; HEVC uchun FFmpeg grabber |
| `CUDA: avail=False` | PyTorch cu128 qayta o‘rnating |
| Yuz juda sekin / CPU | `onnxruntime-gpu<1.23`; torch lib; logda CUDA provider |
| FPS past, lag yuqori | Focus main og‘ir; sub’ga o‘ting yoki `AI_BG_EVERY_N` oshiring |
| LM Studio / boshqa AI GPU’ni to‘ldirgan | VRAM bo‘shating |
| Model yuklanmayapti | Internet; `.pt` fayllar `ai\` da |
| Login ishlamaydi | `AUTH_USERNAME` / `AUTH_PASSWORD` |

Jarayonlarni tozalash:

```powershell
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -match 'worker\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 10. Muhim fayllar (qisqa xarita)

| Yo‘l | Vazifa |
|------|--------|
| `start-ai.bat` | Ishga tushirish |
| `ai/worker.py` | Asosiy server + AI |
| `ai/config.py` | Kamera, AI, login |
| `ai/enroll_faces.py` | Yuz bazasini yig‘ish |
| `ai/litter_detect.py` | Axlat |
| `ai/fight_detect.py` | Urush |
| `www/ai.html` | Operator UI |
| `www/login.html` | Login |
| `faces\` | Tanish odamlar |
| `data\face_db.npz` | Embedding baza |
| `data\events.jsonl` | Hodisalar |
| `data\unknown\` | Begona yuzlar |
| `data\fight\` | Urush snapshot |
| `data\litter\` | Axlat snapshot |

---

## 11. Xavfsizlik eslatmalari

1. Taqdimot / Git’ga **kamera parollarini** ochiq joylamang
2. `AUTH_PASSWORD` va `AUTH_SECRET` ni o‘zgartiring
3. Tizim lokal tarmoq uchun mo‘ljallangan; internetga ochishda tunnel + kuchli parol kerak
4. Video asosan shu kompyuterda qayta ishlanadi

---

## 12. Tezkor “hammasi tayyormi?” (1 daqiqa)

```powershell
cd C:\Projects\camera-kurbonov
ai\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print('GPU OK', torch.cuda.get_device_name(0))"
ping -n 1 192.168.1.101
ping -n 1 192.168.1.102
ping -n 1 192.168.1.103
ping -n 1 192.168.1.104
Test-Path .\ai\.venv\Scripts\python.exe
Test-Path .\tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe
Test-Path .\www\ai.html
```

Keyin: `start-ai.bat` → login → 4 kamera oqimi → odam/yuz ko‘rinishi.

---

**Tayyor.** Savol chiqsa: avval konsol logini (CUDA / Face providers / kamera error) va `nvidia-smi` ni tekshiring.
