# Multi-Camera AI Surveillance System (Lokal AI Kamera Tizimi)

Ko'p kamerali (Multi-Camera) aqlli video tahlil va xavfsizlik monitoring tizimi. Tizim barcha videolarni to'liq lokal kompyuterda (GPU/CUDA yordamida) qayta ishlaydi.

---

## Asosiy Imkoniyatlar (Features)

- **Ko'p kamerali monitoring (Multi-Camera RTSP):** Bir vaqtning o'zida bir nechta IP kameralarni (Hikvision, OEM va boshqalar) 2x2 grid yoki fokus rejimida ko'rish.
- **Yuz tanish va Ro'yxatdan o'tkazish (Face Recognition & Enrollment):**
  - InsightFace algoritmi orqali yuqori aniqlikda yuzni tanish.
  - Tanish odamlarga yashil ramka, noma'lumlarga qizil ramka va ogohlantirish (beep / log).
  - Web UI orqali yangi shaxs va rasmlarini qo'shish.
- **Anti-Spoofing & Jonlilik (Liveness Detection):**
  - Rasmdan yoki telefondan ko'rsatilgan qalbaki yuzlarni (photo/screen spoofing) aniqlash.
  - Mikro-harakatlar orqali tiriklikni tekshirish.
- **Obyekt va Odamlarni aniqlash (YOLOv8 Detection):**
  - 80 xil COCO obyektlarini aniqlash (odam, noutbuk, telefon, stul va boshqalar).
- **Urush / Bezozorlikni aniqlash (Fight Detection):**
  - YOLO Pose va ST-GCN / Heuristic orqali odamlar orasidagi tajovuzkor harakatlarni avtomatik aniqlash.
- **Axlat tashlashni aniqlash (Littering Detection):**
  - Butilka, stakan, idish kabi narsalarni yerga tashlash holatlarini kuzatish.
- **Web Boshqaruv Paneli:**
  - Real-vaqtli video oqim (HLS / MJPEG).
  - Voqealar jurnali (Events log) va noma'lum shaxslar galereyasi.
  - Login / Parol bilan himoyalangan avtorizatsiya.

---

## Loyiha Strukturasi (Folder Structure)

`	ext
camera-kurbonov/
├── ai/                     # AI moduli (YOLO, InsightFace, Pose, Flask/Waitress worker)
│   ├── config.py           # Kamera va AI parametrlari sozlamalari
│   ├── config.example.py   # Sozlamalar shabloni
│   ├── worker.py           # Asosiy AI server va tahlil mexanizmi
│   ├── enroll_faces.py     # Yuz bazasini (npz) yaratish skripti
│   ├── face_match.py       # Yuz tanish logikasi
│   ├── face_antispoof.py   # Qalbaki yuzlarni aniqlash
│   ├── fight_detect.py     # Urush / janjal detektori
│   ├── litter_detect.py    # Axlat tashlash detektori
│   └── requirements.txt    # Kerakli Python kutubxonalari
├── faces/                  # Tanish odamlarning rasmlari (Papka nomi = FIO)
├── data/                   # Ma'lumotlar bazasi, hodisalar va snapshotlar
├── tools/                  # FFmpeg va yordamchi utilitalar
├── www/                    # Web frontend interfeysi (HTML/JS/CSS)
├── public/                 # PHP orqali masofaviy snapshot ko'rish fayllari
├── start-ai.bat            # Loyihani bir bosishda ishga tushirish skripti
├── YORIQNOMA.md            # To'liq o'rnatish va sozlash qo'llanmasi
└── README.md
`

---

## O'rnatish va Ishga Tushirish (Quick Start)

### 1. Talablar
- **OS:** Windows 10 / 11 (64-bit)
- **Python:** 3.12 (PATH ga qo'shilgan)
- **GPU:** NVIDIA (CUDA qo'llab-quvvatlaydigan, masalan RTX 3060/4060/5070 yoki undan yuqori)
- **FFmpeg:** 	ools\ papkasida yoki tizim PATH'ida

### 2. Python Virtual Muhitini Sozlash

Loyiha papkasida PowerShell orqali:

`powershell
# 1. Virtual muhit yaratish
py -3.12 -m venv ai\.venv

# 2. PyTorch CUDA o'rnatish (GPU uchun)
ai\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Qolgan paketlarni o'rnatish
ai\.venv\Scripts\python.exe -m pip install -r ai\requirements.txt
`

### 3. Sozlamalarni kiritish
i/config.py faylini oching va kameralaringizning RTSP havolalari hamda IP manzillarini kiriting.

### 4. Ishga tushirish
Shunchaki **start-ai.bat** faylini ikki marta bosing yoki buyruqlar satrida ishga tushiring:

`powershell
.\start-ai.bat
`

Brauzerda oching:
👉 **http://127.0.0.1:8080/ai.html**

Standart login ma'lumotlari:
- **Login:** `admin`
- **Parol:** `CHANGE_ME_PASSWORD` *(sozlamalardan o'zgartirish mumkin)*

---

## Qo'shimcha Hujjatlar
- Batafsil ma'lumot va tekshiruvlar ro'yxati: [YORIQNOMA.md](YORIQNOMA.md)
