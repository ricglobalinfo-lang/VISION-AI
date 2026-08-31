"""AI camera configuration — multi-camera, all local."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AI_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FACES_DIR = ROOT / "faces"
WWW_DIR = ROOT / "www"
FFMPEG = ROOT / "tools" / "ffmpeg-master-latest-win64-gpl-shared" / "bin" / "ffmpeg.exe"

FACE_DB_PATH = DATA_DIR / "face_db.npz"
EVENTS_PATH = DATA_DIR / "events.jsonl"
UNKNOWN_DIR = DATA_DIR / "unknown"

# Multi-camera list.
# rtsp / rtsp_sub = yengil (grid + orqa kameralar, past lag)
# rtsp_main = ochiq (focus) kamera uchun tiniq efir
CAMERAS = [
    {
        "id": "cam1",
        "name": "Kamera 1",
        "ip": "192.168.1.101",
        "rtsp": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.101:554/Streaming/Channels/102",
        "rtsp_sub": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.101:554/Streaming/Channels/102",
        "rtsp_main": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.101:554/Streaming/Channels/101",
        "brand": "hikvision",
    },
    {
        "id": "cam2",
        "name": "Kamera 2",
        "ip": "192.168.1.102",
        "rtsp": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.102:554/ch1/sub/av_stream",
        "rtsp_sub": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.102:554/ch1/sub/av_stream",
        "rtsp_main": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.102:554/ch1/main/av_stream",
        "brand": "oem",
    },
    {
        "id": "cam3",
        "name": "Kamera 3",
        "ip": "192.168.1.103",
        "rtsp": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.103:554/ch1/sub/av_stream",
        "rtsp_sub": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.103:554/ch1/sub/av_stream",
        "rtsp_main": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.103:554/ch1/main/av_stream",
        "brand": "oem",
    },
    {
        "id": "cam4",
        "name": "Kamera 4",
        "ip": "192.168.1.104",
        "rtsp": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.104:554/ch1/sub/av_stream",
        "rtsp_sub": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.104:554/ch1/sub/av_stream",
        "rtsp_main": "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.104:554/ch1/main/av_stream",
        "brand": "oem",
    },
]

# Display / encode — focus (single) kamera HQ, qolganlari yengil
DISPLAY_WIDTH_SUB = 960
DISPLAY_WIDTH_MAIN = 1280  # 1440 → 1280 (lag/FPS)
JPEG_QUALITY_SUB = 78
JPEG_QUALITY_MAIN = 86  # 92 → 86 (lag/FPS)
# legacy aliases
DISPLAY_WIDTH = DISPLAY_WIDTH_SUB
JPEG_QUALITY = JPEG_QUALITY_SUB

# AI (GPU — RTX 5070, parallel multi-cam)
AI_WIDTH = 1280
YOLO_MODEL = "yolov8x.pt"  # l → x (ko‘proq GPU / aniqlik)
YOLO_IMGSZ = 960  # 800 → 960
YOLO_CONF = 0.45
YOLO_DEVICE = 0  # CUDA gpu0; CPU uchun "cpu"
YOLO_HALF = True  # FP16 on GPU
AI_FULL_ALL_CAMS = True  # har kamera to‘liq AI (focus emas)
AI_BATCH_YOLO = True  # barcha kameralarni bitta batch da
AI_POST_WORKERS = 4  # face/fight/litter parallel workerlar
AI_PIPELINE = True  # YOLO N+1 || face/pose N (GPU bandligini oshiradi)
# Focus kamera: ikkinchi yuqori aniqlikdagi YOLO pass
AI_FOCUS_REFINE = True
AI_FOCUS_IMGSZ = 1280
# Orqa kameralar: to‘liq AI har N-chi tsikl (1=har doim). Focus — har doim.
# 3 ≈ CPU/efir silliqligi ↑, alert ~100–200ms kechikishi mumkin
AI_BG_EVERY_N = 3
# Empty = detect ALL COCO classes (80 xil buyum)
YOLO_CLASSES = ()
# Ofisda tez-tez chalkashadigan klasslar — yuqoriroq ishonch talab
YOLO_CLASS_MIN_CONF = {
    "person": 0.35,
    "laptop": 0.60,
    "keyboard": 0.55,
    "mouse": 0.55,
    "remote": 0.55,
    "cell phone": 0.55,
    "book": 0.55,
    "tv": 0.55,
    "monitor": 0.55,
    "suitcase": 0.55,
    "handbag": 0.65,
    "backpack": 0.72,
    "chair": 0.35,
    "couch": 0.45,
    "dining table": 0.45,
    "bench": 0.50,
    "bottle": 0.50,
    "cup": 0.50,
    "bowl": 0.50,
}
# Shakl filtri: laptop/tv/book odatda enli; baland quti (stul) rad etiladi
YOLO_WIDE_CLASSES = ("laptop", "keyboard", "tv", "book", "remote")
YOLO_WIDE_MIN_ASPECT = 0.90  # width/height
# Kurtka/stul → ryukzak chalkashligi: baland quti + past ishonch
YOLO_TALL_FALSE_CLASSES = ("backpack", "handbag", "suitcase")
YOLO_TALL_MAX_ASPECT = 0.85  # width/height dan past = juda baland
YOLO_TALL_MIN_CONF = 0.78

# Display names (Uzbek). Missing keys fall back to English COCO name.
YOLO_LABELS_UZ = {
    "person": "odam",
    "bicycle": "velosiped",
    "car": "mashina",
    "motorcycle": "mototsikl",
    "airplane": "samolyot",
    "bus": "avtobus",
    "train": "poyezd",
    "truck": "yuk mashinasi",
    "boat": "qayiq",
    "traffic light": "svetofor",
    "fire hydrant": "gidrant",
    "stop sign": "stop belgi",
    "parking meter": "parking hisoblagich",
    "bench": "skameyka",
    "bird": "qush",
    "cat": "mushuk",
    "dog": "it",
    "horse": "ot",
    "sheep": "qo‘y",
    "cow": "sigir",
    "elephant": "fil",
    "bear": "ayiq",
    "zebra": "zebra",
    "giraffe": "jirafa",
    "backpack": "ryukzak",
    "umbrella": "soyabon",
    "handbag": "sumka",
    "tie": "galstuk",
    "suitcase": "chamadon",
    "frisbee": "frisbi",
    "skis": "chang‘i",
    "snowboard": "snoubord",
    "sports ball": "sport to‘pi",
    "kite": "varrak",
    "baseball bat": "beysbol tayoq",
    "baseball glove": "beysbol qo‘lqop",
    "skateboard": "skeytbord",
    "surfboard": "syorf",
    "tennis racket": "tennis raketka",
    "bottle": "butulka",
    "wine glass": "bokel",
    "cup": "stakan",
    "fork": "sanchqi",
    "knife": "pichoq",
    "spoon": "qoshiq",
    "bowl": "kosa",
    "banana": "banan",
    "apple": "olma",
    "sandwich": "sendvich",
    "orange": "apelsin",
    "broccoli": "brokkoli",
    "carrot": "sabzi",
    "hot dog": "hot-dog",
    "pizza": "pitssa",
    "donut": "ponchik",
    "cake": "tort",
    "chair": "stul",
    "couch": "divan",
    "potted plant": "guldon",
    "bed": "karavot",
    "dining table": "ovqat stoli",
    "toilet": "unitaz",
    "tv": "televizor",
    "laptop": "noutbuk",
    "mouse": "sichqoncha",
    "remote": "pult",
    "keyboard": "klaviatura",
    "cell phone": "telefon",
    "microwave": "mikroto‘lqinli",
    "oven": "duxovka",
    "toaster": "toster",
    "sink": "rakovina",
    "refrigerator": "muzlatgich",
    "book": "kitob",
    "clock": "soat",
    "vase": "vaza",
    "scissors": "qaychi",
    "teddy bear": "ayiqcha",
    "hair drier": "fen",
    "toothbrush": "tish cho‘tkasi",
}

FACE_MATCH_THRESHOLD = 0.36  # aniq tanish (galereya + margin)
FACE_SOFT_THRESHOLD = 0.28  # qisman/yomon kadr — faqat katta margin bilan
FACE_MATCH_MARGIN = 0.05  # 1-o‘rin va 2-o‘rin orasidagi farq (aralashmaslik)
FACE_MATCH_MARGIN_SOFT = 0.03
FACE_GALLERY_MAX_PROTOS = 16  # har bir odam uchun maks. embedding
FACE_ENROLL_MIN_QUALITY = 0.40
FACE_ENROLL_DUPLICATE_WARN = 0.82  # enroll: boshqa odam bilan juda o‘xshash
FACE_DET_SIZE = (1600, 1600)  # distant / partial faces (GPU)
FACE_DET_THRESH = 0.40
FACE_EVERY_N = 1  # per camera visit in round-robin

# Face runs on person crops (not tiny AI frame). Soft gates; person-box is primary filter.
FACE_MIN_DET_SCORE = 0.38  # allow partial faces into temporal bank
FACE_MIN_SIDE = 28  # on crop (already zoomed)
FACE_MIN_DISPLAY_SIDE = 40  # display’da juda mayda “yuz”larni rad etish
FACE_MIN_BRIGHTNESS = 14.0
FACE_MIN_SHARPNESS = 5.0
FACE_REQUIRE_PERSON = False  # crop already from YOLO person
FACE_CROP_MIN = 224  # upscale person head crop
FACE_TEMPORAL_SEC = 2.5  # multi-frame window
FACE_TEMPORAL_TOP_K = 5
FACE_CONFIRM_HITS = 4  # galereya + margin bilan qat’iyroq tasdiqlash
FACE_IDENTITY_HOLD_SEC = 90.0  # tanish ismni uzoqroq saqlash
FACE_ROOM_SESSION_ENABLED = True  # xonada tanilgach chiqguncha saqlash
FACE_ROOM_EMPTY_SEC = 180.0  # xona bo‘sh 3 daqiqa → sessiya tugaydi
FACE_ROOM_MATCH_IOU = 0.10
FACE_ROOM_MAX_OCCUPANTS = 8
FACE_SPATIAL_TTL_SEC = 12.0  # track uzilganda ham qisqa uzilish
FACE_SPATIAL_IOU_THRESH = 0.20
FACE_UNKNOWN_MIN_QUALITY = 0.55  # mayda/xira begonalarni kamaytirish
FACE_USE_GPU = True  # try CUDAExecutionProvider when available
FACE_ENROLL_DET_SIZE = (640, 640)  # portret enroll — 1600 ba’zan yuzni yo‘qotadi
FACE_ENROLL_DET_THRESH = 0.35

# Anti-spoof (qog‘oz/ekran) + mikro-harakat jonlilik
FACE_ANTISPOOF_ENABLED = True
FACE_ANTISPOOF_MODEL_DIR = DATA_DIR / "antispoof_models"
FACE_ANTISPOOF_REAL_THRESH = 0.55  # ensemble real ehtimollik
FACE_ANTISPOOF_MOTION_BYPASS = 0.72  # kuchli real → motion kutmaslik
FACE_LIVENESS_ENABLED = True
FACE_LIVENESS_MIN_FRAMES = 4
FACE_LIVENESS_MIN_SEC = 0.45
FACE_LIVENESS_MOTION_THRESH = 0.08
FACE_LIVENESS_SOFT_THRESH = 0.035
FACE_LIVENESS_WINDOW_SEC = 3.5
# Jonlilik bir marta o‘tgach qayta tekshirilmaydi — faqat xona bo‘shaganda (FACE_ROOM_EMPTY_SEC)
FACE_LIVENESS_TRACK_MERGE_IOU = 0.22

UNKNOWN_COOLDOWN_SEC = 180.0  # begona spam ↓
KNOWN_COOLDOWN_SEC = 15.0
UNKNOWN_MATCH_THRESHOLD = 0.38
UNKNOWN_SAVE_INTERVAL_SEC = 45.0
UNKNOWN_MAX_PHOTOS_PER_PERSON = 20
# Begona yuzlar (kamera saqlagan) bazada qancha turadi, keyin o‘chiriladi
UNKNOWN_TTL_SEC = 2 * 3600  # 2 soat
UNKNOWN_PURGE_INTERVAL_SEC = 60.0  # tozalash chastotasi

HOST = "0.0.0.0"
PORT = 8080

# Web login (tunnel / LAN). Override: set AUTH_PASSWORD env.
AUTH_ENABLED = True
AUTH_USERNAME = "admin"
AUTH_PASSWORD = "CHANGE_ME_PASSWORD"
AUTH_SECRET = "CHANGE_ME_SESSION_SECRET_KEY"

COLOR_KNOWN = (80, 200, 80)
COLOR_UNKNOWN = (60, 60, 230)
COLOR_SPOOF = (0, 140, 255)  # orange — rasm/ekran
COLOR_OBJECT = (220, 180, 60)
COLOR_PERSON_BOX = (200, 160, 40)
COLOR_FIGHT = (40, 40, 255)  # BGR red
COLOR_LITTER = (0, 165, 255)  # BGR orange

# Fight / urush — precision-first (1 odam / suhbat / qo‘l = URUSH emas)
FIGHT_ENABLED = True
FIGHT_BACKEND = "stgcn"  # stgcn | heuristic
FIGHT_TRACKER = "bytetrack"
FIGHT_POSE_MODEL = "yolov8l-pose.pt"
FIGHT_POSE_IMGSZ = 960
FIGHT_POSE_HALF = True
FIGHT_SEQ_LEN = 48
FIGHT_ACTION_THRESH = 0.78
FIGHT_ACTION_DEVICE = 0
FIGHT_MIN_PEOPLE = 2
FIGHT_MIN_PERSON_CONF = 0.50
FIGHT_MAX_NORM_DIST = 0.85
FIGHT_MIN_IOU = 0.04
FIGHT_MAX_PAIR_IOU = 0.38  # yuqori IoU = bitta odam ikki box
FIGHT_FLOW_THRESH = 3.8
FIGHT_JITTER_THRESH = 0.10
FIGHT_SCORE_THRESH = 0.80
FIGHT_CONFIRM_SEC = 1.6
FIGHT_HOLD_SEC = 2.5
FIGHT_COOLDOWN_SEC = 20.0
FIGHT_SCAN_ALL = True
FIGHT_REQUIRE_POSE = True
FIGHT_MIN_POSE_SCORE = 0.50
FIGHT_MIN_FLOW_SCORE = 0.55
FIGHT_MIN_INTRUSION = 0.22  # qo‘l boshqa tanaga kirishi majburiy
FIGHT_SIDE_BY_SIDE_PENALTY = True
FIGHT_PAIR_MIN_LEN = 24

# Litter / axlat tashlash (bottle/cup/bowl/wine glass → yerga)
LITTER_ENABLED = True
LITTER_CLASSES = ("bottle", "cup", "bowl", "wine glass")
LITTER_MIN_CONF = 0.35  # biroz sezgirroq
LITTER_MIN_PERSON_CONF = 0.35
LITTER_ASSOC_DIST = 1.35  # object–person center dist / person height
LITTER_HAND_MAX_RATIO = 0.72  # object cy within person box (from top) = "qo‘lda"
LITTER_GROUND_MIN_RATIO = 0.85  # 0.88 → 0.85
LITTER_GROUND_STILL_PX = 18.0  # groundda deyarli harakatsiz
LITTER_GROUND_HOLD_SEC = 0.7  # 1.0 → 0.7
LITTER_CONFIRM_SEC = 1.0  # 1.4 → 1.0
LITTER_HOLD_SEC = 3.0
LITTER_COOLDOWN_SEC = 20.0
LITTER_SCORE_THRESH = 0.58  # 0.62 → 0.58
LITTER_SCAN_ALL = True
# A: pol ROI — stoldagi narsani "yerga" deb olishni kamaytiradi
LITTER_REQUIRE_FLOOR = True
LITTER_FLOOR_RATIO = 0.68  # obyekt cy >= frame_h * ratio
# B: pose "tashlash" — bilak pastga harakati
LITTER_USE_POSE = True
LITTER_THROW_DROP_RATIO = 0.10  # wrist pastga / person.h
LITTER_THROW_WINDOW_SEC = 0.8
LITTER_THROW_MIN = 0.30  # past bo‘lsa score jarimasi
LITTER_FOCUS_EVERY_FRAME = True  # focus kamerada litter har AI tsikl (bg skip’dan mustaqil emas — focus allaqachon har doim)
# C / Mode B: yerda allaqachon yotgan axlat (tashlash zanjirisiz)
LITTER_STATIC_ENABLED = True
LITTER_STATIC_REQUIRE_PERSON = False  # bo‘sh kabinetda ham ushlash
LITTER_STATIC_FLOOR_RATIO = 0.70  # throw’dan biroz qattiqroq pol
LITTER_STATIC_MIN_CONF = 0.40
LITTER_STATIC_HOLD_SEC = 3.5  # pol’da shu muddat turishi
LITTER_STATIC_CONFIRM_SEC = 2.0
LITTER_STATIC_COOLDOWN_SEC = 45.0
LITTER_STATIC_SCORE_THRESH = 0.55
LITTER_STATIC_STILL_PX = 22.0
LITTER_STATIC_MIN_SIDE = 10  # juda mayda chalkashlikni rad etish
LITTER_STATIC_MAX_SIDE_FRAC = 0.22  # kadr kengligining 22% dan katta = odatda mebel/emal
