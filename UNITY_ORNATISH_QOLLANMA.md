# Unity o‘rnatish bo‘yicha batafsil qo‘llanma (Windows)

Ushbu qo‘llanmada Unity’ni noldan boshlab to‘g‘ri o‘rnatish, kerakli modullarni tanlash va birinchi loyihani ishga tushirish bosqichlari keltirilgan.

---

## 1) O‘rnatishdan oldin tayyorgarlik

### Minimal tavsiyalar
- **OS:** Windows 10 yoki 11 (64-bit)
- **RAM:** kamida 8 GB (yaxshisi 16 GB)
- **Disk bo‘sh joyi:** kamida 20–30 GB
- **Internet:** barqaror ulanish (ilk o‘rnatishda katta fayllar yuklanadi)

### Muhim eslatma
- Unity’ni odatda **Unity Hub** orqali o‘rnatish tavsiya etiladi.
- Unity Editor’ning o‘zini bevosita o‘rnatish ham mumkin, lekin boshqaruv (version, modul, loyiha) uchun Hub qulayroq.

---

## 2) Unity Hub’ni yuklab olish

1. Brauzerni oching.
2. Rasmiy saytga kiring: [https://unity.com/download](https://unity.com/download)
3. **Download Unity Hub** tugmasini bosing.
4. Yuklangan fayl odatda `UnityHubSetup.exe` nomida bo‘ladi.

---

## 3) Unity Hub’ni o‘rnatish

1. `UnityHubSetup.exe` faylini ishga tushiring.
2. `I accept the terms` (yoki shunga o‘xshash) bandini belgilang.
3. `Install` tugmasini bosing.
4. O‘rnatish tugaguncha kuting.
5. `Finish` tugmasini bosib Unity Hub’ni oching.

---

## 4) Unity akkauntga kirish (yoki ro‘yxatdan o‘tish)

1. Unity Hub ochilgach, **Sign in** tugmasini bosing.
2. Unity akkauntingiz bo‘lsa login qiling.
3. Akkaunt bo‘lmasa **Create one** orqali ro‘yxatdan o‘ting.
4. E-pochtani tasdiqlash so‘ralishi mumkin — tasdiqlang.

> Eslatma: Personal foydalanish uchun ko‘pchilik holatda bepul rejim yetarli bo‘ladi.

---

## 5) Unity Editor versiyasini o‘rnatish

1. Unity Hub’da chap menyudan **Installs** bo‘limiga o‘ting.
2. **Install Editor** tugmasini bosing.
3. Tavsiya etiladi:
   - **LTS (Long Term Support)** versiyasini tanlang (barqarorroq).
4. Kerakli versiyani tanlagach **Next** bosing.

---

## 6) Qo‘shimcha modullarni tanlash

Modul tanlash oynasida loyiha turiga qarab belgilang:

### Asosiy tavsiya modullar
- **Microsoft Visual Studio Community** (C# kod yozish uchun)
- **Windows Build Support (IL2CPP)** (Windows uchun build olish)

### Kerak bo‘lsa qo‘shing
- **Android Build Support** (SDK/NDK/OpenJDK bilan) — Android app/game uchun
- **WebGL Build Support** — brauzerga export uchun
- **iOS Build Support** — iOS uchun (odatda Mac talab qilinadi)

So‘ng:
1. **Install** tugmasini bosing.
2. Yuklanish va o‘rnatish tugaguncha kuting (vaqt internet tezligiga bog‘liq).

---

## 7) Unity o‘rnatilganini tekshirish

1. Unity Hub → **Installs** bo‘limiga kiring.
2. Tanlagan Unity versiyangiz ro‘yxatda ko‘rinishi kerak.
3. Versiya yonida xatolik bo‘lmasa, o‘rnatish muvaffaqiyatli.

---

## 8) Birinchi loyiha yaratish

1. Unity Hub’da **Projects** bo‘limiga o‘ting.
2. **New project** tugmasini bosing.
3. Template tanlang:
   - `3D (URP)` yoki oddiy `3D`
   - yoki `2D` (2D loyiha uchun)
4. **Project name** kiriting (masalan: `MyFirstUnityProject`).
5. Saqlash papkasini tanlang.
6. **Create project** bosing.
7. Unity Editor ochiladi (birinchi marta biroz sekin bo‘lishi mumkin).

---

## 9) Visual Studio bilan bog‘lanishni tekshirish

1. Unity ichida `Assets` oynasida:
   - `Create` → `C# Script` qiling.
2. Scriptga ikki marta bosing.
3. Script Visual Studio’da ochilsa — hammasi to‘g‘ri.

Agar ochilmasa:
1. Unity’da `Edit` → `Preferences` → `External Tools` ga kiring.
2. **External Script Editor** ni `Visual Studio`ga o‘rnating.

---

## 10) Ko‘p uchraydigan muammolar va yechimlar

### Muammo: O‘rnatish to‘xtab qoladi
- Internetni tekshiring.
- VPN/proxy bo‘lsa vaqtincha o‘chirib ko‘ring.
- Unity Hub’ni `Run as administrator` qilib ishga tushiring.

### Muammo: Disk joy yetmayapti
- Keraksiz fayllarni o‘chiring.
- Unity install path’ni boshqa diskka o‘rnating (masalan `D:\Unity`).

### Muammo: Visual Studio topilmayapti
- Unity Hub orqali Visual Studio modulini qayta o‘rnating.
- `External Tools` dan qo‘lda tanlang.

### Muammo: Android build ishlamaydi
- `Android Build Support` ichidagi `SDK & NDK tools` va `OpenJDK` belgilanganini tekshiring.

---

## 11) Tavsiya etilgan amaliy ish tartibi

1. Har doim loyiha boshlashda **LTS** versiyadan foydalaning.
2. Har bir loyiha uchun alohida papka saqlang.
3. Kodni yo‘qotmaslik uchun Git bilan version control ishlating.
4. Katta loyiha bo‘lsa backupni cloud’ga saqlang.

---

## 12) Qisqa checklist (tez tekshiruv)

- [ ] Unity Hub o‘rnatildi
- [ ] Unity akkauntga kirdim
- [ ] LTS Unity versiyasi o‘rnatildi
- [ ] Kerakli modullar tanlandi (Visual Studio, Build Support)
- [ ] Yangi loyiha ochildi
- [ ] C# script Visual Studio’da ochildi

---

Agar xohlasangiz, keyingi bosqichda sizga **Unity’da birinchi 3D sahna yaratish**, **kamera qo‘shish**, **player harakatini yozish** bo‘yicha ham alohida amaliy qo‘llanma tayyorlab beraman.
