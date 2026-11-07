# شش اجرای تکمیلی GIFT-64

این ابزار فقط انتخاب پیکربندی‌ها را محدود می‌کند. فایل‌های `gift64.py`،
`data.py`، `train.py` و `run_experiments.py` همان نسخهٔ اصلاح‌شدهٔ قبلی‌اند و
بازنویسی نمی‌شوند. اسکریپت قبل از شروع، یکسان‌بودن هر چهار فایل با نسخهٔ مبنای
سید صفر را بررسی می‌کند. آزمون‌های ده‌گانهٔ بسته نیز پیش از اجرا انجام می‌شوند.

## قرار دادن فایل‌ها

این ZIP را داخل همان پوشهٔ `gift64_corrected` استخراج کن؛ فایل
`run_followup.py` باید کنار `run_experiments.py` باشد. از محیط مجازی قبلی استفاده
کن و بسته‌های نصب‌شده را ارتقا نده. نمونهٔ مسیر قبلی شما:

```bash
cd ~/Thesis/old-paper-gift/GIFT64_Corrected_Experiments/gift64_corrected
unzip ~/Downloads/GIFT64_Followup_Tools.zip
```

اگر ZIP را جای دیگری دانلود کرده‌ای، مسیرش را عوض کن. هر دو فایلِ این بسته
نام تازه دارند و جای کد رمز یا آموزش را نمی‌گیرند.

## تنظیمات اصلی

- فقط `r6_1_A`، `r6_weak4` و `r6_mixed4`، همگی در شش دور.
- سید تولید داده و آموزش: ۱ و ۲؛ اجرای صفر دوباره انجام نمی‌شود.
- تعداد کل ماتریس‌ها در هر دیتاست: ۱٬۰۴۸٬۵۷۶، نصف از هر کلاس.
- ۱۵ دورهٔ آموزش؛ مدل MLP با لایه‌های پنهان ۱۲۸ و ۶۴؛ batch برابر ۱۲۸؛ Adam با نرخ ۰٫۰۰۱.
- کلید ثابت `0123456789abcdef0123456789abcdef` و split seed برابر ۴۲.
- CPU و دو thread، مطابق تنظیم پیش‌فرض قبلی؛ کلاس منفی `encrypted-pairs`.
- ترتیب اختلاف‌ها عیناً از تعریف‌های قبلی خوانده می‌شود.

## اول اجرای کوچک

```bash
venv/bin/python -u run_followup.py \
  --out results/followup_smoke \
  --n 2048 --epochs 2 --seeds 1 2 \
  --device cpu --threads 2 \
  --negative encrypted-pairs \
  --key 0x0123456789abcdef0123456789abcdef \
  --resume
```

این شش اجرای کوچک فقط سالم‌بودن مسیر اجرا را نشان می‌دهند؛ اعدادشان برای مقاله
نیستند. در پیام‌های تولید داده باید فقط سه نام بالا، با سید ۱ و ۲ دیده شوند.

## اجرای اصلی

بعد از موفقیت اجرای کوچک، این دستورها را در Bash اجرا کن:

```bash
mkdir -p results
set -o pipefail
venv/bin/python -u run_followup.py \
  --out results/followup_seed1_2 \
  --n 1048576 --epochs 15 --seeds 1 2 \
  --device cpu --threads 2 \
  --negative encrypted-pairs \
  --key 0x0123456789abcdef0123456789abcdef \
  --resume 2>&1 | tee -a results/followup_seed1_2.log
```

سیدها با فاصله نوشته می‌شوند: `--seeds 1 2`، نه `1,2`.
از `run_followup.py` استفاده کن؛ دستور عادی `run_experiments.py` هر هشت تنظیم
قبلی را اجرا می‌کند. هیچ تنظیم مدل، دورهٔ آموزش یا کلید را بر اساس نتیجه تغییر
نده. این مرحله همچنان آزمایشِ یک کلید است، نه تعمیم به کلیدهای جدید.

## خروجی و ادامه

باید شش فایل `metrics.json` ساخته شوند، در مسیرهای زیر:

```text
results/followup_seed1_2/r6_1_A/seed_1/metrics.json
results/followup_seed1_2/r6_1_A/seed_2/metrics.json
results/followup_seed1_2/r6_weak4/seed_1/metrics.json
results/followup_seed1_2/r6_weak4/seed_2/metrics.json
results/followup_seed1_2/r6_mixed4/seed_1/metrics.json
results/followup_seed1_2/r6_mixed4/seed_2/metrics.json
```

داخل هر پوشه، `history.json`، `predictions.npz`، `hamming_histograms.npz`،
`split_indices.npz` و `model.pt` هم ذخیره می‌شوند. دیتاست موقت بعد از موفقیت
آموزش حذف می‌شود؛ نیازی به `--keep-datasets` نداریم.

`summary.csv`، `summary.json`، `results_table.tex` و `run_manifest.json` در
ریشهٔ خروجی قرار دارند. خلاصهٔ این پوشه فقط سیدهای ۱ و ۲ را دارد؛ برای گزارش
سه‌اجرا، بعداً آن‌ها را با معیارهای سید صفر ترکیب می‌کنیم. فعلاً جدول مقاله را
مستقیماً با خلاصهٔ دو اجرا جایگزین نکن.

با تکرار دقیق همان دستور، اجراهای کامل به‌کمک `--resume` رد می‌شوند. اگر پیام
`Incomplete run` دیدی، ادامه از میانهٔ آموزش پیاده‌سازی نشده است؛ مسیر دقیقِ
همان اجرای ناقص را به نامی دیگر منتقل کن و دستور را تکرار کن. پوشه‌های اجراهای
کامل و خروجی‌های سید صفر را حذف نکن. تنظیمات و فایل‌های منبع را بین ادامه‌ها
تغییر نده.

## بستهٔ نتایج برای بررسی

پس از پیام `Completed`، این دستور ZIP نتایج و لاگ را می‌سازد:

```bash
venv/bin/python -m zipfile -c results/GIFT64_followup_seed1_2_results.zip \
  results/followup_seed1_2 results/followup_seed1_2.log
```

فایل `results/GIFT64_followup_seed1_2_results.zip` را برای بررسی ارسال کن.
این دستور وزن‌ها، پیش‌بینی‌ها و تاریخچه را هم نگه می‌دارد؛ برای تحلیل شکست
مدل‌های نزدیک شانس مفیدند. بستهٔ مقاله و مقالهٔ همراه در این مرحله تغییر نمی‌کنند.

## بررسی انجام‌شده برای ابزار

اثر انگشت سه فایل رمز، داده و آموزش با همان اثر انگشت‌های ثبت‌شده در خروجی
سید صفر تطبیق داده شد. ابزار انتخاب پیکربندی‌ها و مسیر اجرای نسخهٔ اصلی بررسی
می‌شود؛ آموزش جدید مقاله باید روی سیستم شما انجام شود. هیچ نتیجهٔ جدیدی داخل
این بسته وجود ندارد.
