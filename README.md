# KT-GNN-Hybrid

پیاده‌سازی پژوهشی ترکیب انتخاب ویژگی‌های میانی HFDA، سازگارسازی کانالی KT و استنتاج گرافی با GAT. هدف این پروژه آزمودن روش ترکیبی شماست. این پروژه بازتولید عددی کامل KT یا HFDA نیست و ادعای افزایش دقت یا پذیرش مقاله ندارد.

## تحویل پروژه

- ResNet-18، ویژگی‌های block 2/3 و مسیر high-level، وزن‌دهی پویا و نرمال‌سازی.
- سه مرحلهٔ آموزش، نمونه‌برداری اپیزودی قابل تکرار، GAT سه‌لایه و چهارهدی.
- تصمیم residual-min، شاخهٔ GNN با آموزش صریح و رأی‌دهی احتمالاتی.
- ارزیابی ۱/۵-shot، ۲-way برای Malaria و ۵-way برای سایر مقصدها، CI بر اساس دقت اپیزودها.
- ذخیرهٔ optimizer، scheduler، RNG، تنظیمات و اثرانگشت manifest؛ ادامهٔ آموزش از checkpoint.
- CSV ورودی عمومی، تبدیل پوشه‌ها، CUB، miniImageNet و NIH ChestX؛ کنترل جدایی تصویر/کلاس/بیمار.
- TensorBoard، JSONL، CSV نتایج و شناسهٔ تصاویر هر اپیزود.

## شروع روی سرور لینوکس با NVIDIA

Python 3.10 یا جدیدتر، درایور سازگار با CUDA 12.8 و فضای کافی برای داده‌ها لازم است. نصب خودکار درایور انجام نمی‌شود. برای GPU قدیمی‌تر، جفت نسخهٔ سازگار PyTorch/torchvision و CUDA را آگاهانه انتخاب و دوباره آزمون کنید.

```bash
cd kt-gnn-hybrid
nvidia-smi
bash scripts/install_server.sh
source .venv/bin/activate
python -m scripts.smoke --device cuda
```

اسکریپت نصب از wheel رسمی PyTorch استفاده می‌کند. آزمون smoke تصاویر مصنوعی تولید می‌کند و معیار دقت واقعی محسوب نمی‌شود. GAT با PyTorch خالص نوشته شده است؛ نصب torch-geometric یا torch-scatter لازم نیست. فرمول attention استاندارد additive GAT است، نه جایگزین‌کردن GAT با یک MLP.

## داده‌ها

ابتدا [راهنمای داده‌ها](docs/DATASETS_FA.md) و [پروتکل پژوهشی](docs/RESEARCH_PROTOCOL_FA.md) را بخوانید. miniImageNet در فهرست دانلود شما نبود؛ برای مقایسهٔ رایج مقاله باید آن را به‌عنوان مبدأ فراهم کنید. `configs/cub_source.yaml` آزمایش جایگزینِ مبدأ CUB است و هم‌ارز بنچمارک miniImageNet نیست.

در `configs/base.yaml` مسیر `root` هر دیتاست و manifestها را مطابق سرور تغییر دهید. هر CSV:

```csv
path,label,group
images/class_a/001.jpg,class_a,patient_001
images/class_b/002.jpg,class_b,patient_002
```

مسیر تصویر نسبت به `root` است؛ `label` رشتهٔ کلاس است و در هر اپیزود دوباره از صفر شماره‌گذاری می‌شود. ستون `group` برای بیمار/ضایعه اختیاری است. فایل چندبرچسبی را مستقیماً وارد نکنید. نمونهٔ آماده‌سازی:

```bash
python -m scripts.prepare_data --format folders --root /srv/datasets/EuroSAT/2750 --out data/manifests/EuroSAT --split classes --train-classes 5 --test-classes 5
python -m scripts.check_data --config configs/base.yaml --decode-all
```

تقسیم تولیدشده توسط این ابزار «تقسیم سفارشی قابل تکرار» است. برای جدول مقایسهٔ بنچمارک باید manifest رسمی/تأییدشده داشته باشید. با `--split none --output-name train` می‌توانید تقسیم ازقبل‌موجود را حفظ کنید. فایل‌ها کپی یا جابه‌جا نمی‌شوند.

## انتخاب پروتکل

**پیشنهاد شروع: source-only**. encoder و head روی دادهٔ مبدأ آموزش می‌بینند. دادهٔ مقصد فقط هنگام آزمون و برچسب support در اختیار مدل قرار می‌گیرد. این حالت warm-up هدف ندارد. manifest آزمون باید دقیقاً با روش‌های مقایسه‌شونده یکی باشد.

```bash
python -m scripts.run_pipeline --config configs/source_only.yaml --profile source_only
```

**آزمایش دوم: target warm-up**. از train manifest مقصد برای تطبیق استفاده می‌شود؛ test manifest وارد آموزش نمی‌شود. اجراکننده برای هر مقصد Stage1b جدا می‌سازد. این دسترسی بیشتر به داده باید در مقاله گزارش شود و با baselineهای دارای همان دسترسی مقایسه شود.

```bash
python -m scripts.run_pipeline --config configs/base.yaml --profile target_warmup
```

دستور مستقیم `train.stage1b` با تنظیمات پایه همهٔ مقصدهای مجاز را چرخشی استفاده می‌کند؛ برای warm-up تک‌مقصدی `--set 'stage1b.warmup_targets=[EuroSAT]'` بدهید. این دو آزمایش از نظر دادهٔ مجاز یکسان نیستند. در `Malaria` از دادهٔ مبدأ برای آموزش head استفاده می‌شود و target warm-up خاموش است؛ Malaria دو کلاس دارد.

## اجرای مرحله‌به‌مرحله

در تمام مراحل یک فایل تنظیمات/manifest و seed ثابت استفاده کنید. مثال زیر warm-up تنها برای EuroSAT است:

```bash
python -m train.stage1a --config configs/base.yaml
python -m train.stage1b --config configs/base.yaml --init checkpoints/stage1a/last.pt --set 'stage1b.warmup_targets=[EuroSAT]' --set stage1b.output=checkpoints/stage1b/EuroSAT
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 1
python -m eval.evaluate --config configs/base.yaml --checkpoint checkpoints/stage2/EuroSAT/1shot/last.pt --dataset EuroSAT --shot 1
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 5
python -m eval.evaluate --config configs/base.yaml --checkpoint checkpoints/stage2/EuroSAT/5shot/last.pt --dataset EuroSAT --shot 5
```

برای source-only، Stage1b را حذف کنید و به Stage2 مستقیماً checkpoint مرحلهٔ 1a بدهید؛ در همهٔ دستورها `--config configs/source_only.yaml` استفاده کنید.

ادامهٔ آموزش پس از قطع سرور:

```bash
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 1 --resume checkpoints/stage2/EuroSAT/1shot/last.pt
```

تنظیمات مدل، داده، seed و برنامهٔ آموزش را هنگام resume تغییر ندهید. Stage1a از پایان آخرین epoch ذخیره‌شده و Stage1b/2 از آخرین step ذخیره‌شده ادامه می‌دهند. مدل و آمار BatchNorm در encoder مرحلهٔ دوم ثابت می‌مانند.

## انتخاب مدل و جلوگیری از نشت

اگر manifest `val` وجود داشته باشد، Stage1a و Stage2 اعتبارسنجی اپیزودی انجام داده و `best.pt` را نیز ذخیره می‌کنند. به‌صورت پیش‌فرض pipeline از `last.pt` با بودجهٔ آموزش ازپیش‌ثابت استفاده می‌کند. انتخاب `best.pt` باید از قبل برای همهٔ روش‌ها یکسان تعیین شود؛ test برای انتخاب مدل یا alpha نیست. مسیر val باید تعداد کلاس و تصویر کافی داشته باشد.

`class_disjoint` اشتراک کلاس بین تقسیم‌ها را رد می‌کند. `image_disjoint` اجازهٔ اشتراک کلاس را می‌دهد، اما اشتراک مسیر تصویر/بیمار را رد می‌کند؛ این دو نوع آزمایش را در یک جدولِ بدون توضیح ادغام نکنید. اگر ستون group کامل باشد، support و query نیز بیمار مشترک ندارند. کنترل مبتنی بر مسیر جای بررسی تصاویر تکراری با نام دیگر یا داده‌های مشتق‌شده را نمی‌گیرد.

## آزمایش حذف مؤلفه‌ها

[ماتریس آزمایش‌ها](docs/EXPERIMENTS_FA.md) شامل دستورها و تنظیمات است. گزینه‌ها:

```bash
# کنترل بدون گراف، با MLP شرطی روی query و support
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 5 --set model.head.transduction_type=pairwise --set stage2.output=checkpoints/pairwise
# فقط ویژگی high-level
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 5 --set model.head.feature_source=high --set stage2.output=checkpoints/high_only
# residual-only با حفظ بودجه و آموزش دیگر مؤلفه‌ها
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 5 --set model.head.alpha=1.0 --set stage2.output=checkpoints/residual
```

در ارزیابی همان override مدل را تکرار کنید. کنترل pairwise بازتولید KT اصلی نیست. baseline سادهٔ prototype نیز موجود است: `eval.evaluate --baseline prototype --checkpoint ...` با همان dataset/shot/config.

## لاگ و خروجی

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
python -m scripts.aggregate --root results --output results/summary.csv
```

برای اتصال راه‌دور از SSH port forwarding استفاده کنید. فایل `summary.json` علاوه بر mean و CI، seed آموزش/ارزیابی، data access، نسخهٔ PyTorch و تنظیمات را دارد. `episodes.csv` دقت هر اپیزود و هر دو شاخه را ذخیره می‌کند. `episode_indices.jsonl` اجازهٔ تطبیق دقیق اپیزودهای مقایسه را می‌دهد. برای جایگزینی نتیجهٔ موجود باید `--overwrite` بدهید.

برای اجرای طولانی داخل tmux:

```bash
tmux new -s ktgnn
GPU=0 CONFIG=configs/source_only.yaml PROFILE=source_only bash scripts/run_server.sh > run.log 2>&1
# جدا شدن از نشست: Ctrl+B سپس D
```

## حافظه و سرعت

پیش‌فرض‌ها نقطهٔ شروع‌اند و تضمین ظرفیت برای هر GPU نیستند. در کمبود VRAM، `stage1a.batch_size`، `stage1b.source_batch_size`، `stage2.encode_batch_size` و `stage2.query_batch_size` را کم کنید. `evaluation.query_batch_size` فقط گروه‌بندی محاسبهٔ گراف‌های مستقل را تغییر می‌دهد. افزایش shot به ۲۰/۵۰ اندازهٔ گراف و مصرف حافظه را بالا می‌برد. آموزش چند GPU/DDP و AMP در این نسخه پیاده نشده‌اند؛ یک GPU برای هر فرایند در نظر گرفته شده است.

## آزمون‌ها و وضعیت

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke --device cuda
```

آزمون محلی با PyTorch 2.9.0 و torchvision 0.24.0 روی CUDA انجام شده است. تست‌های واحد، گرادیان هر دو شاخه، جدایی بیمار، جلوگیری از نشت، permutation، checkpoint و resume را پوشش می‌دهند. اجرای لینوکس/سرور واقعی شما و آموزش با داده‌های واقعی هنوز انجام نشده است. [گزارش اعتبارسنجی](docs/VALIDATION.md) و [تفاوت‌ها با راهنما/مقالات](docs/IMPLEMENTATION_NOTES.md) را ببینید.

## ساختار

```text
configs/       تنظیمات و پروفایل‌های آزمایش
models/        encoder، GAT، adaptation، projector و classifier
losses/        CE طیفی، KL دامنه، CE دامنه، reconstruction و consistency
data_loaders/  manifest، تبدیلات و episode sampler
train/         stage1a، stage1b، stage2 و checkpoint orchestration
eval/          ارزیابی و اعتبارسنجی
scripts/       آماده‌سازی داده، ممیزی، smoke، اجرا و تجمیع نتایج
tests/         آزمون‌های عددی و داده
docs/          مستندات پژوهشی و اجرای سرور
```
