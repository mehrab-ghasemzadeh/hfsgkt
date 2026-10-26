# ماتریس آزمایش‌های لازم

ابتدا داده‌ها و پروتکل را تثبیت کنید. جدول source-only و target-warmup جدا باشد. همهٔ variantها باید روی یک manifest، seed، backbone و تعداد iteration اجرا شوند. پوشهٔ خروجی و TensorBoard برای هر variant/seed جدا شود.

| نام آزمایش | تغییر | سؤال |
|---|---|---|
| Prototype | `eval.evaluate --baseline prototype` روی encoder ثابت | baseline ساده چقدر است؟ |
| High + GAT | `model.head.feature_source=high` | انتخاب ویژگی میانی کمک می‌کند؟ |
| Hybrid + pairwise MLP | `model.head.transduction_type=pairwise` | گراف فراتر از scoring query-support اثر دارد؟ |
| Hybrid + GAT residual | `model.head.alpha=1.0` | تصمیم بازسازی به‌تنهایی چگونه است؟ |
| Hybrid + GAT vote | تنظیم کامل | شاخهٔ رأی‌دهی کمک می‌کند؟ |
| بدون consistency | `stage2.consistency_weight=0` | پایداری augmentation چقدر مؤثر است؟ |
| بدون warm-up | Stage1b حذف، آموزش head روی مبدأ | استفاده از دادهٔ مقصد چقدر اثر دارد؟ |

این variantها نام KT-original/HFDA-original نمی‌گیرند. برای آن baselineها از کد نویسنده و پروتکل همسان استفاده کنید. به‌ویژه تعداد کلاس‌های مبدأ، دسترسی به labelهای مقصد و ResNet12/18 را در جدول توضیح دهید.

نمونه برای یک variant:

```bash
python -m train.stage2 --config configs/base.yaml --init checkpoints/stage1b/EuroSAT/last.pt --dataset EuroSAT --shot 5 --set model.head.transduction_type=pairwise --set stage2.output=checkpoints/ablation_pairwise --set stage2.log_dir=runs/ablation_pairwise
python -m eval.evaluate --config configs/base.yaml --checkpoint checkpoints/ablation_pairwise/EuroSAT/5shot/last.pt --dataset EuroSAT --shot 5 --set model.head.transduction_type=pairwise --output results/ablation_pairwise/EuroSAT/5shot
```

برای seedهای ۴۲، ۱۲۳ و ۷، پیش‌آموزش و مراحل بعد را با همان seed جدید اجرا کنید؛ فقط تغییر seed ارزیابی جای seed آموزش را نمی‌گیرد. checkpoint مبدأ با seed متفاوت به‌عنوان «seed مستقل کل روش» گزارش نشود.

```bash
python -m scripts.run_pipeline --config configs/source_only.yaml --profile source_only --set seed=123 --set stage1a.output=checkpoints/seed123/stage1a --set stage1a.log_dir=runs/seed123/stage1a --set stage2.output=checkpoints/seed123/stage2 --set stage2.log_dir=runs/seed123/stage2 --set evaluation.output=results/seed123
```

پس از تثبیت روش، sweep محدود روی validation: `k_neighbors` در ۴/۸/۱۲/۱۶، `alpha` در ۰/۰٫۲۵/۰٫۵/۰٫۷۵/۱ و تعداد لایه‌ها در ۱/۲/۳. کل این فضا را از ابتدا اجرا نکنید؛ ابتدا روی یک مسئلهٔ توسعهٔ مستقل، مقادیر معقول را انتخاب کنید و سپس تنظیم را ثابت کنید. با نگاه به test مقدار برنده انتخاب نشود.

خروجی پایهٔ `scripts.aggregate` هر اجرای جدا را در CSV می‌نویسد؛ میانگین و انحراف معیار seedهای آموزش باید از آن رکوردها جدا محاسبه شود. `CI95` فعلی uncertainty اپیزودهاست، نه variance آموزش. برای اختلاف paired، دقت دو روش در `episodes.csv` و یکسانی `episode_indices.jsonl` را مقایسه کنید.

پیشنهاد ترتیب مصرف GPU: ممیزی داده ← smoke ← اجرای کوتاه واقعی ← Prototype و High+GAT ← روش کامل و pairwise ← سه seed ← sweep محدود validation ← جدول نهایی و تحلیل خطا. اگر مدل کامل از baseline ساده بهتر نشد، اول شواهد و فرضیه را بررسی کنید؛ صرفاً تعداد لایه یا بودجهٔ آموزش را برای یک روش زیاد نکنید.
