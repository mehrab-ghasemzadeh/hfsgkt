# پیش‌بینی یک اپیزود جدید

برای استفاده از مدل آموزش‌دیده روی تصاویر خودتان، support CSV شامل `path,label` و query CSV فقط شامل `path` بدهید. تعداد نمونه‌های support هر کلاس می‌تواند متفاوت باشد؛ نوع مسئله و دامنهٔ ورودی باید با آزمایش مدل سازگار باشد. عدد confidence احتمال رأی ترکیبی مدل است و اطمینان کالیبره‌شدهٔ پزشکی محسوب نمی‌شود.

```bash
python -m scripts.predict_episode --checkpoint checkpoints/stage2/EuroSAT/5shot/last.pt --root /srv/new_episode --support /srv/new_episode/support.csv --queries /srv/new_episode/queries.csv --output results/predictions.csv --device cuda
```

هیچ برچسبی برای query به مدل داده نمی‌شود. labelهای خروجی همان رشته‌های label در support هستند. این دستور برای استنتاج است؛ جای پروتکل ارزیابی بنچمارک را نمی‌گیرد.
