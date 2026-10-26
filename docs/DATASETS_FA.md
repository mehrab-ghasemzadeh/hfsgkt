# داده‌ها و آماده‌سازی روی سرور

این پروژه فایل `dslist.txt` را فهرست منابع داده تلقی می‌کند؛ دستورهای دانلود آن اجرا نشده‌اند. مسیرهای واقعی سرور هنوز در اختیار پروژه نیستند. تبدیل‌ها تصاویر موجود را می‌خوانند و manifest می‌سازند؛ داده را دانلود یا جابه‌جا نمی‌کنند.

## منابع شما و تفاوت با بنچمارک

| دادهٔ موجود در فهرست | نقش پیشنهادی | بررسی ضروری پیش از گزارش مقاله |
|---|---|---|
| CUB-200-2011 از wenewone/cub2002011 | مقصد تکمیلی ۵-way | تطبیق نسخه و شناسه‌های کلاس؛ HFDA تعداد ۵۰/۵۰ کلاس CUB را ذکر می‌کند |
| cell-images-for-detecting-malaria | مقصد تکمیلی ۲-way | فقط دو کلاس؛ ۵-way و تقسیم کلاسِ train/test هر دو برای آن معتبر نیستند |
| nih-chest-xrays/data | مقصد اصلی | فایل metadata، حذف نمونه‌های چندبرچسبی، گروه‌بندی بیمار |
| crop-disease-detection-dataset | مقصد اصلی پس از تطبیق نسخه | تعداد کلاس‌ها، تصاویر اصلی/افزوده‌شده و تطابق با نسخهٔ PlantVillage بنچمارک |
| eurosat-dataset | مقصد اصلی پس از تطبیق | نسخهٔ RGB ده‌کلاسه؛ دادهٔ چندطیفی ۱۳کاناله ورودی این مدل نیست |
| skin-cancer9-classesisic | مقصد سفارشی/تکمیلی تا تأیید تطابق | نام ISIC کافی نیست؛ نسخهٔ ۹کلاسه را ISIC2018 استاندارد ننویسید |
| miniImageNet، در فهرست نبود | مبدأ پیشنهادی | تهیهٔ نسخه و تقسیم شناخته‌شده و ثابت؛ ۶۴/۸۰ کلاس را برای همهٔ روش‌ها یکسان کنید |

[مخزن اصلی BSCD-FSL](https://github.com/IBM/cdfsl-benchmark) منبع تعیین نسخه‌ها و loaderهای مقایسه است. مجموعهٔ رسمی [ISIC2018 Task 3 training](https://api.isic-archive.com/collections/66/) شامل ۱۰٬۰۱۵ تصویر است؛ صرف نام یک آرشیو Kaggle، هم‌ارزی با این مجموعه را ثابت نمی‌کند.

## فایل ورودی مشترک

```csv
path,label,group
images/00001.png,Atelectasis,patient_0001
images/00002.png,Mass,patient_0002
```

`group` برای بیمار/ضایعه است و اگر در دسترس نیست خالی می‌ماند. بدون شناسهٔ بیمار نمی‌توان ادعا کرد کنترل جدایی بیماران انجام شده است. از نگاشت دستی اشتباه یا اختصاص یک بیمار به چند split جلوگیری کنید. sampler هنگام وجود group، بیمارهای support را از query حذف می‌کند.

## مثال‌های تبدیل

تمام مسیرها نمونه‌اند و باید با محل استخراج روی سرور تطبیق داده شوند.

### miniImageNet با CSV استاندارد

CSV ورودی `filename,label` است. تصاویر می‌توانند تخت یا داخل پوشهٔ کلاس باشند. هر تقسیم را بدون بازتقسیم حفظ کنید:

```bash
python -m scripts.prepare_data --format mini_csv --root /srv/datasets/miniImageNet --metadata /srv/datasets/miniImageNet/train.csv --split none --output-name train --out data/manifests/miniImageNet
python -m scripts.prepare_data --format mini_csv --root /srv/datasets/miniImageNet --metadata /srv/datasets/miniImageNet/val.csv --split none --output-name val --out data/manifests/miniImageNet
```

برای آزمایش ۸۰ کلاس، CSV train و val استاندارد را از نظر رکورد با هم ترکیب کنید و یک validation مستقل از کلاس‌های باقی‌مانده یا طرح ازپیش‌تعیین‌شده انتخاب کنید. همین کار را برای همهٔ روش‌های مقایسه انجام دهید. `test` مقصد برای انتخاب این تنظیمات استفاده نمی‌شود.

### CUB

```bash
python -m scripts.prepare_data --format cub --root /srv/datasets/CUB_200_2011 --split classes --train-classes 50 --test-classes 50 --out data/manifests/CUB
```

از ۲۰۰ کلاس، این دستور ۵۰ train، ۵۰ test و ۱۰۰ val می‌سازد. seed و فهرست کلاس‌ها ثابت‌اند؛ این انتخاب تصادفی به‌خودی‌خود split تأییدشدهٔ HFDA نیست. در صورت استفاده از val تولیدشده، مسیر آن را به config اضافه کنید؛ فایل اضافه‌شده به‌صورت خودکار مصرف نمی‌شود.

### Malaria

```bash
python -m scripts.prepare_data --format folders --root /srv/datasets/cell_images --split none --output-name test --out data/manifests/Malaria
```

root باید مستقیماً شامل دو پوشهٔ کلاس باشد. default پروژه از Malaria برای آموزش استفاده نمی‌کند؛ head دوکلاسه روی اپیزودهای مبدأ یاد می‌گیرد. برای یک آزمایش سفارشی image-disjoint می‌توان split تصاویر ساخت، اما آن را از نتایج بدون آموزش روی مقصد جدا گزارش کنید. HFDA به‌دلیل کمبود کلاس Malaria، از کلاس‌های miniImageNet برای بخش توسعه استفاده کرده است؛ تقسیم ۵۰٪ کلاس‌های Malaria قابل تعمیم نیست.

### NIH ChestX

```bash
python -m scripts.prepare_data --format chestx --root /srv/datasets/chestx --metadata /srv/datasets/chestx/Data_Entry_2017.csv --split classes --train-classes 2 --test-classes 5 --out data/manifests/ChestX
```

ابزار به‌صورت پیش‌فرض هفت برچسب Atelectasis، Cardiomegaly، Effusion، Infiltration، Mass، Nodule، Pneumothorax را انتخاب می‌کند و No Finding و چندبرچسبی‌ها را کنار می‌گذارد. بیمارانی که بین splitهای کلاس‌بنیاد تداخل دارند حذف می‌شوند. این انتخاب ۲/۵ یک پروتکل سفارشی است؛ آن را نصف‌کردن کلاس‌ها یا بازتولید عین HFDA نام نگذارید. برای source-only، از همهٔ کلاس‌های آزمونِ پروتکل مبنا manifest `test` بسازید و train را در config غیرفعال کنید.

### EuroSAT RGB

```bash
python -m scripts.prepare_data --format folders --root /srv/datasets/EuroSAT/2750 --split classes --train-classes 5 --test-classes 5 --out data/manifests/EuroSAT
```

### CropDisease

```bash
python -m scripts.prepare_data --format folders --root /srv/datasets/cropdisease --split classes --out data/manifests/CropDisease
```

root باید مستقیماً پوشه‌های برچسب‌ها را داشته باشد؛ اگر root شامل `train` و `test` است، نباید خود این دو به‌عنوان برچسب خوانده شوند. از manifestهای واقعی برای حفظ تقسیم موجود استفاده کنید. رکوردهای augmented از یک تصویر مادر باید group مشترک بگیرند و از splitهای متفاوت حذف شوند.

### ISIC ۹کلاسهٔ شما

```bash
python -m scripts.prepare_data --format folders --root /srv/datasets/skin-cancer9/Train --split classes --train-classes 4 --test-classes 5 --out data/manifests/ISIC
```

این دستور فقط محتویات Train آن آرشیو را به آزمایش سفارشی class-disjoint تبدیل می‌کند و Test اصلی آرشیو را مصرف نمی‌کند. در مقاله دقیقاً همین را گزارش کنید؛ اگر می‌خواهید تقسیم اصلی آرشیو حفظ شود، دو manifest با مسیرهای root مشترک بسازید و `image_disjoint` انتخاب کنید. تعداد کلاس‌های هر split و تصاویر کافی برای K+Q باید بررسی شوند.

## ممیزی پیش از آموزش

```bash
python -m scripts.check_data --config configs/base.yaml --decode-all
```

ممیزی شمارش کلاس‌ها و حداقل نمونه‌ها، وجود مسیرها، اشتراک مسیر/بیمار/کلاس طبق policy و خواندن تصاویر را بررسی می‌کند. این کنترل، تشخیص کامل شباهت تصاویر، صحت تشخیص پزشکی یا تأیید split رسمی نیست. برای انتشار، جدول تعداد نمونه/کلاس، حذف‌شده‌ها، نسخهٔ آرشیو و شناسهٔ کلاس‌ها را نگه دارید. با فایل‌های تغییرنام‌یافته یا ضایعه‌های دارای چند عکس، metadata مادر/بیمار را از منبع داده وارد کنید.
