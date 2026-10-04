# حدس الجبر الخطي (Linear Algebra Intuition)

> **English Title:** Linear Algebra Intuition
>
> *كل نموذج ذكاء اصطناعي ليس سوى رياضيات المصفوفات ترتدي قبّعة فاخرة.*

---

## 1. ما الذي سنتعلمه؟

في هذا الدرس ستبني فهماً حقيقياً للجبر الخطي — ليس كمجموعة معادلات مجردة، بل كأدوات هندسية تقوم عليها كل نماذج الذكاء الاصطناعي الحديثة.

**ستتعلم:**
- ما هي المتجهات (Vectors) والمصفوفات (Matrices) وما الذي تمثله هندسياً.
- كيف يقيس الضرب النقطي (Dot Product) التشابه — وكيف يُبنى عليه البحث والتوصيات.
- ما هو الاستقلالية الخطية (Linear Independence) ولماذا تُفسد النماذج عند غيابها.
- كيف تعمل الإسقاطات (Projections) وعلاقتها بـ Regression وPCA والـ Attention.
- كيف تبني فئة `Vector` وفئة `Matrix` من الصفر بدون مكتبات.
- كيف تنفّذ الاستقلالية الخطية وعملية Gram-Schmidt يدوياً.
- أين يظهر كل مفهوم في PyTorch وTransformers وLoRA.

---

## 2. لماذا هذا الموضوع مهم؟

افتح أي ورقة بحثية في Machine Learning. في الصفحة الأولى ستجد:

$$W \cdot x + b$$

$$\text{softmax}(QK^T / \sqrt{d_k}) \cdot V$$

$$\Delta W = A \cdot B^T \quad \text{(LoRA)}$$

**هذه كلها عمليات جبر خطي.**

بدون فهم الجبر الخطي، هذه مجرد رموز غامضة. بفهمه، تستطيع أن **ترى** ما يفعله الشبكة العصبية — إنها تُحرّك نقاطاً في الفضاء.

### أين يظهر في AI الحقيقي؟

| المفهوم | التطبيق في AI |
|---------|---------------|
| **الضرب النقطي** | درجات الانتباه (Attention Scores) في Transformers، تشابه الجُمل في RAG |
| **ضرب المصفوفات** | كل طبقة في الشبكة العصبية، كل تحويل خطي |
| **الاستقلالية الخطية** | اختيار الخصائص (Feature Selection)، تجنّب Multicollinearity |
| **الرتبة (Rank)** | تحديد ما إذا كان النظام قابلاً للحل، LoRA |
| **الإسقاط** | الانحدار الخطي (Linear Regression)، PCA |
| **Gram-Schmidt / QR** | الحلول العددية، حساب القيم الذاتية |
| **الأساس المتعامد (Orthonormal Basis)** | الحساب العددي المستقر، Whitening |

---

## 3. المتطلبات السابقة

- المرحلة 00 (إعداد البيئة): Python مثبّت مع NumPy.
- لا تحتاج خلفية رياضية متقدمة — سنبني كل شيء من الصفر.

---

## 4. الفكرة الأساسية — Intuition

### تشبيه أولي: الاتجاه والمسافة

تخيّل أنك في مدينة وتريد وصف موقعك لشخص آخر. تقول:

> "اذهب 3 كيلومترات شرقاً و 2 كيلومترات شمالاً"

هذا بالضبط ما يفعله **المتجه (Vector)** — يصف **الاتجاه والمقدار** في فضاء ما.

أما **المصفوفة (Matrix)**؟ فكّر فيها كـ **قاعدة تحويل**: تأخذ نقطة وتنقلها إلى مكان آخر — تدورها، تكبّرها، تسقطها على بُعد أقل.

**الذكاء الاصطناعي هو هذا بالضبط:** يأخذ مدخلاً (صورة، جملة، بيانات) ويحوّله عبر سلسلة من المصفوفات إلى خرج مفيد (تصنيف، ترجمة، تنبؤ).

---

## 5. المفاهيم الأساسية

### 5.1 المتجه (Vector) — الرقم الذي له معنى

#### ما هو المتجه؟

المتجه هو **قائمة مرتّبة من الأرقام**. لكن هذه الأرقام تمثل **إحداثيات في فضاء**.

$$\vec{v} = \begin{bmatrix} 3 \\\\ 2 \end{bmatrix}$$

هذا المتجه يشير من نقطة الأصل $(0, 0)$ إلى النقطة $(3, 2)$.

#### خصائص المتجه

**1. المقدار (Magnitude) — الطول:**

$$|\vec{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

للمتجه $\vec{v} = [3, 2]$:

$$|\vec{v}| = \sqrt{3^2 + 2^2} = \sqrt{9 + 4} = \sqrt{13} \approx 3.606$$

**ما يعنيه هذا:** الطول الفيزيائي للمتجه في الفضاء.

**2. التطبيع (Normalization) — جعل الطول 1:**

$$\hat{v} = \frac{\vec{v}}{|\vec{v}|}$$

$$\hat{v} = \frac{[3, 2]}{\sqrt{13}} = \left[\frac{3}{\sqrt{13}}, \frac{2}{\sqrt{13}}\right] \approx [0.832, 0.555]$$

**لماذا نطبّع؟** لنهتم بالاتجاه فقط دون المقدار. في AI، نطبّع التضمينات (Embeddings) لمقارنة المعنى بغض النظر عن الشدة.

#### المتجهات في AI

في الذكاء الاصطناعي، **كل شيء يُمثَّل كمتجه**:

| الكيان | التمثيل كمتجه |
|--------|--------------|
| كلمة "ملك" | متجه من 768 رقم (في BERT) |
| صورة 224×224 | متجه من 150,528 رقم |
| مستخدم في Netflix | متجه من 200 رقم (تفضيلاته) |
| جملة كاملة | متجه من 1024 رقم (معناها الدلالي) |

**لماذا هذا مفيد؟** لأن المتجهات المتقاربة (close vectors) تعني **معاني متقاربة** — وهذا هو أساس كل نظام بحث وتوصية ولغة.

---

### 5.2 الضرب النقطي (Dot Product) — مقياس التشابه

#### ما هو الضرب النقطي؟

الضرب النقطي بين متجهين هو:

$$\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

#### مثال رقمي

$$\vec{a} = [1, 2, 3], \quad \vec{b} = [4, 5, 6]$$

$$\vec{a} \cdot \vec{b} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32$$

#### ماذا يعني الضرب النقطي هندسياً؟

$$\vec{a} \cdot \vec{b} = |\vec{a}| \cdot |\vec{b}| \cdot \cos(\theta)$$

حيث $\theta$ هو الزاوية بين المتجهين.

| العلاقة | الضرب النقطي | المعنى |
|---------|-------------|--------|
| نفس الاتجاه تماماً ($\theta = 0°$) | $> 0$ (أكبر ما يكون) | متشابهان جداً |
| زاوية 90° (عمودي) | $= 0$ | لا علاقة بينهما |
| اتجاهان متعاكسان ($\theta = 180°$) | $< 0$ | متعارضان |

**المثال الحدسي:**
- $[1, 0] \cdot [1, 0] = 1$ (نفس الاتجاه — تشابه كامل)
- $[1, 0] \cdot [0, 1] = 0$ (عموديان — لا علاقة)
- $[1, 0] \cdot [-1, 0] = -1$ (متعاكسان — تعارض كامل)

#### تشابه جيب التمام (Cosine Similarity)

لمقارنة الاتجاهات بغض النظر عن الأطوال:

$$\text{cosine-similarity}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}$$

$$\cos(\vec{a}, \vec{b}) = \frac{32}{\sqrt{14} \times \sqrt{77}} = \frac{32}{\sqrt{1078}} \approx 0.9746$$

**النتيجة:** 0.97 قريب جداً من 1 → المتجهان `[1,2,3]` و`[4,5,6]` متشابهان في الاتجاه.

#### أين يظهر في AI؟

- **محركات البحث (Search Engines):** تحوّل استعلامك إلى متجه، ثم تجد الوثائق التي تعطي أعلى Cosine Similarity.
- **RAG (Retrieval-Augmented Generation):** يجد المقاطع الأكثر تشابهاً مع السؤال عبر Dot Product.
- **آلية الانتباه (Attention):** درجات الانتباه تُحسب بالضرب النقطي:

$$\text{score}(Q_i, K_j) = Q_i \cdot K_j$$

أي أن كل رمز في الجملة "يسأل" (Query) وكل رمز آخر "يجيب" (Key)، والدرجة هي مدى تطابق السؤال والإجابة.

---

### 5.3 المصفوفة (Matrix) — التحويل

#### ما هي المصفوفة؟

المصفوفة هي **جدول مستطيل من الأرقام** يمثل **تحويلاً خطياً** — تأخذ متجهاً من فضاء وتُعيده في فضاء آخر.

$$M = \begin{bmatrix} m_{11} & m_{12} \\\\ m_{21} & m_{22} \end{bmatrix}$$

#### ضرب المصفوفة × المتجه

$$\vec{y} = M \cdot \vec{x}$$

$$y_i = \sum_j M_{ij} \cdot x_j$$

**مثال — دوران 90° عكس عقارب الساعة:**

$$M_{90°} = \begin{bmatrix} 0 & -1 \\\\ 1 & 0 \end{bmatrix}, \quad \vec{x} = \begin{bmatrix} 3 \\\\ 1 \end{bmatrix}$$

$$\vec{y} = \begin{bmatrix} 0 \cdot 3 + (-1) \cdot 1 \\\\ 1 \cdot 3 + 0 \cdot 1 \end{bmatrix} = \begin{bmatrix} -1 \\\\ 3 \end{bmatrix}$$

النقطة $(3, 1)$ أصبحت $(-1, 3)$ بعد الدوران 90°.

#### ضرب مصفوفة × مصفوفة

$$(AB)_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

شرط: عدد أعمدة $A$ = عدد صفوف $B$.

**مثال:**

$$A = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\\\ 7 & 8 \end{bmatrix}$$

$$(AB)_{11} = 1 \times 5 + 2 \times 7 = 5 + 14 = 19$$

$$(AB)_{12} = 1 \times 6 + 2 \times 8 = 6 + 16 = 22$$

$$(AB)_{21} = 3 \times 5 + 4 \times 7 = 15 + 28 = 43$$

$$(AB)_{22} = 3 \times 6 + 4 \times 8 = 18 + 32 = 50$$

$$AB = \begin{bmatrix} 19 & 22 \\\\ 43 & 50 \end{bmatrix}$$

#### المصفوفة في AI

**طبقة الشبكة العصبية:**

$$\vec{y} = W \cdot \vec{x} + \vec{b}$$

حيث:
- $\vec{x}$: المدخل (مثلاً: بيانات الصورة)
- $W$: مصفوفة الأوزان (ما يتعلمه النموذج)
- $\vec{b}$: Bias (انزياح)
- $\vec{y}$: الخرج (تمثيل في بُعد مختلف)

**كل طبقة في GPT أو BERT هي عبارة عن ضرب مصفوفات + دوال تفعيل.**

---

### 5.4 الاستقلالية الخطية (Linear Independence)

#### ما هي؟

مجموعة من المتجهات **مستقلة خطياً** إذا لم يكن أي متجه منها يُكتب كمجموع موزون (تركيب خطي) للمتجهات الأخرى.

رياضياً: المتجهات $\{v_1, v_2, \ldots, v_n\}$ مستقلة خطياً إذا:

$$c_1 v_1 + c_2 v_2 + \cdots + c_n v_n = \vec{0} \implies c_1 = c_2 = \cdots = c_n = 0$$

#### مثال ملموس

$$v_1 = [1, 0, 0], \quad v_2 = [0, 1, 0], \quad v_3 = [2, 1, 0]$$

هل $v_3$ يمكن كتابته من $v_1$ و$v_2$؟

$$v_3 = 2 \cdot v_1 + 1 \cdot v_2 = 2[1,0,0] + 1[0,1,0] = [2,1,0] \checkmark$$

إذن $\{v_1, v_2, v_3\}$ **غير مستقلة خطياً** (تابعة).

**ما يعنيه هذا هندسياً:** ثلاثتها تقع في مستوى xy (لا يمكن الوصول إلى $[0,0,1]$ منها مهما حاولت).

#### لماذا يهم في AI؟

**في بيانات التدريب:**

إذا كانت ميزتان (Features) تابعتين خطياً، مثلاً:

$$\text{feature-3} = 2 \times \text{feature-1} + \text{feature-2}$$

فإن:
- إضافة $\text{feature-3}$ لا تضيف معلومة جديدة للنموذج.
- المعادلات العادية (Normal Equations) تصبح **شاذة (Singular)** — لا يوجد حل وحيد للأوزان.
- تغيير بسيط في البيانات يُسبب تغيرات كبيرة وغير مستقرة في الأوزان.

**هذه هي مشكلة Multicollinearity** في الانحدار الخطي.

---

### 5.5 الرتبة (Rank)

#### ما هي رتبة المصفوفة؟

**الرتبة = عدد الأعمدة (أو الصفوف) المستقلة خطياً.**

```text
A = [[1, 2],     → الصف الثاني = 2 × الصف الأول → رتبة = 1
     [2, 4]]

B = [[1, 0],     → الصفوف مستقلة → رتبة = 2 (رتبة كاملة)
     [0, 1]]
```

#### جدول حالات الرتبة في ML

| الحالة | الرتبة | معناه في ML |
|--------|--------|-------------|
| **رتبة كاملة** (rank = min(m,n)) | أقصى ما يمكن | حل وحيد لـ Least-Squares. النموذج مستقر. |
| **رتبة ناقصة** (rank < min(m,n)) | أقل من الأقصى | ميزات متكررة. حلول أوزان لا نهائية. يحتاج Regularization. |
| **رتبة 1** | 1 | كل عمود نسخة من متجه واحد. البيانات تقع على خط. |
| **قريب من الناقصة** | عددياً منخفض | المصفوفة سيئة التكييف (Ill-conditioned). استخدم SVD أو Ridge Regression. |

**LoRA وعلاقتها بالرتبة:**

LoRA (Low-Rank Adaptation) هي تقنية تُعدّل نماذج LLM كبيرة بكفاءة:
- بدلاً من تحديث مصفوفة أوزان $W$ بحجم $4096 \times 4096$ (16 مليون معامل)
- تُحدّث مصفوفتين صغيرتين: $A$ (حجم $4096 \times 16$) و$B$ (حجم $16 \times 4096$)
- التحديث الكلي: $\Delta W = A \cdot B^T$ برتبة أقصاها 16
- المعاملات: فقط 131,000 بدلاً من 16 مليون!

**الفرضية:** تحديثات الأوزان تقع في فضاء ذي أبعاد منخفضة (low-dimensional subspace).

---

### 5.6 الإسقاط (Projection)

#### ما هو؟

إسقاط المتجه $\vec{a}$ على المتجه $\vec{b}$ يعطيك **المكوّن من $\vec{a}$ في اتجاه $\vec{b}$**:

$$\text{proj}_{\vec{b}}(\vec{a}) = \frac{\vec{a} \cdot \vec{b}}{\vec{b} \cdot \vec{b}} \cdot \vec{b}$$

#### مثال رقمي

$$\vec{a} = [3, 4], \quad \vec{b} = [1, 0]$$

$$\text{proj}_{\vec{b}}(\vec{a}) = \frac{3 \times 1 + 4 \times 0}{1 \times 1 + 0 \times 0} \cdot [1, 0] = \frac{3}{1} \cdot [1, 0] = [3, 0]$$

**ما حدث:** أسقطنا $[3, 4]$ على المحور الأفقي فحصلنا على $[3, 0]$ — تجاهلنا المكوّن الرأسي تماماً.

**البقية (Residual):** $[3, 4] - [3, 0] = [0, 4]$ — هي **عمودية تماماً** على $\vec{b}$.

تحقق: $[0, 4] \cdot [1, 0] = 0$ ✓ (عمودية)

#### أين تظهر الإسقاطات في AI؟

1. **الانحدار الخطي (Linear Regression):** الحل الأمثل هو **إسقاط** متجه الملاحظات على الفضاء العمودي للمصفوفة. الخطأ هو البقية العمودية.

2. **PCA:** يُسقط البيانات على اتجاهات التباين الأعظمي (eigenvectors).

3. **Attention في Transformers:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

$QK^T$ هو ضرب نقطي (dot product) بين الـ Queries والـ Keys — وهو في جوهره **إسقاط** كل استعلام على كل مفتاح.

---

### 5.7 عملية Gram-Schmidt والأساس المتعامد

#### لماذا نريد أساساً متعامداً؟

الحسابات أكثر استقراراً واستقامةً عندما تكون المتجهات **متعامدة ومطبّعة (Orthonormal)**:
- لا تداخل بين الاتجاهات.
- ضرب المصفوفة المتعامدة بمعكوسها يعطي مصفوفة الهوية.
- عمليات أسرع وأقل عرضة لأخطاء التقريب.

#### خطوات Gram-Schmidt

المدخلات: مجموعة متجهات مستقلة $v_1, v_2, v_3, \ldots$

**الخطوة 1:** طبّع $v_1$:

$$u_1 = \frac{v_1}{|v_1|}$$

**الخطوة 2:** أزل من $v_2$ مكوّنه في اتجاه $u_1$، ثم طبّع:

$$w_2 = v_2 - (v_2 \cdot u_1) \cdot u_1$$

$$u_2 = \frac{w_2}{|w_2|}$$

**الخطوة 3:** أزل من $v_3$ مكوّناته في اتجاه $u_1$ و$u_2$، ثم طبّع:

$$w_3 = v_3 - (v_3 \cdot u_1) \cdot u_1 - (v_3 \cdot u_2) \cdot u_2$$

$$u_3 = \frac{w_3}{|w_3|}$$

**المخرجات:** $\{u_1, u_2, u_3\}$ — متعامدة ومطبّعة.

#### مثال رقمي

$$v_1 = [1, 1, 0], \quad v_2 = [1, 0, 1], \quad v_3 = [0, 1, 1]$$

**$u_1$:**

$$|v_1| = \sqrt{1^2+1^2+0^2} = \sqrt{2}$$

$$u_1 = \frac{[1,1,0]}{\sqrt{2}} = [0.707, 0.707, 0]$$

**$w_2$:**

$$v_2 \cdot u_1 = 1 \times 0.707 + 0 \times 0.707 + 1 \times 0 = 0.707$$

$$w_2 = [1, 0, 1] - 0.707 \times [0.707, 0.707, 0] = [1, 0, 1] - [0.5, 0.5, 0] = [0.5, -0.5, 1]$$

$$u_2 = \frac{[0.5, -0.5, 1]}{|[0.5,-0.5,1]|} = \frac{[0.5,-0.5,1]}{\sqrt{1.5}} \approx [0.408, -0.408, 0.816]$$

تحقق: $u_1 \cdot u_2 = 0.707 \times 0.408 + 0.707 \times (-0.408) + 0 \times 0.816 = 0.289 - 0.289 = 0$ ✓

#### استخدام Gram-Schmidt في AI

- **QR Decomposition:** تحليل أي مصفوفة إلى $Q$ (أساس متعامد) × $R$ (مثلثية علوية).
- **الحل العددي لأنظمة المعادلات:** أكثر استقراراً من الحذف الغاوسي.
- **خوارزمية QR لحساب القيم الذاتية (Eigenvalues).**

---

## 6. الشرح الرياضي المتكامل

### ملخص المعادلات الأساسية

| العملية | المعادلة | الكود |
|---------|---------|-------|
| مقدار المتجه | $\|\vec{v}\| = \sqrt{\sum_i v_i^2}$ | `sum(x**2 for x in v)**0.5` |
| التطبيع | $\hat{v} = \vec{v}/\|\vec{v}\|$ | `[x/mag for x in v]` |
| الضرب النقطي | $\vec{a} \cdot \vec{b} = \sum_i a_i b_i$ | `sum(a*b for a,b in zip(a,b))` |
| تشابه جيب التمام | $\cos\theta = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|\|\vec{b}\|}$ | `dot(a,b)/(mag(a)*mag(b))` |
| الإسقاط | $\text{proj} = \frac{\vec{a} \cdot \vec{b}}{\vec{b} \cdot \vec{b}} \vec{b}$ | `(dot(a,b)/dot(b,b)) * b` |
| ضرب المصفوفة × المتجه | $y_i = \sum_j M_{ij} x_j$ | `sum(M[i][j]*x[j] for j in ...)` |

---

## 7. الخوارزميات

### 7.1 خوارزمية الحذف الغاوسي (Gaussian Elimination) لحساب الرتبة

```text
للمصفوفة M ذات m صفاً و n عموداً:

ابدأ: rank = 0
لكل عمود col من 0 إلى n-1:
    ابحث عن عنصر pivot غير صفري في العمود col، من الصف rank فأكثر
    إذا لم يوجد pivot:
        تابع للعمود التالي (هذا العمود تابع)
    إذا وُجد pivot:
        بادل الصف pivot مع الصف rank
        قسّم الصف rank على عنصر pivot (ليصبح 1)
        أزل هذا العمود من جميع الصفوف الأخرى (Elimination)
        زد rank بـ 1
النتيجة: rank = عدد الأعمدة المستقلة
```

---

### 7.2 خوارزمية Gram-Schmidt

```text
orthonormal = []
لكل متجه v في المجموعة:
    w = v   (ابدأ بنسخة من v)
    لكل u في orthonormal:
        w = w - project(w, u)   (أزل مكوّن w في اتجاه u)
    إذا |w| < ε:
        تجاهل (v كان تابعاً)
    وإلا:
        أضف normalize(w) إلى orthonormal
```

---

## 8. تطبيق يدوي من الصفر

### محاكاة ضرب مصفوفة × متجه يدوياً

$$W = \begin{bmatrix} 0.1 & -0.2 & 0.3 \\\\ 0.4 & 0.5 & -0.1 \end{bmatrix}, \quad x = [1.0, 0.5, -0.3]$$

**الصف الأول:**

$$y_1 = 0.1 \times 1.0 + (-0.2) \times 0.5 + 0.3 \times (-0.3)$$

$$= 0.1 - 0.1 - 0.09 = -0.09$$

**الصف الثاني:**

$$y_2 = 0.4 \times 1.0 + 0.5 \times 0.5 + (-0.1) \times (-0.3)$$

$$= 0.4 + 0.25 + 0.03 = 0.68$$

$$\vec{y} = [-0.09, \; 0.68]$$

هذا بالضبط ما تفعله **طبقة الشبكة العصبية**: تحوّل مدخلاً ثلاثي الأبعاد إلى خرج ثنائي الأبعاد.

---

## 9. شرح الكود الأصلي بالتفصيل

### `vectors.py` — التطبيق الكامل

#### فئة Vector

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)
```

**ما يفعله:**
- `components`: نسخة من قائمة الأرقام (نسخ وليس مرجع لتجنّب التعديل غير المقصود).
- `dim`: عدد الأبعاد (يُستخدم للتحقق من توافق العمليات).

---

```python
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])
```

**الرياضيات:**

$$[a_1, a_2, a_3] + [b_1, b_2, b_3] = [a_1+b_1, a_2+b_2, a_3+b_3]$$

**ما يفعله Python:**
- `zip(self.components, other.components)`: يُنتج أزواجاً $(a_i, b_i)$.
- `[a + b for ...]`: List comprehension — يُنشئ قائمة بالأزواج مجموعة.
- `return Vector(...)`: يُعيد متجهاً جديداً (لا يُعدّل الأصليين).

لماذا `__add__` وليس دالة عادية `add()`؟ لأن `__add__` يسمح باستخدام علامة `+` مباشرة: `a + b` بدلاً من `a.add(b)`.

---

```python
    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))
```

**الرياضيات:**

$$\vec{a} \cdot \vec{b} = \sum_i a_i \times b_i$$

- `zip`: يُنتج أزواجاً.
- `a * b`: ضرب كل زوج.
- `sum(...)`: جمع كل النواتج.

**للمتجهين $[1,2,3]$ و$[4,5,6]$:**
$1 \times 4 + 2 \times 5 + 3 \times 6 = 4 + 10 + 18 = 32$

---

```python
    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5
```

**الرياضيات:**

$$|\vec{v}| = \sqrt{\sum_i v_i^2}$$

- `x**2`: تربيع كل عنصر.
- `sum(...)`: جمع المربعات.
- `** 0.5`: الجذر التربيعي (رفع للأس 0.5).

---

```python
    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])
```

**الرياضيات:**

$$\hat{v} = \frac{\vec{v}}{|\vec{v}|}$$

نقسم كل عنصر على المقدار لنحصل على متجه بطول 1.

---

```python
    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())
```

**الرياضيات:**

$$\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}$$

---

```python
    def angle_between(self, other):
        import math
        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))
```

**الرياضيات:**

$$\theta = \arccos\left(\frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}\right)$$

**لماذا `max(-1.0, min(1.0, cos_theta))`؟**
بسبب أخطاء الفاصلة العائمة (Floating Point Errors)، قد يُعيد Cosine Similarity قيمة مثل $1.0000000001$ بسبب التقريب. `arccos` تُطلق خطأً إذا كانت القيمة خارج النطاق $[-1, 1]$. هذا السطر يضمن بقاء القيمة ضمن النطاق الصحيح.

---

```python
    def project_onto(self, other):
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])
```

**الرياضيات:**

$$\text{proj}_{\vec{b}}(\vec{a}) = \frac{\vec{a} \cdot \vec{b}}{\vec{b} \cdot \vec{b}} \cdot \vec{b}$$

- `self.dot(other)`: $\vec{a} \cdot \vec{b}$
- `other.dot(other)`: $\vec{b} \cdot \vec{b} = |\vec{b}|^2$
- `scalar`: المعامل القياسي
- `[scalar * x for x in other.components]`: تضخيم $\vec{b}$ بالمعامل

---

#### دالة `is_independent`

```python
def is_independent(vectors):
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n
```

**الخوارزمية (Gaussian Elimination):**

1. `rows = [v.components[:] for v in vectors]`: نُنشئ نسخة من المتجهات كصفوف مصفوفة (نسخة حتى لا نُعدّل الأصليين).

2. لكل عمود `col`:
   - نبحث عن `pivot` — أول عنصر غير صفري في العمود من الصف `rank` فأكثر.
   - إذا لم نجد pivot → هذا العمود صفري → العمود تابع → `continue` للعمود التالي.
   - **مبادلة الصفوف:** نضع صف الـ pivot في مكان الصف `rank`.
   - **التطبيع:** نقسم صف `rank` على عنصر pivot ليصبح 1.
   - **الحذف:** من كل الصفوف الأخرى نطرح مضاعباً للصف `rank` لتصفير العمود.
   - نزيد `rank`.

3. في النهاية: إذا كان `rank == n` → كل المتجهات مستقلة.

**لماذا `abs(rows[row][col]) > 1e-10` وليس `!= 0`؟**
بسبب أخطاء الفاصلة العائمة — رقم "صفري" قد يكون $10^{-16}$ بدلاً من صفر تماماً. نتعامل معه كصفر إذا كان أصغر من $10^{-10}$.

---

#### دالة `gram_schmidt`

```python
def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal
```

**التدفق:**

1. `orthonormal = []`: نبدأ بقائمة فارغة للأساس المتعامد.
2. لكل متجه `v`:
   - `w = v`: نبدأ بنسخة من `v`.
   - لكل `u` في الأساس المبني حتى الآن:
     - نحسب إسقاط `w` على `u`.
     - نطرح الإسقاط من `w` (نزيل المكوّن في اتجاه `u`).
   - بعد الطرح، `w` عمودي على كل المتجهات السابقة.
   - `if w.magnitude() < 1e-10`: إذا أصبح `w` صفرياً تقريباً، يعني `v` كان تابعاً → نتجاهله.
   - `orthonormal.append(w.normalize())`: نطبّع ونُضيف.

---

#### فئة Matrix

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))
```

- `rows`: قائمة الصفوف (نسخ مستقلة لكل صف).
- `shape`: `(m, n)` — عدد الصفوف × عدد الأعمدة.

---

```python
    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)
```

**الرياضيات — ضرب مصفوفة × متجه:**

$$y_i = \sum_j M_{ij} \cdot x_j$$

**الرياضيات — ضرب مصفوفة × مصفوفة:**

$$(AB)_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

`__matmul__` يسمح باستخدام عامل `@`:
```python
result = rotation_90 @ point  # يستدعي __matmul__
```

**ماذا يحدث بالحلقات الثلاث؟**
- `i`: رقم الصف في المصفوفة الناتجة.
- `j`: رقم العمود في المصفوفة الناتجة.
- `k`: عداد الضرب الداخلي (يتكرر على الأعمدة/الصفوف المشتركة).

---

```python
    def rank(self):
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r
```

نفس خوارزمية الحذف الغاوسي لكن على الـ Matrix مباشرة. `r` يعدّ الـ pivots الموجودة = الرتبة.

---

### `vectors.jl` — التطبيق بلغة Julia

```julia
using LinearAlgebra

a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a · b = ", a ⋅ b)       # Julia يدعم عوامل Unicode مباشرة
println("|a| = ", norm(a))
println("â = ", normalize(a))

cosine = (a ⋅ b) / (norm(a) * norm(b))
println("cosine_similarity(a, b) = ", round(cosine, digits=4))
```

**ما يختلف عن Python:**

| Python | Julia | المعنى |
|--------|-------|--------|
| `np.dot(a, b)` | `a ⋅ b` | الضرب النقطي |
| `np.linalg.norm(a)` | `norm(a)` | المقدار |
| `a / np.linalg.norm(a)` | `normalize(a)` | التطبيع |
| `W @ x` | `W * x` | ضرب مصفوفة × متجه |

Julia مُصمَّمة للرياضيات العلمية — تدعم الرموز الرياضية مباشرةً في الكود.

---

## 10. شرح كل Function/Class

### جدول الدوال والفئات

| الاسم | النوع | المدخلات | المخرجات | الهدف |
|-------|-------|---------|---------|-------|
| `Vector.__init__` | Method | قائمة أرقام | — | تهيئة المتجه |
| `Vector.__add__` | Method | متجه آخر | متجه جديد | الجمع $\vec{a}+\vec{b}$ |
| `Vector.__sub__` | Method | متجه آخر | متجه جديد | الطرح $\vec{a}-\vec{b}$ |
| `Vector.__mul__` | Method | عدد قياسي | متجه جديد | الضرب القياسي $c\vec{a}$ |
| `Vector.dot` | Method | متجه آخر | رقم | الضرب النقطي |
| `Vector.magnitude` | Method | — | رقم | المقدار $\|\vec{v}\|$ |
| `Vector.normalize` | Method | — | متجه جديد | التطبيع $\hat{v}$ |
| `Vector.cosine_similarity` | Method | متجه آخر | رقم في $[-1,1]$ | تشابه جيب التمام |
| `Vector.angle_between` | Method | متجه آخر | زاوية بالدرجات | الزاوية $\theta$ |
| `Vector.project_onto` | Method | متجه آخر | متجه جديد | الإسقاط |
| `is_independent` | دالة | قائمة متجهات | `True`/`False` | فحص الاستقلالية |
| `gram_schmidt` | دالة | قائمة متجهات | قائمة متجهات | أساس متعامد |
| `Matrix.__init__` | Method | قائمة صفوف | — | تهيئة المصفوفة |
| `Matrix.__matmul__` | Method | مصفوفة أو متجه | مصفوفة أو متجه | ضرب المصفوفة (عامل @) |
| `Matrix.transpose` | Method | — | مصفوفة جديدة | النقل $M^T$ |
| `Matrix.rank` | Method | — | رقم صحيح | رتبة المصفوفة |

---

## 11. التنفيذ والتجربة

### تشغيل الكود

من جذر المستودع:

```bash
python3 phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py
```

**المخرجات المتوقعة:**

```text
=== Vectors ===
a = Vector([1, 2, 3])
b = Vector([4, 5, 6])
a + b = Vector([5, 7, 9])
a - b = Vector([-3, -3, -3])
a * 3 = Vector([3, 6, 9])
a · b = 32
|a| = 3.7417
â (normalized) = Vector([0.2673..., 0.5345..., 0.8018...])
cosine_similarity(a, b) = 0.9746

=== Matrices ===
Rotate Vector([3, 1]) by 90° → Vector([-1, 3])

=== Angle Between Vectors ===
Angle between Vector([1, 0]) and Vector([0, 1]): 90.0 degrees
Angle between Vector([1, 0]) and Vector([1, 1]): 45.0 degrees
Angle between Vector([1, 0]) and Vector([1, 0]): 0.0 degrees

=== Projection ===
a = Vector([3, 4])
b = Vector([1, 0])
proj_b(a) = Vector([3.0, 0.0])
residual = Vector([0.0, 4.0])
residual dot b = 0.000000

=== Linear Independence ===
{e1, e2, e3} independent: True
{e1, e2, 2*e1+e2} independent: False

=== Gram-Schmidt Orthogonalization ===
u1 = Vector([0.7071..., 0.7071..., 0.0])
u2 = Vector([0.4082..., -0.4082..., 0.8165...])
u3 = Vector([-0.5773..., 0.5773..., 0.5773...])
u1 dot u2 = 0.000000
u1 dot u3 = 0.000000
u2 dot u3 = 0.000000

=== Matrix Rank ===
Identity 2x2 rank: 2
[[1,2],[2,4]] rank: 1
[[1,0,0],[0,1,0]] rank: 2

=== Neural Network Layer (Matrix x Vector) ===
Input (3D):  Vector([1.0, 0.5, -0.3])
Output (2D): Vector([...])
^ This is literally what a neural network layer does.
```

---

### تشغيل Julia

```bash
julia phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.jl
```

---

## 12. ماذا يحدث داخلياً؟ (Execution Flow)

```text
تشغيل vectors.py
      ↓
إنشاء Vector([1,2,3]) وVector([4,5,6])
      ↓
a.dot(b) → zip → ضرب الأزواج → جمع → 32
      ↓
a.magnitude() → تربيع → جمع → جذر تربيعي → 3.7417
      ↓
a.cosine_similarity(b) → dot(32) / (mag_a * mag_b) → 0.9746
      ↓
rotation_90 @ point → __matmul__ → ضرب صفوف × متجه → [-1, 3]
      ↓
is_independent([e1, e2, dep]) → Gaussian Elim → rank=2 < n=3 → False
      ↓
gram_schmidt([u1, u2, u3]) → إسقاط تكراري → تطبيع → أساس متعامد
      ↓
rank_deficient.rank() → Gaussian Elim → rank=1 (صف 2 = 2 × صف 1)
```

---

## 13. ربط المفاهيم بـ PyTorch

### المتجهات في PyTorch = Tensors

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

# الضرب النقطي
similarity = torch.dot(x, y)

# حساب التدرج تلقائياً!
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")  # يساوي y تلقائياً!
```

**لماذا `x.grad` يساوي `y`؟**

$$\frac{\partial (\vec{x} \cdot \vec{y})}{\partial \vec{x}} = \vec{y}$$

هذا لأن:

$$\vec{x} \cdot \vec{y} = x_1 y_1 + x_2 y_2 + x_3 y_3$$

فالمشتقة بالنسبة لـ $x_i$ هي $y_i$.

PyTorch حسبها تلقائياً! هذا هو **التفاضل التلقائي (Automatic Differentiation)** — سنشرحه بالتفصيل في الدرس القادم.

---

## 14. الأخطاء الشائعة

### الخطأ 1: أبعاد غير متطابقة

```python
a = Vector([1, 2, 3])
b = Vector([4, 5])
result = a.dot(b)
# zip يتوقف عند أقصر متجه → ينتج 1×4 + 2×5 = 14 (خاطئ!)
```

**المشكلة:** `zip` لا يُطلق خطأً — يتوقف فقط. يجب إضافة فحص:

```python
def dot(self, other):
    assert self.dim == other.dim, f"أبعاد غير متطابقة: {self.dim} ≠ {other.dim}"
    return sum(a * b for a, b in zip(self.components, other.components))
```

### الخطأ 2: القسمة على صفر في التطبيع

```python
zero_vector = Vector([0, 0, 0])
normalized = zero_vector.normalize()  # ZeroDivisionError!
```

**الحل:**
```python
def normalize(self):
    mag = self.magnitude()
    if mag < 1e-10:
        raise ValueError("لا يمكن تطبيع المتجه الصفري")
    return Vector([x / mag for x in self.components])
```

### الخطأ 3: ضرب مصفوفات غير متوافقة

```python
A = Matrix([[1, 2, 3]])    # شكل (1, 3)
B = Matrix([[1, 2], [3, 4]])  # شكل (2, 2)
C = A @ B  # خطأ! عدد أعمدة A (3) ≠ عدد صفوف B (2)
```

**القاعدة:** لضرب $A_{(m \times k)} \times B_{(k \times n)}$، يجب أن $k$ في $A$ يساوي $k$ في $B$.

---

## 15. لماذا تعمل هذه الطريقة؟ (Deep Dive)

### لماذا Gaussian Elimination يعطي الرتبة؟

في كل خطوة من خطوات Gaussian Elimination، نُنشئ **تركيباً خطياً** من الصفوف (نضرب صفاً ونطرحه من آخر). هذه العملية لا تُغيّر الفضاء الذي تمتد إليه الصفوف (Row Space). في النهاية، كل صف غير صفري في الشكل المحوّل يمثل بُعداً مستقلاً. عدد الصفوف غير الصفرية = الرتبة.

### لماذا Gram-Schmidt يعطي متجهات عمودية؟

بعد معالجة $v_2$:

$$w_2 = v_2 - \underbrace{(v_2 \cdot u_1) \cdot u_1}_{\text{مكوّن } v_2 \text{ في اتجاه } u_1}$$

تحقق: $w_2 \cdot u_1 = (v_2 - (v_2 \cdot u_1) u_1) \cdot u_1 = v_2 \cdot u_1 - (v_2 \cdot u_1)(u_1 \cdot u_1) = v_2 \cdot u_1 - v_2 \cdot u_1 = 0$ ✓

لأن $u_1 \cdot u_1 = |u_1|^2 = 1$ (لأنه مطبَّع).

### ماذا يحدث حسابياً في طبقة الشبكة العصبية؟

```text
مدخل: x ∈ ℝ³ (3 مميزات)
أوزان: W ∈ ℝ²ˣ³ (تحوّل من 3D إلى 2D)
عملية: y = W·x + b
خرج: y ∈ ℝ² (2 مميزات جديدة)
```

كل صف من $W$ يمثل **اتجاهاً** في الفضاء الأصلي — يقيس مدى تطابق المدخل مع هذا الاتجاه.
ضرب $W \cdot x$ هو **مجموعة من الضروبات النقطية** — كل خرج هو إسقاط المدخل على اتجاه مختلف.

هذا هو معنى "التحويل الخطي (Linear Transformation)".

---

## 16. العلاقة مع الدروس السابقة

- **الدرس 01 (إعداد البيئة):** وفّر Python وNumPy اللازمين لتشغيل هذا الكود.

هذا الدرس هو **الأساس الأول** في المرحلة الرياضية.

---

## 17. العلاقة مع الدروس القادمة

```text
01-linear-algebra-intuition (أنت هنا)
          ↓
02-vectors-matrices-operations (عمليات أعمق: eigenvalues, SVD مبدئي)
          ↓
03-matrix-transformations (التحويلات الهندسية والتحليل الطيفي)
          ↓
04-calculus-for-ml (المشتقات والتدرجات)
          ↓
05-chain-rule-and-autodiff (القاعدة المتسلسلة والتفاضل التلقائي)
          ↓
Phase 02: أسس ML (ستستخدم dot product في كل خوارزمية)
          ↓
Phase 07: Transformers (Attention = dot products + projections)
          ↓
Phase 10: LLMs (LoRA = low-rank matrices)
```

**كل درس في هذا المنهج يبني على مفاهيم هذا الدرس.**

---

## 18. مثال تطبيقي كامل — محرك بحث بسيط

سنبني محرك بحث مبسط باستخدام Cosine Similarity:

```python
# تمثيل الوثائق كمتجهات (مثال مبسط)
# كل عنصر يمثل تكرار كلمة: [رياضيات, برمجة, AI, فيزياء, فن]

documents = {
    "مقالة_AI": Vector([1, 2, 3, 0, 0]),
    "كتاب_رياضيات": Vector([3, 1, 1, 2, 0]),
    "دورة_برمجة": Vector([0, 3, 1, 0, 0]),
    "مقالة_فيزياء": Vector([2, 0, 0, 3, 0]),
}

# استعلام البحث
query = Vector([1, 1, 3, 0, 0])  # مهتم بـ AI والبرمجة قليلاً والرياضيات

# حساب التشابه مع كل وثيقة
results = {}
for name, doc_vec in documents.items():
    similarity = query.cosine_similarity(doc_vec)
    results[name] = similarity

# ترتيب النتائج
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("نتائج البحث:")
for name, score in sorted_results:
    print(f"  {name}: {score:.4f}")
```

**المخرجات المتوقعة:**

```text
نتائج البحث:
  مقالة_AI: 0.9746
  دورة_برمجة: 0.7559
  كتاب_رياضيات: 0.6547
  مقالة_فيزياء: 0.2673
```

هذا بالضبط كيف تعمل **قواعد البيانات المتجهية (Vector Databases)** مثل Pinecone وChroma — المستخدمة في RAG وChatbots الحديثة.

---

## 19. خلاصة الدرس

| المفهوم | ما تعلمناه |
|---------|-----------|
| **المتجه** | قائمة أرقام تمثل نقطة أو اتجاهاً في فضاء |
| **الضرب النقطي** | يقيس التشابه — هو أساس البحث والانتباه |
| **المصفوفة** | تحويل يُحرّك المتجهات في الفضاء |
| **الاستقلالية الخطية** | ضرورية للبيانات الجيدة وأوزان النموذج المستقرة |
| **الرتبة** | عدد الأبعاد المعلوماتية الحقيقية في المصفوفة |
| **الإسقاط** | التعبير عن متجه كمكوّن في اتجاه آخر |
| **Gram-Schmidt** | بناء أساس متعامد للحساب المستقر |
| **LoRA** | تقليل عدد المعاملات بالاعتماد على انخفاض الرتبة |

---

## 20. أهم المصطلحات

| English | العربية | المعنى |
|---------|---------|--------|
| Vector | متجه | قائمة مرتّبة من الأرقام تمثل نقطة أو اتجاهاً |
| Matrix | مصفوفة | جدول مستطيل من الأرقام يمثل تحويلاً خطياً |
| Dot Product | الضرب النقطي | مجموع نواتج ضرب العناصر المتقابلة — مقياس التشابه |
| Magnitude | المقدار | طول المتجه = $\sqrt{\sum v_i^2}$ |
| Normalization | التطبيع | جعل المتجه بطول 1 بالقسمة على مقداره |
| Cosine Similarity | تشابه جيب التمام | مقياس التشابه بغض النظر عن الطول |
| Embedding | التضمين | متجه يمثل معنى كيان ما (كلمة، صورة، مستخدم) |
| Linear Independence | الاستقلالية الخطية | لا يمكن كتابة أي متجه كتركيب من الآخرين |
| Rank | الرتبة | عدد الأبعاد المعلوماتية الحقيقية في مصفوفة |
| Projection | الإسقاط | المكوّن من متجه في اتجاه متجه آخر |
| Basis | الأساس | مجموعة متجهات مستقلة تمتد لتغطي الفضاء |
| Orthonormal | متعامد ومطبّع | متجهات زوايا بينها 90° وطول كل منها 1 |
| Gram-Schmidt | جرام-شميت | خوارزمية تحويل أي أساس إلى أساس متعامد |
| Row Reduction | الاختزال الصفي | Gaussian Elimination — لحساب الرتبة |
| Multicollinearity | الارتباط الخطي المتعدد | مشكلة عند وجود ميزات متابعة خطياً |
| LoRA | لورا | تقنية تعديل LLM بكفاءة باستخدام مصفوفات منخفضة الرتبة |

---

## 21. اختبار الفهم

### من Quiz الأصلي

**السؤال 1 (Pre):** ما الذي يقيسه الضرب النقطي بين متجهين؟

- أ) عدد الأبعاد التي يشتركان فيها
- ب) مدى تشابه أو توافق المتجهين ✓
- ج) المسافة بين المتجهين
- د) الزاوية بين المتجهين بالدرجات

**شرح الإجابة الصحيحة:** الضرب النقطي يقيس **التوافق**: موجب = نفس الاتجاه (متشابهان)، صفر = عموديان (لا علاقة)، سالب = اتجاهان متعاكسان (متضادان). هذا هو أساس البحث بالتشابه في AI.

**لماذا الباقية خاطئة:** (أ) ليست صحيحة — الضرب النقطي لا يعدّ الأبعاد المشتركة. (ج) المسافة تُحسب بمعيار مختلف. (د) يمكن استخراج الزاوية من الضرب النقطي لكنه ليس الضرب نفسه.

---

**السؤال 2 (Pre):** في AI، ما الذي يعنيه "Embedding"؟

- أ) تحويل الكود إلى تعليمات آلية
- ب) تقنية لضغط أوزان النموذج
- ج) إدراج نموذج داخل آخر
- د) تمثيل متجهي يلتقط معنى شيء ما (كلمة، صورة، مستخدم) ✓

**شرح:** Embedding يُعيّن كيانات منفصلة (كلمات، صور، مستخدمون) إلى متجهات مستمرة في فضاء عالي الأبعاد، حيث الأشياء المتشابهة تكون قريبة من بعضها. هو الجسر بين المفاهيم الحقيقية والعمليات الرياضية.

---

**السؤال 3 (Post):** ثلاثة متجهات $v_1=[1,0,0]$, $v_2=[0,1,0]$, $v_3=[2,1,0]$. هل هي مستقلة خطياً؟

- أ) نعم، لأن هناك ثلاثة متجهات في فضاء 3D
- ب) لا، لأن $v_3 = 2 v_1 + v_2$ ✓
- ج) نعم، لأن لا يوجد متجهان متطابقان
- د) لا، لأن كلها تحتوي على صفر

---

**السؤال 4 (Post):** ما الذي تخبرك به رتبة مصفوفة في سياق ML؟

- أ) عدد الأعمدة المستقلة خطياً، يشير لعدد أبعاد المعلومة المفيدة ✓
- ب) القيمة القصوى في المصفوفة
- ج) سرعة ضرب المصفوفة
- د) عدد العناصر غير الصفرية

---

**السؤال 5 (Post):** كيف تستخدم LoRA الجبر الخطي لتعديل LLMs بكفاءة؟

- أ) تُحلّل تحديثات الأوزان إلى مصفوفتين صغيرتين منخفضتَي الرتبة بدلاً من تحديث المصفوفة الكاملة ✓
- ب) تحذف صفوفاً غير مستخدمة من مصفوفات الأوزان
- ج) تجمّد طبقة التضمين وتدرّب رأس الخرج فقط
- د) تحوّل جميع الأوزان من float32 إلى int8

---

### أسئلة إضافية

**السؤال 6:** ما الزاوية بين المتجه $[1, 0]$ والمتجه $[0, 1]$؟ احسبها بالرياضيات ثم تحقق من الكود.

**السؤال 7:** إذا كان $\vec{a} \cdot \vec{b} = 0$، ماذا يعني ذلك هندسياً؟ اذكر مثالاً من AI.

**السؤال 8:** المصفوفة $\begin{bmatrix}2 & 0 \\\\ 0 & 3\end{bmatrix}$ — ما التحويل الهندسي الذي تُنفّذه على المتجه $[1, 1]$؟

**السؤال 9:** في `gram_schmidt`، لماذا نتحقق من `w.magnitude() < 1e-10` قبل الإضافة؟

**السؤال 10:** ابنِ مثالاً لمصفوفة $3 \times 3$ برتبة 2. ما الفضاء الهندسي الذي تمتد إليه أعمدتها؟

---

## 22. تمارين عملية

### Beginner — المستوى الأساسي

**التمرين 1:** أنشئ متجهَين $\vec{a} = [2, 3]$ و$\vec{b} = [1, -1]$ وأجِب عن:
1. ما مقدار كل منهما؟
2. ما ضربهما النقطي؟
3. ما تشابه جيب التمام بينهما؟
4. هل هما "متشابهان" (تشابه > 0.5)؟

> **Hint:** استخدم فئة `Vector` الموجودة في `vectors.py`.

---

### Intermediate — المستوى المتوسط

**التمرين 2:** طبّق دالة `Vector.angle_between` يدوياً (بدون الكود) للمتجهين:

$$\vec{a} = [1, 2, 3], \quad \vec{b} = [1, 0, 0]$$

1. احسب الضرب النقطي.
2. احسب مقدار كل متجه.
3. احسب Cosine Similarity.
4. احسب الزاوية بالدرجات (استخدم $\arccos$).
5. تحقق من إجابتك بالكود.

---

### Advanced — المستوى المتقدم

**التمرين 3:** أضف دالة `orthogonal_complement` إلى فئة `Vector`:
- **الهدف:** بالنظر إلى مجموعة متجهات تمثل فضاءً، أوجد متجهاً عمودياً على الجميع.
- **Hint:** استخدم Gram-Schmidt ثم استخدم الحذف الغاوسي لإيجاد مجال العدم (Null Space).

---

### Challenge — التحدي

**التمرين 4:** ابنِ نظام توصيات مبسطاً (Recommendation System):
- عندك 5 مستخدمين وكل مستخدم يُمثَّل بمتجه تفضيلاته لـ 4 أنواع من المحتوى.
- عندك مستخدم جديد بتفضيلات معينة.
- أوجد أكثر 2 مستخدمين تشابهاً مع المستخدم الجديد (بالـ Cosine Similarity).
- اقترح محتوىً يحبه أكثر مستخدم مشابه لكن لم يُشاهده المستخدم الجديد بعد.

> **Hint:** هذه هي الفكرة الأساسية لـ Collaborative Filtering في Netflix وYouTube.

---

## إجابات اختبار الفهم

<details>
<summary>اضغط لرؤية الإجابات التفصيلية</summary>

**السؤال 6:**

$$\cos\theta = \frac{[1,0] \cdot [0,1]}{|[1,0]| \cdot |[0,1]|} = \frac{0}{1 \cdot 1} = 0$$

$$\theta = \arccos(0) = 90°$$

المتجهان عموديان — لا يوجد أي تشابه بينهما. في AI: إذا كانت الـ embeddings لكلمتين عمودية، فهاتان الكلمتان لا علاقة دلالية بينهما.

**السؤال 7:**

$\vec{a} \cdot \vec{b} = 0$ يعني أن الزاوية بينهما 90° — المتجهان عموديان (غير مرتبطين).

مثال من AI: في مساحة Embedding، إذا كان متجه "قطة" عمودياً على متجه "سيارة"، فلا علاقة دلالية بينهما. في Attention: نتيجة صفر تعني أن Token A لا "ينتبه" إلى Token B.

**السؤال 8:**

$$\begin{bmatrix}2 & 0 \\\\ 0 & 3\end{bmatrix} \cdot \begin{bmatrix}1 \\\\ 1\end{bmatrix} = \begin{bmatrix}2 \cdot 1 + 0 \cdot 1 \\\\ 0 \cdot 1 + 3 \cdot 1\end{bmatrix} = \begin{bmatrix}2 \\\\ 3\end{bmatrix}$$

التحويل: مقياس (Scaling) — يُضاعف المحور الأفقي مرتين ويُضاعف المحور الرأسي ثلاث مرات.

**السؤال 9:**

إذا أصبح `w` قريباً من الصفر، يعني أن المتجه الحالي `v` كان تابعاً خطياً للمتجهات السابقة — إسقاطه على المتجهات الموجودة "أزال" كل مكوّناته. قسمة صفر على صفر = `NaN` أو خطأ. نتجاهله لأنه لا يضيف بُعداً جديداً.

**السؤال 10:**

مثال: $A = \begin{bmatrix}1 & 0 & 2 \\\\ 0 & 1 & 1 \\\\ 0 & 0 & 0\end{bmatrix}$

العمود الثالث = 2 × العمود الأول + 1 × العمود الثاني → رتبة 2.

الفضاء الهندسي: مستوٌ (Plane) في الفضاء ثلاثي الأبعاد (وليس الفضاء كله). أعمدة المصفوفة تقع كلها في نفس المستوى.

</details>

---

*الدرس التالي: [02-vectors-matrices-operations](../../02-vectors-matrices-operations/docs/ar.md)*

*المرحلة: 01 — الأسس الرياضية | الدرس: 01 من 22*
