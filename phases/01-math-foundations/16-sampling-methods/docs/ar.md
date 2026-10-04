# طرق أخذ العينات (Sampling Methods)
> أخذ العينات هو الطريقة التي يستكشف بها الذكاء الاصطناعي فضاء الاحتمالات.

## 1. ما الذي سنتعلمه؟
- تنفيذ Inverse CDF، وطريقة الرفض (Rejection Sampling)، وImportance Sampling من الصفر.
- بناء آليات اختيار الكلمات لنماذج اللغات الكبيرة (LLMs): Temperature, Top-k, Top-p.
- فهم حيلة إعادة المعلمة (Reparameterization Trick) في الشبكات العصبية مثل VAEs.
- تشغيل خوارزمية Metropolis-Hastings التي تُعد من أشهر طرق Markov Chain Monte Carlo (MCMC).

## 2. لماذا هذا الموضوع مهم؟
عمليات أخذ العينات هي القلب النابض للعديد من تطبيقات الذكاء الاصطناعي:
- **في النماذج التوليدية (Generative Models):** لتوليد صور جديدة، أو نصوص متماسكة، نحتاج إلى سحب عينة من التوزيع الاحتمالي للنموذج.
- **في التعلم المعزز (RL):** لتقييم السياسات الجديدة نسبة إلى القديمة (Importance Sampling).
- **في الاستدلال البايزي (Bayesian Inference):** لتقدير التوزيعات المعقدة التي لا يمكن حلها تحليلياً.
بدون تقنيات أخذ العينات، لن نتمكن من تدريب نماذج الانتشار (Diffusion Models) أو توليد نصوص متنوعة وإبداعية بواسطة LLMs.

## 3. المتطلبات السابقة
- فهم الاحتمالات والتوزيعات (Distributions).
- التفاضل والتكامل الأساسي (حساب المساحات والمشتقات).
- مهارات البرمجة بلغة بايثون لبناء الخوارزميات من الصفر.

## 4. الفكرة الأساسية — Intuition
الحواسيب لا تستطيع توليد أرقام عشوائية إلا من توزيع منتظم (Uniform Distribution)، أي أرقام بين $0$ و $1$. لكي نحصل على بيانات معقدة (مثل صورة وجه إنسان، أو كلمة معينة في جملة)، يجب أن نقوم بـ "تحويل" هذه الأرقام المنتظمة البسيطة إلى توزيع معقد. عملية أخذ العينات هي بالضبط هذا التحويل المنهجي.

## 5. المفاهيم الأساسية والتفسير الرياضي

### 5.1 التحويل العكسي (Inverse CDF / Inverse Transform Sampling)
إذا كان لدينا متغير عشوائي منتظم $U \sim \text{Uniform}(0,1)$، فإننا يمكننا الحصول على عينة من التوزيع المستهدف عبر تطبيق الدالة العكسية للتوزيع التراكمي:
$$X = F^{-1}(U)$$
حيث $F(x)$ هي الـ CDF.

**مثال التوزيع الأسي (Exponential Distribution):**
دالة الكثافة (PDF): $p(x) = \lambda e^{-\lambda x}$
دالة التراكم (CDF): $F(x) = 1 - e^{-\lambda x}$
لإيجاد المعكوس، نساوي $F(x)$ بـ $u$:
$$u = 1 - e^{-\lambda x}$$
$$e^{-\lambda x} = 1 - u$$
$$-\lambda x = \ln(1 - u)$$
$$x = \frac{-\ln(1 - u)}{\lambda}$$
وبما أن $1-u$ له نفس التوزيع مثل $u$ (بين 0 و 1)، يمكننا التبسيط إلى $x = \frac{-\ln(u)}{\lambda}$.

### 5.2 طريقة الرفض (Rejection Sampling)
نستخدمها عندما يكون التوزيع المستهدف $p(x)$ معقداً، فنقترح توزيعاً بسيطاً $q(x)$ (مثل التوزيع الطبيعي) يمكننا السحب منه بسهولة. نختار ثابتاً $M$ بحيث يكون $M \cdot q(x) \geq p(x)$ لجميع قيم $x$.
**الخوارزمية:**
1. اسحب $x \sim q(x)$.
2. اسحب $u \sim \text{Uniform}(0,1)$.
3. اقبل العينة إذا كان: $u < \frac{p(x)}{M \cdot q(x)}$، وإلا ارفضها وأعد المحاولة.

**الرياضيات والقصور:**
معدل القبول هو $\frac{1}{M}$. في الأبعاد العالية، قيمة $M$ تزداد بشكل أسّي لتغطية التوزيع، مما يجعل معدل القبول يقترب من الصفر (لعنة الأبعاد).

### 5.3 أخذ عينات الأهمية (Importance Sampling)
لا نهدف هنا لتوليد عينات، بل لتقدير **القيمة المتوقعة** لدالة $f(x)$ تحت التوزيع $p(x)$ باستخدام عينات من التوزيع $q(x)$.
$$E_{p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = E_{q}\left[f(x) w(x)\right]$$
حيث $w(x) = \frac{p(x)}{q(x)}$ هو "وزن الأهمية".
**في PPO (التعلم المعزز):** نستخدم وزن الأهمية لتقييم السياسة الجديدة نسبة إلى القديمة: $\frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$.

### 5.4 مونت كارلو (Monte Carlo)
يستخدم العشوائية لتقدير القيم. التقدير:
$$E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i)$$
الخطأ يتناسب مع $O(1/\sqrt{N})$، والميزة الكبرى أنه **مستقل عن عدد الأبعاد**.

### 5.5 متروبوليس هاستينغز (Metropolis-Hastings MCMC)
لأخذ عينات من توزيع لا نعرف منه سوى $p(x)$ غير المعاير (بدون ثابت القسمة).
1. نبدأ عند $x_0$.
2. نقترح $x' \sim q(x'|x_t)$.
3. نحسب نسبة القبول:
$$\alpha = \frac{p(x') q(x_t|x')}{p(x_t) q(x'|x_t)}$$
4. نقبل $x'$ باحتمال $\min(1, \alpha)$.
**لماذا تعمل؟ (Detailed Balance):**
تضمن الخوارزمية خاصية التوازن الدقيق: $p(x) T(x \to x') = p(x') T(x' \to x)$، مما يضمن أن السلسلة ستستقر عند التوزيع المستهدف $p(x)$.

### 5.6 أخذ العينات في نماذج اللغة (LLMs)

#### 5.6.1 Temperature Sampling
تتحكم في "جرأة" أو "إبداع" النموذج. نحسب الاحتمالات كالتالي:
$$p_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$$
- $T \to 0$: يصبح اختيارات حتمية (Argmax).
- $T = 1$: يعطي Softmax العادي.
- $T > 1$: يزيد من التنوع ويقلل الفروق بين الكلمات (أكثر إبداعاً).

#### 5.6.2 Top-k و Top-p (Nucleus)
- **Top-k**: تختار أعلى $k$ كلمات من حيث الاحتمالية. مشكلته أنه ثابت، قد يقطع كلمات محتملة إذا كان التوزيع مشتتاً، أو يُدخل كلمات سيئة إذا كان التوزيع متركزاً.
- **Top-p**: ترتب الكلمات وتأخذ أقل عدد منها بحيث يكون مجموع احتمالاتها التراكمي $\geq p$. يتكيف ديناميكياً مع ثقة النموذج.

### 5.7 حيلة إعادة المعلمة (Reparameterization Trick) في VAEs
الشبكات العصبية تُدرب بـ Backpropagation. لا يمكننا حساب مشتقة لعينة عشوائية مثل $z \sim \mathcal{N}(\mu, \sigma^2)$.
**الحل:** نفصل العشوائية عن المعلمات.
1. نسحب $\epsilon \sim \mathcal{N}(0, 1)$ (لا توجد معلمات هنا).
2. نحسب: $z = \mu + \sigma \cdot \epsilon$
الآن $z$ دالة حتمية قابلة للاشتقاق:
$$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$
مما يسمح للمشتقات بالتدفق وتحديث $\mu$ و $\sigma$.

### 5.8 Gumbel-Softmax
النسخة المستمرة لأخذ العينات من توزيع متقطع (Categorical).
نضيف ضجيج Gumbel $g_i = -\log(-\log(u))$:
$$y_i = \frac{\exp((\log(p_i) + g_i)/\tau)}{\sum \exp((\log(p_j) + g_j)/\tau)}$$
يُستخدم في الذكاء الاصطناعي عندما نحتاج إلى قرار متقطع ولكن نريد تمرير المشتقات (Backprop).

## 6. الكود الأصلي بالتفصيل

```python
import math, random

# 1. Inverse CDF للتوزيع الأسي
def sample_exponential_inverse_cdf(lam):
    u = random.random() # سحب رقم منتظم بين 0 و 1
    return -math.log(u) / lam # تطبيق المعادلة العكسية الرياضية

# 2. طريقة الرفض
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample() # سحب عينة من التوزيع المقترح السهل
        u = random.random()
        # شرط القبول حسب نسبة الكثافات وثابت M
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x

# 3. Importance Sampling
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x) # حساب وزن الأهمية
        total += f(x) * w
    return total / n # المتوسط

# 4. حساب باي باستخدام مونت كارلو
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        # نقطة عشوائية في مربع 2x2
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1: # هل النقطة داخل الدائرة (نصف قطرها 1)؟
            inside += 1
    return 4 * inside / n # نسبة مساحة الدائرة للمربع

# 5. Metropolis-Hastings (MCMC)
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x) # اقتراح نقطة جديدة بناء على الحالية
        # حساب نسبة القبول (اللوغاريتم لتجنب الأرقام الصغيرة جدًا)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        # قبول النقطة الجديدة مع احتمال معين
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in: # تجاهل فترة الإحماء (Burn-in)
            samples.append(x)
    return samples

# دوال مساعدة للـ LLM Sampling
def softmax(logits):
    max_l = max(logits) # للثبات العددي (Numerical Stability)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def sample_from_probs(probs):
    # دالة مساعدة لاختيار فهرس بناءً على الاحتمالات
    u = random.random()
    cumsum = 0
    for i, p in enumerate(probs):
        cumsum += p
        if u <= cumsum:
            return i
    return len(probs) - 1

# 6. Temperature Sampling
def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits] # القسمة على درجة الحرارة
    probs = softmax(scaled)
    return sample_from_probs(probs)

# 7. Top-k Sampling
def top_k_sample(logits, k):
    # ترتيب تنازلي مع الاحتفاظ بالفهارس
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k] # أخذ أعلى k فقط
    top_logits = [l for _, l in top]
    probs = softmax(top_logits) # إعادة حساب الاحتمالات لـ k فقط
    idx = sample_from_probs(probs)
    return top[idx][0] # إرجاع الفهرس الأصلي

# 8. Top-p (Nucleus) Sampling
def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p: # التوقف عند تخطي العتبة p
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs] # إعادة المعايرة
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]

# 9. حيلة إعادة المعلمة
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1) # سحب عشوائي لا يعتمد على المعلمات
    return mu + sigma * epsilon # معادلة حتمية قابلة للاشتقاق

# 10. Gumbel-Softmax
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u)) # دالة Inverse CDF لتوزيع Gumbel

def gumbel_softmax(logits, temperature):
    # إضافة الضجيج العشوائي إلى اللوغاريتمات ثم تطبيق softmax مع الحرارة
    gumbels = [math.log(p) if p > 0 else -float('inf') for p in logits] 
    # في الكود الأصلي، يفترض تمرير الاحتمالات، فإذا كانت logits هي احتمالات نستخدمها.
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

## 7. ماذا يحدث داخلياً؟ (Execution Flow)
في LLMs:
1. يخرج النموذج قائمة من الأرقام (Logits) لكل كلمة ممكنة في القاموس.
2. نطبق `temperature_sample` لتقليل أو زيادة التباين.
3. لتجنب الكلمات غير المنطقية، نطبق `top_k` أو `top_p` لعزل أفضل الخيارات.
4. أخيرًا يتم تحويل الأرقام المفلترة إلى احتمالات (Softmax) ويتم سحب عينة عشوائية واحدة تصبح الكلمة التالية.

## 8. الأخطاء الشائعة
- **نسيان Numerical Stability في Softmax**: تطبيق `exp` مباشرة قد يسبب Overflow. يجب طرح أكبر قيمة أولاً.
- **في MCMC Burn-in**: أخذ العينات الأولى في MCMC يؤدي لنتائج سيئة لأن السلسلة لم تستقر بعد على التوزيع المستهدف.
- **Top-k مع قيم عالية جداً**: إذا وضعت $k=1000$، فقد تُدخل كلمات غير مناسبة إطلاقاً؛ Top-p أفضل في التكيف.

## 9. أين تُستخدم في AI الحقيقي؟
- **Reparameterization Trick**: تُستخدم في Variational Autoencoders (VAEs) لإنشاء صور ووجوه وهمية باستخدام التعلم العميق.
- **Top-p / Temperature**: موجودة كمعلمات أساسية (Parameters) في واجهات ChatGPT وموديلات OpenAI و LLaMA.
- **Diffusion Models (Midjourney / DALL-E)**: كل خطوة في عملية إزالة الضجيج (Denoising) تعتبر خطوة سحب عينة رياضية مرتبطة بـ Reparameterization.
- **Gumbel-Softmax**: البحث عن المعماريات العصبية (Neural Architecture Search).

## 10. خلاصة الدرس
طرق أخذ العينات تمكّن الذكاء الاصطناعي من الانتقال من العمليات الحتمية البسيطة إلى توليد بيانات معقدة إبداعياً. تعلّمنا كيف نروّض العشوائية لتعمل لصالحنا، وكيف نجعل الدوال التوليدية قابلة للاشتقاق لتتناسب مع خوارزميات الـ Backpropagation.

## 11. أهم المصطلحات

| English | العربية | المعنى |
|---------|---------|--------|
| Sampling | أخذ العينات | سحب قيم من توزيع احتمالي معين. |
| Logits | اللوجيتس | المخرجات الخام للشبكة العصبية قبل الـ Softmax. |
| Reparameterization | إعادة المعلمة | فصل العشوائية عن المتغيرات لتسهيل الاشتقاق. |
| Burn-in | فترة الإحماء | العينات الأولى في MCMC التي تُهمل لأنها غير دقيقة. |
| Deterministic | حتمي | غير عشوائي؛ يعطي نفس النتيجة دائمًا. |

## 12. اختبار الفهم (Quiz)
1. **لماذا لا يمكن عمل Backpropagate عبر عملية Sampling عادية؟**
   *الإجابة:* لأن عملية الـ Sampling تولد قفزات غير متصلة وقرارات متقطعة، والمشتقة بالنسبة للمعلمات (مثل $\mu$) تكون غير معرفة لأن كل استدعاء يُرجع قيمة مختلفة عشوائياً.
2. **ما هي المشكلة الأساسية مع Top-k مقارنة بـ Top-p؟**
   *الإجابة:* Top-k يستخدم عددًا ثابتًا. إذا كان النموذج واثقًا جدًا من كلمة واحدة، سيستمر في تضمين $k-1$ كلمة غير مناسبة. Top-p يتكيف ويأخذ الكلمات بناءً على الكتلة الاحتمالية (Confidence).
3. **لماذا خطأ Monte Carlo هو $O(1/\sqrt{N})$؟**
   *الإجابة:* بناءً على قانون الأعداد الكبيرة ونظرية النهاية المركزية، التباين ينخفض بقسمته على $N$، وبالتالي الانحراف المعياري (الذي يمثل الخطأ) يكون متناسباً مع مقلوب الجذر التربيعي لـ $N$.
4. **كيف تضمن Metropolis-Hastings التقارب للتوزيع المستهدف؟**
   *الإجابة:* من خلال تلبية شرط "Detailed Balance" (التوازن الدقيق)، مما يجعل التوزيع المستهدف هو التوزيع الثابت (Stationary Distribution) لسلسلة ماركوف.

## 13. تمارين عملية
1. **Beginner:** غيّر درجة الحرارة $T$ في دالة `temperature_sample` لترى كيف تتغير الاحتمالات عند $T=0.1$ وعند $T=5.0$.
2. **Intermediate:** اكتب دالة تدمج بين `Top-k` و `Top-p` في نفس الوقت.
3. **Advanced:** طبّق Gumbel-Softmax على شبكة عصبية بسيطة باستخدام PyTorch لاختيار مكون متقطع وتمرير المشتقات.
