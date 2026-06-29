Title
The application is Use Case Driven
Status
Accepted
Context

The system integrates multiple AI providers.

Providers may change over time.

Business workflows should remain stable.
Decision

Business workflows are represented as Use Cases.

Use Cases coordinate all services.

Providers must never coordinate themselves.
Consequences

✔ Easy provider replacement

✔ Easier testing

✔ Clear business workflows

✔ Better scalability


ADR-001 — Learning Resources

Decision:

Learning Session يمكن أن تحتوى على أكثر من Learning Material.

Reason:

لأن المتعلم قد يعتمد على كتاب، سلايدات، وملخص لنفس الهدف التعليمى.

Constraint:

فى الإصدار الأول، النظام لن يتحقق تلقائيًا من ترابط المصادر، بل سيفترض أن المتعلم هو من يختار المصادر الصحيحة.

ADR-002 — Feedback Strategy

Decision:

سيكون لدينا مستويان من التغذية الراجعة:

Interaction Feedback (بعد كل شرح)

"وضح أكثر"
"هات مثال"
"اختصر"

ويستخدم لتحسين الجلسة الحالية.

Session Review (بعد نهاية الجلسة)

تقييم عام
هل تحقق هدف التعلم؟
تعليق اختيارى

ويستخدم لتحسين المنتج مستقبلًا وبناء Dataset للتقييم.