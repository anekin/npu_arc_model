# Wave 2: normalization

The lead was precision and sparsity ambiguity. Bare TOPS cannot be ranked across this set.
The normalized schema is:

`device, scope, numeric peak, operation type, precision, sparsity assumption, included
engines, CPU partition, real-time partition, safety island, memory capacity, memory
interface, usable/peak bandwidth, power definition, thermals, concurrency, target workload,
software, lifecycle, evidence quality`.

Hard rule: store missing values as unknown. Do not infer SoC power by dividing TOPS by a
published accelerator TOPS/W value, and do not convert eTOPS to conventional TOPS.
