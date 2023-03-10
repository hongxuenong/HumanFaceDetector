Rec mobile V3, loop through 1000 images:

 | model (GPU V100) | speed (ms) | STD between torch |
 | ---------------- | ---------- | ----------------- |
 | paddle           | 22.51      | -                 |
 | paddle infer     | 7.26       | -                 |
 | torch            | 7.90       | -                 |
 | onnx             | 4.69       | 9.5300955e-08     |
 | tensorRT         | 2.69       | 1.02662455e-07    |
 | tensorRT fp16    | 2.05       | 1.7818458e-05     |


 | model (CPU) | speed (ms) |
 | ----------- | ---------- |
 | torch       | 191.80     |
 | onnx        | 49.81      |