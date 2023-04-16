# uk4b: Metadata Pretraining Towards Instruction Finetuning

We pretrain unidirectional language models on 4B tokens from [UberText 2.0](https://lang.org.ua/en/ubertext/). We enrich document text with weakly structured metadata, such as title, tags, and publication year, enabling metadata-conditioned text generation and text-conditioned metadata prediction at the same time. We pretrain GPT-2 Small, Medium, and Large models on a single GPU, reporting training times, BPC on BrUK, BERTScore, and BLEURT on titles for 1000 News from the Future.

[![See metadata pretraining video (2m33s)](https://img.youtube.com/vi/FYJZBXfLaDA/default.jpg)](https://youtu.be/FYJZBXfLaDA)


Model checkpoints are available at https://a.wilab.org.ua/gpt/. BLEURT/BERTscore evaluation on News from the Future is available on [lang-uk/bleurt_eval](https://github.com/lang-uk/bleurt_eval)

Next, we venture to formatting POS and NER datasets as instructions, and train low-rank attention adapters, performing these tasks as constrained text generation. See video (2m50s): https://www.youtube.com/watch?v=NDXJ9hXtf-o

[![See instruction finetuning video (2m50s)](https://img.youtube.com/vi/NDXJ9hXtf-o/default.jpg)](https://youtu.be/NDXJ9hXtf-o)

See POS and NER adapters can be trained using [examples/Makefile](examples/Makefile).

This repository fuses [karpathy/NanoGPT](https://github.com/karpathy/nanoGPT) and [asivokon/unlp-2023-shared-task](https://github.com/asivokon/unlp-2023-shared-task)


Authors:

- Volodymyr Kyrylov @proger
- Dmytro Chaplynskyi @dchaplinsky
