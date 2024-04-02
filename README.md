# Learning NLP and Large Language Model (LLM)
- Personal learning path for NLP and Large Language Model (LLM)
- Paper recommendations

Chinese version: [README_CN.md](README_CN.md) (in progress)

# NLP Learning Materials

**Notes: **

1. **I don't recommend older materials (even if they are "classic") because everything changes fast in NLP**
2. **You need to know basic knowledge of machine learning and Python**
   1. **Machine learning courses:** 
      1. **[Home | CS 189/289A (eecs189.org)](https://eecs189.org/)**
      2. **[CS229: Machine Learning (stanford.edu)](https://cs229.stanford.edu/) (theoretic perspective)**
   2. **Python: too many courses... (ex: [CS50's Introduction to Programming with Python (harvard.edu)](https://cs50.harvard.edu/python/2022/) )**

Courses / Tutorials: 

- [Stanford CS 224N | Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/index.html)
- [CS224U: Natural Language Understanding - Spring 2023 (stanford.edu)](https://web.stanford.edu/class/cs224u/)
- [Lena Voita (lena-voita.github.io)](https://lena-voita.github.io/)
- [Introduction - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) (good intro about huggingface libraries)

Books:

- [Speech and Language Processing (stanford.edu)](https://web.stanford.edu/~jurafsky/slp3/)
- [Natural Language Processing with Transformers Book](https://transformersbook.com/)

Blogs:

- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Understanding Large Language Models -- A Transformative Reading  List](https://sebastianraschka.com/blog/2023/llm-reading-list.html)

# Important Papers (In Progress)

Surveys:

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169) 
- [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)
- [Understanding Large Language Models -- A Transformative Reading  List](https://sebastianraschka.com/blog/2023/llm-reading-list.html)
- [Foundational must read GPT/LLM papers](https://community.openai.com/t/foundational-must-read-gpt-llm-papers/197003/10)

## LLM Popular Models

- [BERT](https://aclanthology.org/N19-1423.pdf)

- [GPT 1.0](https://openai.com/research/language-unsupervised)

- [GPT 2.0](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

- [GPT 3.0](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) 

- [T5](https://jmlr.org/papers/v21/20-074.html)

- [FLAN](https://openreview.net/forum?id=gEZrGCozdqR)

- [InstructGPT](https://arxiv.org/abs/2203.02155)

- [Flan-T5/PaLM](https://arxiv.org/abs/2210.11416)

- [GLM-130B](https://arxiv.org/abs/2210.02414)

- [LLaMA](https://arxiv.org/abs/2302.13971)

- [LLaMA 2](https://arxiv.org/abs/2307.09288)

- [GPT-4](https://openai.com/research/gpt-4)

- [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)

- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)

- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)

- [WizardLM](https://github.com/nlpxucan/WizardLM)
  - [WizardCoder](https://arxiv.org/abs/2306.08568)

  - [WizardMath](https://arxiv.org/abs/2308.09583)

- [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)

- [Baichuan2-7B](https://github.com/baichuan-inc/Baichuan2)

## LLM Learning and Tuning Techniques

### Prompt Tuning

[Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)

### In-Context Learning

- [A Survey for In-context Learning](https://arxiv.org/abs/2301.00234)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
- [In-Context Learning Paper List](https://github.com/dqxiu/ICL_PaperList)

### Instruction Tuning

- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)

### Chain of Thoughts

[Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### Parameter-Efficient Fine-Tuning (PEFT)

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across  Scales and Tasks](https://arxiv.org/abs/2110.07602)
- [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)
- [Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for  Pre-trained Language Models](https://arxiv.org/abs/2203.06904)

## LLM Inference, Benchmark and Development Tools

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [MLC-LLM](https://mlc.ai/mlc-llm/docs/)
- [FastChat](https://github.com/lm-sys/FastChat)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Huggingface](https://huggingface.co/)
- [Gradio](https://github.com/gradio-app/gradio)
- [vllm](https://github.com/vllm-project/vllm)
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)
