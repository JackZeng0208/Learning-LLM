# 语言大模型（LLM）阅读清单

## LLM基础知识

综述：

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169) 
- [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)
- [Understanding Large Language Models -- A Transformative Reading  List](https://sebastianraschka.com/blog/2023/llm-reading-list.html)
- [Foundational must read GPT/LLM papers](https://community.openai.com/t/foundational-must-read-gpt-llm-papers/197003/10)
- https://zhuanlan.zhihu.com/p/597586623
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

博客：

- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## LLM著名模型

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

## LLM学习和微调技术

### 提示学习（Prompt Tuning）：

- [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)
- [什么是 prompt learning？简单直观理解 prompt learning](https://blog.csdn.net/qq_35357274/article/details/122288060)

### 上下文学习（In-context Learning）：

- [A Survey for In-context Learning](https://arxiv.org/abs/2301.00234)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
- [In-Context Learning Paper List](https://github.com/dqxiu/ICL_PaperList)

### 指令微调（Instruction Tuning）：

- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)

### 思维链（Chain of Thought，CoT）：

- [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### PEFT：

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across  Scales and Tasks](https://arxiv.org/abs/2110.07602)
- [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)
- [Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for  Pre-trained Language Models](https://arxiv.org/abs/2203.06904)

## LLM推理，评测和开发工具

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [MLC-LLM](https://mlc.ai/mlc-llm/docs/)
- [FastChat](https://github.com/lm-sys/FastChat)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Huggingface](https://huggingface.co/)
- [Gradio](https://github.com/gradio-app/gradio)
- [vllm](https://github.com/vllm-project/vllm)
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)