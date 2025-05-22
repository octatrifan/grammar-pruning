# 🚀 Grammar Pruning: Enabling Low-Latency Zero-Shot Task-Oriented Semantic Parsing for Edge AI

This repository hosts the official implementation and datasets for our paper, **"Grammar Pruning: Enabling Low-Latency Zero-Shot Task-Oriented Language Models for Edge AI"**. Our innovative approach enables precise, real-time semantic parsing directly on resource-constrained edge devices. 📱⚙️

## Abstract
Edge deployment of task-oriented semantic parsers demands high accuracy under tight latency and memory budgets. We present Grammar Pruning, a lightweight zero-shot framework that begins with a user-defined schema of API calls and couples a rule-based entity extractor with an iterative grammar-constrained decoder: extracted items dynamically prune the context-free grammar, limiting generation to only those intents, slots, and values that remain plausible at each step. This aggressive search-space reduction both eliminates hallucinations and slashes decoding time. On the adapted FoodOrdering, APIMixSNIPS, and APIMixATIS benchmarks, Grammar Pruning with small language models achieves an average execution accuracy of over 90\%—rivaling State-of-the-Art, cloud-based solutions—while sustaining at least 2x lower end-to-end latency than existing methods. By requiring nothing beyond the domain’s full API schema values yet delivering precise, real-time natural-language understanding, Grammar Pruning positions itself as a practical building block for future edge-AI applications that cannot rely on large models or cloud offloading.

![Grammar Pruning Illustration](https://github.com/user-attachments/assets/bbc8c86f-20c9-45d1-8bed-30a5cae6150e)


*Figure 1: Grammar Pruning dynamically prunes generation grammars, ensuring fast, accurate, and hallucination-free semantic parsing.*

## 📌 Key Features

* ✅ **Zero-Shot Adaptability**: No fine-tuning needed when deploying to new domains.
* 🧹 **Dynamic Grammar Reduction**: Limits responses strictly to valid schema elements, eliminating inaccuracies.
* 🛠️ **Edge-Optimized**: Designed for minimal computational resources, suitable for small language models.
* ⚡ **High Accuracy, Low Latency**: Consistently achieves over 90% accuracy, delivering results twice as fast as comparable methods.

## 📊 Experimental Highlights

| Dataset                  | Model Size | Accuracy (%) | Latency (s) |
| ------------------------ | ---------- | ------------ | ----------- |
| 🍔 FoodOrdering (Burger) | 1.5B       | 96.2%        | <1s         |
| ☕ FoodOrdering (Coffee)  | 1.5B       | 91.1%        | <1s         |
| 🎤 APIMixSNIPS           | 4B         | 96.1%        | <1s         |
| ✈️ APIMixATIS            | 4B         | 92.2%        | <1s         |

## 📚 Datasets

We provide meticulously adapted versions of three prominent semantic parsing benchmarks:

* 🍕 [FoodOrderingDataset](https://github.com/amazon-science/food-ordering-semantic-parsing-dataset)
* 🎤 **API**[MixSNIPS](https://github.com/VinAIResearch/MISCA/tree/main/data/mixsnips)
* ✈️ **API**[MixATIS](https://github.com/VinAIResearch/MISCA/tree/main/data/mixsnips)

Datasets are formatted as Python API calls for seamless parsing and easy validation. Schemas for each datasets are included.

## 🗂️ Repository Structure

```
├── APIMixATIS
│   ├── atis_data.json
│   ├── atis_data_augmented.json
│   └── atis_data_schema.json
├── APIMixSNIPS
│   ├── snips_data.json
│   ├── snips_data_augmented.json
│   └── snips_data_schema.json
├── FoodOrderingDataset
│   ├── data
│   └── scripts
├── Implementation
│   ├── APIMixATIS & APIMixSNIPS
│   │   ├── DSCP
│   │   ├── GPT
│   │   ├── GrammarPruning
│   │   ├── ThinkingMode
│   │   └── utils
│   └── FoodOrdering
│       └── Fine-Tuning and Accuracy Notebooks
├── LICENSE
└── README.md
```



### ▶️ Running Experiments

Refer to implementation notebooks and scripts provided in respective dataset directories for quick and intuitive setup.

## 📖 Citation

If this work helps your research, please consider citing our paper:

```bibtex
@article{grammarpruning2025,
  title={Grammar Pruning: Enabling Low-Latency Zero-Shot Task-Oriented Language Models for Edge AI},
  author={Anonymous},
  journal={Anonymous},
  year={2025}
}
```

## 📜 License

Distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.
