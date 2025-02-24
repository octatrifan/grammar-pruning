# Grammar Pruning for Low-Latency Task-Oriented Parsing

This repository contains the implementation for the paper:

**"Grammar Pruning: Enabling Low-Latency Task-Oriented Language Models for Edge AI"**

## 📜 Overview
Grammar pruning is a novel approach for **real-time, task-oriented semantic parsing** in resource-constrained environments. Unlike traditional constrained decoding, grammar pruning dynamically restricts the model’s output space by leveraging a **rule-based Named-Entity Recognition (NER) module** to extract relevant entities from user input. This technique ensures **structured, accurate, and low-latency** natural language understanding, making it ideal for edge AI applications.

![Untitled (3)](https://github.com/user-attachments/assets/b679f8de-d1af-4d13-8ea5-2a77e0fefb1a)


## 🏗 Methodology
1. **Named-Entity Recognition (NER)**: Extracts menu items from user input and maps them to predefined categories.
2. **Grammar Pruning**: Dynamically constrains model output based on detected entities, preventing hallucinations and enforcing valid schema.
3. **Efficient Model Execution**: Small Language Models (SLMs) optimized for edge devices with quantization and low-latency techniques.
4. **Constrained Decoding**: Enforces structured outputs using input-dependent grammars.

## 🗃 Dataset
Our work is using the [FoodOrderingDataset](https://github.com/amazon-science/food-ordering-semantic-parsing-dataset), which consists of task-oriented parsing requests in the food-ordering domain, derived from five categories:

### Dataset Files:
- Processed JSON files for each category (e.g., `burger_dataset.json`, `coffee_dataset.json`)
- Training files (`train_pizza_dataset.json`, etc.)
- Named-Entity Recognition (NER) processed files (`pizza_dataset_NER.json`, etc.)

## 📊 Experiments & Results
Our method was tested in a **zero-shot** setting on the **FoodOrderingDataset**, showing **significant improvements** over prior methods like [Cross-TOP](https://arxiv.org/abs/2206.05352):
| Approach | Burger Accuracy | Coffee Accuracy |
|----------|----------------|----------------|
| Cross-TOP | 73.3% | 54.8% |
| Grammar Pruning (Ours) | **96.2%** | **91.1%** |

### 🏎 Latency Performance (Raspberry Pi 5 vs. NVIDIA 4090 GPU)
| Device | Model | Time-to-First-Token (TTFT) | Total Latency |
|--------|--------|-----------------|--------------|
| Raspberry Pi | Qwen2.5 0.5B (Q4) | 2.10s | 3.79s |
| Raspberry Pi | Qwen2.5 1.5B (Q4) | 4.77s | 7.29s |
| NVIDIA 4090 GPU | Qwen2.5 1.5B (F16) | 0.27s | 0.77s |

## 🔧 Configuration
- **Models:** Supports Qwen2.5 0.5B/1.5B with quantization (`Q4` for edge devices, `F16` for full precision)
- **Grammar Pruning:** Uses [Guidance](https://github.com/guidance-ai/guidance) for constrained decoding
- **Edge Deployment:** Optimized for **low-power** devices like Raspberry Pi

## 📜 License
This project is licensed under the **MIT License**.


## 💬 Contact
For questions or collaborations, please open an **issue** or **pull request**. 🚀

