# Small Language Models for Mental Health Analysis and On-Device Deployment

## Project Overview

Recent advances in AI have explored using language models to assist mental health care, for tasks like emotion classification and supportive journaling. However, deploying powerful Large Language Models (LLMs) in real-world mental health applications poses challenges in privacy, computation, and latency. Most state-of-the-art models (e.g. GPT-4, ChatGPT) run on cloud servers due to their size (billions of parameters), requiring users’ sensitive data to be sent off-device. This raises confidentiality concerns and can deter people from fully sharing their struggles. To address this, our project investigates Small Language Models (SLMs) – compact LLMs that can run directly on personal devices. SLMs promise to keep data localized (improving privacy) and run efficiently on mobile hardware. We aim to fine-tune such SLMs for mental health text analysis tasks (e.g. stress, depression, emotion detection) and evaluate their effectiveness. By doing so, we bridge a gap in current research, which has mostly focused on either very small specialist models or large cloud models, leaving the potential of on-device generative models underexplored. This project demonstrates how well fine-tuned SLMs can perform in understanding and generating mental health-related text, and how they can be deployed for on-device use without sacrificing too much performance.

## Models and Data

### Model Selection
We utilize three decoder-only language models as our base SLMs, chosen for their manageable size (∼1.5B to 3B parameters) and strong general pre-training. These are publicly available models pre-trained on large text corpora:
-   LLaMA-3.2B: ~3.2 billion parameter model by Meta AI (a downsized variant of LLaMA).
-   Qwen-2.5B: ~3.1 billion parameter model by Alibaba Cloud.
-   DeepSeek R1 (Qwen-1.5B Distilled): a 1.5B parameter distilled model from DeepSeek AI, based on Qwen.

These models are decoder (causal) LMs, as opposed to prior state-of-the-art MentalBERT or MentalRoBERTa which are smaller encoder-only models. We prefer decoder models to enable future chatbot-like interactions (generative responses) in a mental health context.

### Fine-Tuning Method
Instead of full model fine-tuning (which is memory-intensive for 3B parameters), we apply Quantized Low-Rank Adaptation (QLoRA) for efficient fine-tuning. QLoRA adds a few trainable low-rank adapter matrices to the model and uses 4-bit quantization to significantly reduce memory usage during training. This approach preserves model performance close to full fine-tuning while cutting GPU memory requirements by ~3×. In practice, QLoRA allowed us to fine-tune a ~3B model on a single 10GB GPU (RTX 3080) by using 4-bit precision for the model weights. During inference, the low-rank adapters are merged into the base model, incurring no extra latency. We fine-tuned each model in an instruction-following manner: the SLMs are trained as causal language models to generate appropriate responses given a prompted scenario, rather than training a separate classification head. This means our fine-tuning data is formatted as instruction → response pairs, so the models learn to produce helpful textual answers for mental health prompts, leveraging their generative nature.

**Practical Tips for LoRA/QLoRA Fine-Tuning:**
*   **Apply LoRA to All Layers:** For best performance, it's often beneficial to apply LoRA to all linear layers in the transformer architecture, including query, key, value, and output layers. This project follows this approach. If compute is a constraint, you might experiment with applying it to a subset, but comprehensive application is generally recommended.
*   **Rank (r) and Alpha (α):** The rank `r` determines the dimensionality of the trainable matrices. A common starting point for `r` is 8 or 16. `alpha` is a scaling factor, often set to twice the rank (e.g., if `r=8`, `alpha=16`). It's advisable to treat `alpha` as a hyperparameter you can tune. In QLoRA, `alpha` is an important hyperparameter to tune.
*   **Dataset Size and Epochs:** LoRA typically requires fewer epochs than full fine-tuning. For smaller datasets (like those often used in specialized domains), even 1-3 epochs can yield good results, as seen in our training settings (e.g., `num_epochs 2`). Overfitting can occur if trained for too long on smaller datasets.
*   **Learning Rate:** A learning rate common for AdamW optimizer with LoRA is in the range of 1e-4 to 5e-4. Our experiments use a learning rate of 1e-5, which can be effective especially with QLoRA and longer training or larger batch sizes. Experimentation is key.
*   **Evaluate Regularly:** Monitor performance on a validation set frequently to avoid overfitting and to find the optimal number of training steps.
*   **Consider Batch Size and Gradient Accumulation:** Adjust `batch_size` and `gradient_accumulation_steps` based on your GPU memory. QLoRA significantly reduces memory, but these still play a role. Our setup uses a `batch_size` of 2 and `gradient_accumulation_steps` of 16, effectively creating a larger logical batch size.

### Training Data
The project leverages a combination of multiple mental health datasets, aggregated to provide diverse training examples. We unified five datasets commonly used in prior research (covering stress, depression, suicide risk, and emotions) into one training set. This multi-task dataset approach follows findings by Xu et al. that diversity in fine-tuning data can improve general performance across platforms. The included datasets are:
-   ISEAR (Emotion) – A public emotion classification dataset with short texts labeled as Joy, Fear, Anger, Sadness, Disgust, Guilt, or Shame. (7-class single-sentence emotion recognition)
-   Dreaddit (Stress) – Reddit-based stress detection dataset (posts labeled stressed vs not stressed).
-   Depression (Reddit) – A collection of Reddit posts labeled for signs of depression (binary classification).
-   DepSeverity (Depression Severity) – A dataset with posts categorized by depression severity level (e.g. mild, moderate, severe; multi-class).
-   SDCNL (Suicidal Ideation) – A dataset for suicidal ideation detection (posts labeled as containing suicidal intent vs. depression discussions).

All datasets are openly available (via GitHub or HuggingFace) to ensure easy reproducibility. We used an approximate 80/20 train-test split as provided or recommended in these sources (for example, ISEAR provides 6896 training and 770 test instances).

To enhance the model’s generative training, we augmented the data with additional context using a GPT-4 based assistant. For each training sample, we constructed an instruction prompt mimicking a conversation with a psychiatrist: a system message establishes the role (“experienced psychiatrist”) and a user message describes the person’s writing (symptoms or feelings) along with a request to analyze or provide a diagnosis. The intended label or output (e.g. the emotion category or risk level) was then integrated into a reference response. These enriched instruction-response pairs (generated using a GPT-4 variant dubbed GPT-4o-mini) were saved as JSONL files for training. This process provided the SLMs with more context-rich examples, effectively teaching them how to respond in a helpful, empathetic manner given a scenario, rather than just predicting a label.

## Installation and Environment Setup

To run this project, you should prepare a Python environment with the required libraries and frameworks:

1.  **Clone the Repository**: Download or clone the repository code to your local machine:
    ```bash
    https://github.com/junnystateofmind/slm_iclab.git
    cd slm_iclab
    ```

2.  **Python and Virtual Environment**: Ensure you have Python 3.9+ installed. It’s recommended to create a virtual environment (using `venv` or `Conda`) to manage project dependencies:
    ```bash
    # Using venv
    python3 -m venv venv
    source venv/bin/activate
    ```
    (Alternatively, use `conda create -n slm-env python=3.10` and `conda activate slm-env`.)

3.  **Install Dependencies**: Install the required Python libraries. The repository provides a `requirements.txt` listing all dependencies. Use `pip` to install them:
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries include:
    -   PyTorch (with CUDA support for GPU acceleration)
    -   Hugging Face Transformers and Datasets (for model and data loading)
    -   BitsAndBytes (for 4-bit quantization support used in QLoRA)
    -   PEFT (Parameter-Efficient Fine-Tuning library for LoRA/QLoRA adapters)
    -   scikit-learn / scipy (for evaluation metrics like cosine similarity, etc.)
    -   Evaluate or BERTScore package (for computing BERTScore)

4.  **Hardware Requirements**:
    -   For training, a GPU is highly recommended. Fine-tuning a 3B model with QLoRA was feasible on a single NVIDIA RTX 3080 (10GB VRAM) in our experiments. Ensure your GPU drivers and CUDA are properly configured.
    -   If you plan to deploy on an edge device, the fine-tuned model can be quantized further (e.g. 8-bit or 4-bit) for CPU inference. In our on-device tests, we used Galaxy S25+ with 12GB RAM for running inference, which necessitated using an efficient quantized model format.

5.  **Optional - Data Setup**:
    By default, the training scripts will download or expect the datasets (ISEAR, Dreaddit, etc.) from their sources. If needed, update the paths or download instructions in the configuration to point to your local copies. Make sure the data is in the expected format (our fine-tuning script expects a unified JSONL or CSV with an “instruction” and “response” field for each entry, after the augmentation step).

## Usage

### 1. Fine-Tuning

All training scripts live in **`train_and_evaluate/`**.
Pass your project root with `--root_path`; every other flag has a default.
Refer to the "Practical Tips for LoRA/QLoRA Fine-Tuning" in the "Fine-Tuning Method" section for guidance on hyperparameters.

#### 1-a. Qwen-2.5-3B
```bash
cd Tiny-Mental-LLM
python train_and_evaluate/QLoRA_Qwen_2.5_3B_train.py \
  --root_path $(pwd) \
  --num_epochs 2 \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5
```

#### 1-b. LLaMA-3.2-3B
```bash
cd Tiny-Mental-LLM
python train_and_evaluate/QLora_Llama_3.2_3B_train.py \
  --root_path $(pwd) \
  --num_epochs 2 \
  --batch_size 2
```

Fine-tuned weights (merged or adapter-only) are written to `pretrained_models/<model>/fine_tuned/`.

---

### 2. Quick Validation

Open `Evaluate.ipynb` (or `test_finetuning.ipynb`) to run BERTScore and the LLM-based multi-criteria evaluation:

```bash
jupyter notebook train_and_evaluate/Evaluate.ipynb
```

Inside the notebook set:
```python
MODEL_PATH = "<root>/pretrained_models/Qwen2.5-3B-Instruct/fine_tuned"
PROMPT = "USER: I feel anxious all the time.\nASSISTANT:"
```

Run all cells to generate and score responses.

---

### 3. On-Device Deployment (Optional)

#### 1. ONNX export
```bash
cd sLM_SNPE
python huggingface_onnx_qwen.py \
  --hf_model_path <MODEL_PATH> \
  --onnx_out qwen3b_fp16.onnx
```

#### 2. Quantize & optimize for SNPE
Use Qualcomm’s `snpe-dlc-quantize` and `snpe-dlc-graph-prepare` to produce an INT8/INT4 DLC for Hexagon HTP.

---

### 4. Command Cheat-Sheet

| Stage           | Script / Notebook             | Minimum flag(s)                |
|-----------------|-------------------------------|--------------------------------|
| Train – Qwen    | `QLoRA_Qwen_2.5_3B_train.py`  | `--root_path`                  |
| Train – LLaMA   | `QLora_Llama_3.2_3B_train.py` | `--root_path`                  |
| Evaluate / demo | `Evaluate.ipynb`              | `MODEL_PATH` (in notebook)     |
| ONNX export     | `huggingface_onnx_qwen.py`    | `--hf_model_path`, `--onnx_out`|

TIP: Reduce checkpoint size by increasing `--save_steps` and `--logging_steps`.

## Results and Performance Overview

After fine-tuning, the small models demonstrated notable improvements in their ability to analyze and generate mental health-related text. We summarize key outcomes below:

-   **Semantic Similarity (BERTScore)**: The fine-tuned SLMs achieved high semantic similarity to reference answers (which were generated by a GPT-4-quality model). For instance, LLaMA-3.2B and Qwen-2.5B fine-tuned models reached mean BERTScore around 0.876–0.879 on the test prompts, indicating their responses closely match the ground-truth content. This is a substantial gain over a smaller distilled model (DeepSeek R1, 1.5B) without fine-tuning, which scored ~0.805. These scores suggest the fine-tuned 3B models can produce answers with almost the same meaning as a strong reference answer ~88% of the time. Fine-tuning clearly helped: the distilled model (which was not further tuned on our data) lagged behind.
-   **LLM-Aided Quality Evaluation**: We evaluated each model’s responses on critical qualities using an LLM-assisted scoring system. A GPT-4-level evaluator scored answers on a 0–5 scale across six criteria (Accuracy, Empathy, Clarity, Guidance, Safety, Helpfulness). The fine-tuned models performed impressively given their size. For example, the LLaMA-3.2B SLM scored 3.7–3.9 on Accuracy, Empathy, and Clarity, compared to the cloud GPT-4 model’s scores around 4.5 on those criteria. The Qwen-2.5B model had similar scores (mid-3s) on these metrics. In contrast, the smaller DeepSeek 1.5B model (without fine-tuning) only scored around 2.4–2.6, indicating much weaker performance. Notably, the fine-tuned SLMs even outperformed GPT-4 on the Actionable Guidance dimension, scoring ~2.7–2.8 vs GPT-4’s 2.15. This suggests our models sometimes give more concrete advice (though GPT-4 was more consistent, with lower variance). In terms of Safety, the Qwen-2.5B model achieved a high score (~4.1) comparable to GPT-4’s 4.36, an encouraging sign for deploying it in sensitive applications. Overall, these qualitative evaluations highlight that fine-tuning significantly boosts SLMs’ capability to produce relevant, empathetic, and safe responses. While a gap remains between the 3B SLMs and a large model like GPT-4 (e.g. GPT-4 averaged around 4.8 in Overall Helpfulness vs. ~3.6 for the SLMs), the improvement over the base small model is dramatic across all metrics. This demonstrates the viability of SLMs for on-device mental health support, especially when fine-tuned to the domain.
-   **Emotion Classification Performance**: We also tested the fine-tuned models on a nuanced emotion profiling task using the StudEmo benchmark (which provides human ratings across 10 emotion dimensions). Instead of a classification accuracy, we measured how well the model’s generated emotion scores align with human annotations. The fine-tuned Qwen-2.5B model achieved a cosine similarity ~0.42 to the human-rated emotion vectors, substantially higher alignment than the smaller DeepSeek model (~0.21). However, both models struggled to output the complete structured schema of 10 emotions reliably – Qwen only included the full set in ~18.6% of its responses, whereas DeepSeek did so 51.6% of the time (though often with default values). This indicates that while the fine-tuned larger SLM captured the overall emotional tone better, it had difficulty following the exact output format. The results highlight a limitation in using SLMs for structured classification-type output: models that excel in free-form text generation can find it challenging to precisely mimic a fixed scoring template. Future fine-tuning or prompt-engineering may improve schema adherence. Despite these challenges, the model’s ability to partially mirror human emotion judgments is promising for applications like assessing emotional content in user text. It also underscores the need for improved strategies (or perhaps fine-tuning with explicit schema supervision) if such granular tasks are required.

In summary, our experiments show that fine-tuned small LMs can achieve strong performance in mental health text analysis. They approximate the output quality of much larger models in many aspects (semantic relevance, empathy, clarity), all while being deployable on-device. There remains a performance gap to the very best large models, especially in complex reasoning and strict output formats, but the trade-off in privacy and computational cost is favorable. These findings support the use of SLMs as a foundation for on-device mental health support tools, with further research warranted to enhance their accuracy and reliability.

## References

-   Kim et al. – MindfulDiary: An application of LLMs to help users with structured self-reflection in a diary format.
-   Nepal et al. – MindScape: A system leveraging language models to provide therapeutic introspection and support.
-   Song et al. – Highlighted privacy concerns with cloud-based AI in mental health, noting users may hold back personal details if data is sent to third-party servers.
-   Ji et al. – Introduced MentalBERT, an encoder-only model (~110M params) fine-tuned for mental health text classification, which was a previous state-of-the-art in the domain.
-   Xu et al. – Experimented with fine-tuning larger models (Flan-T5, Alpaca) on mental health tasks, showing they can rival smaller specialized models but with much greater computational cost.
-   Yang et al. – Conducted studies on LLMs for mental health: (a) fine-tuning a LLaMA model for well-being analysis, and (b) evaluating ChatGPT-like models on mental health text reasoning.
-   Dettmers et al. – Proposed QLoRA (Quantized LoRA), a technique combining low-rank adaptation with 4-bit quantization for efficient fine-tuning of large models.
-   Ngo et al. (2022) – Created StudEmo, a dataset with multi-dimensional emotion annotations by human raters, used to evaluate how well models capture nuanced emotions.
-   Raschka (2023) – “Fine-Tuning LLMs with LoRA” (Article). Provided practical insights and best practices for LoRA-based fine-tuning of language models, such as tips on scaling factors and layer application.
-   Practical Tips for Fine-Tuning LLMs using LoRA (Low-Rank Adaptation) - Continuum Labs AI Training. (Accessed 2025).

```