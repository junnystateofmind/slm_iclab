
# Small LMs, Chain‑of‑Thought Reasoning, and Tool Use Performance

## 1. Why Chain‑of‑Thought (CoT) Matters  
CoT prompting asks the model to **think step‑by‑step** before giving the final answer.  
Even sub‑3 B models show higher accuracy **and** fewer JSON / API‑schema errors when we:

* request an explicit `reasoning` (or `thought`) field first, then  
* let the model fill the remaining structured output.

> *Example:* adding a `reasoning` field raised a 3 B model’s GSM8K math accuracy by **≈ 60 %** [[Wei et al. 2022]](#ref3).

---

## 2. ReAct = Reason + Act  

The **ReAct** loop interleaves  

```
Thought → Action(tool) → Observation → … → Final
```

Yao *et al.* showed this pattern cuts hallucination and beats RL policies on WebShop / ALFWorld tasks [[Yao et al. 2023]](#ref1).

*Mini‑example*

```text
Q: Population of the largest French city squared?
Thought: I should look up the largest French city.  
Action: wiki_search("largest city in France")  
Observation: …Paris, pop ≈ 2.16 M…  
Thought: Now square that number with a calculator.  
Action: calculator("2160000 ** 2")  
Observation: 4.67e12  
Final: 4.67 × 10^12
```

---

## 3. Tool Use with Very Small Models  

| Approach | Key idea | Take‑away for ≤ 3 B |
|----------|----------|---------------------|
| **Toolformer** | Self‑labels a GPT‑J‑6 B with a handful of API demos | Mid‑size models learn *when* to call tools → arithmetic & QA jump to GPT‑3‑175 B level [[Schick et al. 2023]](#ref2) |
| **Gorilla** | Fine‑tuned on 1 600 API docs | LLaMA‑7 B > GPT‑4 on correct API calls; method scales down to 3 B with domain‑specific fine‑tune [[Patel et al. 2023]](#ref4) |
| **Qwen‑Math‑3 B** | Adds code‑execution traces during SFT | 3 B model writes & runs Python to solve math [[Qwen Team 2024]](#ref5) |
| **DeepSeek‑V2‑Lite** | 2.4 B active params (MoE) + 32 K context | MoE efficiency ≈ dense 7 B reasoning quality [[DeepSeek AI 2024]](#ref6) |

---

## 4. Prompting & Fine‑Tuning Cheatsheet  

| Phase | What to try |
|-------|-------------|
| Prototype | *Zero‑shot* CoT trigger: **“Let’s think step by step.”** |
| Stabilise | Add **2‑3 few‑shot** CoT examples (Q → reason → API call → answer). |
| Agentify | Use **ReAct scratchpad** format; feed back each Observation. |
| Train | 1 k API‑call pairs → **QLoRA** fine‑tune (INT8) on Qwen‑3 B. |
| Measure | Toolformer tasks, APIBench subset, GSM8K. |

---

## 5. Deployment on ≤ 3.5 GB  

* **INT8 weights** ≈ 3 GB for 3 B params → fits Snapdragon NPU (SNPE) or Apple ANE.  
* **4‑bit** can cut mem × 2 but may drop multi‑step accuracy by **30 %+** [[Dettmers et al. 2022]](#ref7).  
* Keep scratchpad short; summarise retrieved docs to limit KV cache.  
* Real demo: Qwen‑3 B agent controlling a PC in real‑time [[Qwen Team Video]](#ref8).

---

## 6. Suggested Benchmarks  

| Category | Dataset / Task | Metric |
|----------|----------------|--------|
| CoT math | **GSM8K** | accuracy |
| API call | **Toolformer suite** | API‑syntax F1 |
| Prog API | **APIBench** | exact API match |
| ReAct QA | **HotpotQA** + wiki search | answer + evidence |
| Interactive | **WebShop mini** | task success |
| Mobile | Snapdragon dev board | tokens · s⁻¹, RAM, battery |

---

## References  

| id | Citation |
|----|----------|
| <a id="ref1"></a>[Yao et al. 2023] | Yao S., Zhao Y., et al. **“ReAct: Synergizing Reasoning and Acting in Language Models.”** arXiv:2210.03629 |
| <a id="ref2"></a>[Schick et al. 2023] | Schick T., Dwivedi Y., et al. **“Toolformer: Language Models Can Teach Themselves to Use Tools.”** arXiv:2302.04761 |
| <a id="ref3"></a>[Wei et al. 2022] | Wei J., Wang X., et al. **“Chain‑of‑Thought Prompting Elicits Reasoning in Large Language Models.”** arXiv:2205.11916 |
| <a id="ref4"></a>[Patel et al. 2023] | Patel S., Shum K., et al. **“Gorilla: Large Language Model Connected with Massive APIs.”** arXiv:2305.15334 |
| <a id="ref5"></a>[Qwen Team 2024] | Qwen Team. **“Qwen‑2.5 Release.”** GitHub: <https://github.com/QwenLM/Qwen2> |
| <a id="ref6"></a>[DeepSeek AI 2024] | DeepSeek AI. **“DeepSeek‑V2: Towards Deeper Language Understanding.”** arXiv:2403.00775 |
| <a id="ref7"></a>[Dettmers et al. 2022] | Dettmers T., et al. **“8‑bit Matrix Multiplication for Transformers at Scale.”** arXiv:2208.07339 |
| <a id="ref8"></a>[Qwen Team Video] | Demo video: “Qwen‑3B Multimodal Agent Controlling a PC.” YouTube (2024). |

