import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention
import onnx
from onnx import external_data_helper, helper, checker

##############################################
# 1. Monkey patch (tril/triu, Flash Attention)
##############################################
def patched_tril(x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    n, m = x.shape[-2], x.shape[-1]
    i = torch.arange(n, device=x.device).unsqueeze(1)
    j = torch.arange(m, device=x.device).unsqueeze(0)
    mask = (i - j >= -diagonal).to(x.dtype)
    return x * mask

def patched_triu(x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    n, m = x.shape[-2], x.shape[-1]
    i = torch.arange(n, device=x.device).unsqueeze(1)
    j = torch.arange(m, device=x.device).unsqueeze(0)
    mask = (j - i >= -diagonal).to(x.dtype)
    return x * mask

torch.tril = patched_tril
torch.triu = patched_triu

def patched_llama_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    return LlamaAttention.original_forward(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )

def monkey_patch_llama_attention():
    if not hasattr(LlamaAttention, "original_forward"):
        LlamaAttention.original_forward = LlamaAttention.forward
    LlamaAttention.forward = patched_llama_attention_forward

##############################################
# 2. Qwen 래퍼
##############################################
class QwenOnnxWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=AutoConfig.from_pretrained(model_name),
            low_cpu_mem_usage=True
        )
        self.base_model.eval()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        return outputs.logits

##############################################
# 3. 메인: '한 방에' 최종 onnx 파일 만들기
##############################################
def main():
    monkey_patch_llama_attention()

    # 폴더/파일명
    export_dir = "qwen_onnx"
    os.makedirs(export_dir, exist_ok=True)
    onnx_temp = os.path.join(export_dir, "qwen_no_flash_raw.onnx")  # 임시 파일
    onnx_final_dir = "qwen_onnx_merged"
    os.makedirs(onnx_final_dir, exist_ok=True)
    onnx_final = os.path.join(onnx_final_dir, "qwen_no_flash_merged.onnx")

    # 모델 준비
    model_name = "junnystateofmind/Qwen2.5-3B_merged_rank256_alpha512"
    wrapper = QwenOnnxWrapper(model_name)

    # 더미 입력
    B, S = 1, 128
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)
    attention_mask = torch.ones((B, S), dtype=torch.long)

    # -----------------------
    # (A) 1차 Export (외부 데이터 형식)
    # -----------------------
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        onnx_temp,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        use_external_data_format=True
    )
    print(f"[Export] ONNX 임시 파일 생성: {onnx_temp}")

    # 크기 확인
    size_bytes = os.path.getsize(onnx_temp)
    print(f"  {onnx_temp} 크기: {size_bytes} bytes (본체). 외부 weight 파일은 같은 디렉토리에 여러 개 생성되었을 수 있음.")

    # pre-check (경로 기반)
    try:
        checker.check_model(onnx_temp)
        print("  ✅ [PreCheck] 임시 모델 유효합니다.")
    except Exception as e:
        print(f"  ❌ [PreCheck] 실패: {e}")
        return

    # -----------------------
    # (B) in-memory 로드 & IR/OpSet 수정
    # -----------------------
    onnx_model = onnx.load(onnx_temp, load_external_data=True)

    if onnx_model.ir_version < 9:
        print(f"IR 버전 {onnx_model.ir_version} -> 9로 업데이트")
        onnx_model.ir_version = 9

    if not any(op.domain == "" for op in onnx_model.opset_import):
        print("기본 도메인 opset_import 누락 -> 추가.")
        onnx_model.opset_import.append(helper.make_opsetid("", 14))

    # -----------------------
    # (C) 외부 데이터 한 파일로 병합 + 최종 onnx 저장
    # -----------------------
    external_data_helper.convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location="qwen_weights.bin"
    )
    onnx.save(onnx_model, onnx_final)
    print(f"[Final] 최종 ONNX 저장: {onnx_final}")

    # 최종 검증
    try:
        checker.check_model(onnx_final)
        print("  ✅ [FinalCheck] 최종 모델도 유효합니다.")
    except Exception as e:
        print(f"  ❌ [FinalCheck] 실패: {e}")

if __name__ == "__main__":
    main()
