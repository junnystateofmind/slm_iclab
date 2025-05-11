import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def create_calibration_data(
    csv_path="train.csv",
    text_col="input",
    max_rows=500,
    hf_model_name="junnystateofmind/Llama_3.2-3B_merged_rank256_alpha512",
    seq_len=128,
    output_dir="calib_bin",
    calib_txt_path="calibration_data.txt"
):
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        df = df.dropna(subset=[text_col]).head(max_rows)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  

        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)

        calib_lines = []

        for i, row in enumerate(df.itertuples(), start=1):
            text = str(getattr(row, text_col))

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="np"
            )

            input_ids = tokens["input_ids"].astype(np.int32)
            attn_mask = tokens["attention_mask"].astype(np.int32)

            input_ids_path = os.path.join(abs_output_dir, f"input_ids_{i}.bin")
            attn_mask_path = os.path.join(abs_output_dir, f"attention_mask_{i}.bin")

            input_ids.tofile(input_ids_path)
            attn_mask.tofile(attn_mask_path)

            # ✅ SNPE가 올바르게 인식하는 형식으로 변경
            calib_lines.append(f"{input_ids_path} {attn_mask_path}")

        abs_calib_txt_path = os.path.abspath(calib_txt_path)
        with open(abs_calib_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(calib_lines))

        print(f"[✅ DONE] {len(df)}개 문장에서 .bin 파일 생성 완료!")
        print(f" - .bin 파일들: '{abs_output_dir}' 디렉토리에 저장됨.")
        print(f" - calibration_data.txt='{abs_calib_txt_path}' (총 {len(calib_lines)}줄)")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    create_calibration_data()
