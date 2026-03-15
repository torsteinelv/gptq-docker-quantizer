import os
import random
from itertools import islice
from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer
from huggingface_hub import login, HfApi
from gptqmodel import GPTQModel, QuantizeConfig

# Laster inn .env filen automatisk (hvis den finnes)
load_dotenv()

def _build_calib_texts(n_total: int) -> List[str]:
    from datasets import load_dataset
    n_norsk = int(n_total * 0.60)
    n_code = int(n_total * 0.25)
    n_coco = n_total - n_norsk - n_code
    texts = []

    if n_norsk > 0:
        ds = load_dataset("NbAiLab/norwegian-alpaca", split=f"train[:{n_norsk}]")
        for r in ds:
            texts.append(f"Instruksjon: {r.get('instruction', '')}\nSvar: {r.get('output', '')}")

    if n_code > 0:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        for r in islice(ds, n_code):
            texts.append((r.get("query", "") or "") + "\n" + (r.get("answer", "") or ""))

    if n_coco > 0:
        ds = load_dataset("sentence-transformers/coco-captions", split="train", streaming=True)
        for r in islice(ds, n_coco):
            texts.append(f"Describe this image: {r.get('caption1', '')}")

    random.shuffle(texts)
    return texts[:n_total]

def main():
    # Henter variabler fra miljøet (.env) med standardverdier som fallback
    hf_token = os.environ.get("HF_TOKEN")
    source_model_id = os.environ.get("SOURCE_MODEL_ID", "Qwen/Qwen2.5-7B")
    target_repo_id = os.environ.get("TARGET_REPO_ID", "telvenes/My-Quantized-Model")
    quant_bits = int(os.environ.get("QUANT_BITS", 4))
    group_size = int(os.environ.get("GROUP_SIZE", 128))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", 1024))
    calib_samples = int(os.environ.get("CALIB_SAMPLES", 1024))
    quant_path = "/tmp/quantized-model-final"

    if not hf_token:
        raise ValueError("❌ HF_TOKEN er ikke satt! Sjekk .env-filen din.")

    print(f"--- STARTER KVANTISERING ---")
    print(f"Modell inn: {source_model_id}")
    print(f"Modell ut: {target_repo_id}")
    print(f"Innstillinger: {quant_bits}-bit, Group size: {group_size}, Max len: {max_seq_len}, Samples: {calib_samples}")
    
    print("Logger inn på Hugging Face...")
    login(token=hf_token)

    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True, use_fast=True)

    print(f"Bygger kalibreringsdata ({calib_samples} setninger)...")
    samples = _build_calib_texts(calib_samples)

    print("Klargjør kalibreringsdata for GPTQModel...")
    calib_data = []
    for text in samples:
        encoded = tokenizer(text, return_tensors="pt", max_length=max_seq_len, truncation=True)
        calib_data.append({
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        })

    print(f"Klargjør QuantizeConfig...")
    quantize_config = QuantizeConfig(
        bits=quant_bits,
        group_size=group_size, 
        sym=True,
        desc_act=False,
        device="cuda:0", 
        offload_to_disk=False,
    )

    print("Laster den ukvantiserte modellen...")
    model = GPTQModel.load(
        source_model_id,
        quantize_config=quantize_config,
        trust_remote_code=True,
        experts_implementation="eager",
    )

    print("Starter kvantiseringen. La GPU-en jobbe! ☕...")
    model.quantize(calib_data)

    print(f"Lagrer kvantisert modell og tokenizer lokalt til {quant_path}...")
    model.save(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print("Lagring vellykket! Starter opplasting via HfApi...")
    api = HfApi(token=hf_token)
    
    api.create_repo(repo_id=target_repo_id, exist_ok=True, private=False)
    
    api.upload_folder(
        folder_path=quant_path,
        repo_id=target_repo_id,
        commit_message=f"🚀 Kvantisert til {quant_bits}-bit med {calib_samples} samples (Docker/Lokal)"
    )

    print(f"✅ MODELL FULLFØRT OG LASTET OPP! Sjekk: https://huggingface.co/{target_repo_id}")

if __name__ == "__main__":
    main()
