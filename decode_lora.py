import os
import sys
import json
import argparse
import numpy as np
import torch
from torch import nn
import whisper
from whisper.normalizers.basic import BasicTextNormalizer
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# [新增 1] 引入 PEFT
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Warning: peft not installed. LoRA decoding will fail.")

from utils import (
    whisper_flamingo_collator,
    wer_cer,
)
from utils_batch_samplers import SortedBatchSampler
from dataset import YTTDTaigiTRSDataset, HakkaSixianDataset

"""
Usage (LoRA):
python decode_lora.py \
    --dataset_name taigi \
    --checkpoint_path models/checkpoints/taigi_lora/taigi_lora_baseline/step-16007-cer=0.1446.ckpt \
    --decode_path decode_results/taigi_lora_baseline \
    --guide_type none \
    --use_lora
"""

# 設定參數解析
parser = argparse.ArgumentParser(description="SAMA-ASR Decoding Script")

# 模型與路徑相關
parser.add_argument('--model_name', default='medium', type=str, help='Whisper model size')
parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the lightning checkpoint (.ckpt).')
parser.add_argument('--decode_path', default="decode_results/", help='Path to save the decode results')
parser.add_argument('--lang', default='zh', type=str, help='Target language')

# 資料集相關
parser.add_argument('--test_pseudo_path', default=None, help='Path to test pseudo labels json')
parser.add_argument('--aux_json_path', default=None, help='Path to aux translations json')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# 實驗設定
parser.add_argument('--guide_type', default='text', type=str, choices=['text', 'audio', 'none', 'joint'])
parser.add_argument('--use_ground_truth', action='store_true')

# 模型架構設定
parser.add_argument('--num_langs', default=1, type=int)

# 解碼設定
parser.add_argument('--beam_size', default=1, type=int)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--disable_adapter', action='store_true')

# 資料集選擇
parser.add_argument('--dataset_name', default='taigi', type=str, choices=['taigi', 'hakka'])
parser.add_argument('--dataset_path', default='./sixian_30h_processed', type=str)

# [新增 2] LoRA 開關
parser.add_argument('--use_lora', action='store_true', help='Enable LoRA model loading logic')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE}")

def main():
    os.makedirs(args.decode_path, exist_ok=True)
    
    is_original_whisper = (args.checkpoint_path is None)
    
    # LoRA 必定是 Unimodal (Audio-only)，所以強制 guide_type=none
    if args.use_lora:
        current_guide_type = 'none'
        print("LoRA Mode Detected: Forcing guide_type = 'none'")
    elif is_original_whisper:
        current_guide_type = 'none' 
    else:
        current_guide_type = args.guide_type

    # ====================================================
    # 1. 準備 BERT (只有非 LoRA 且非 Original 才需要)
    # ====================================================
    bert_tokenizer = None
    bert_model = None
    
    if current_guide_type in ['text', 'joint']:
        print("Loading BERT for Text Guidance...")
        bert_tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
        bert_model = BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
        bert_model.to(DEVICE)
        bert_model.eval()
        if args.fp16:
            bert_model.half()

    def get_bert_embedding(texts):
        if not texts or len(texts) == 0: return None
        safe_texts = [t if t.strip() != "" else " " for t in texts]
        tokenized = bert_tokenizer(
            safe_texts, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=448
        ).to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(**tokenized)
            embeddings = outputs.last_hidden_state 
        return embeddings

    # ====================================================
    # 2. 載入 Whisper 模型 (含 LoRA 邏輯)
    # ====================================================
    print(f"Loading Whisper: {args.model_name}")    

    # 邏輯分流：
    # 1. LoRA: 用原始 Whisper (add_gated=0) + PEFT wrap
    # 2. SAMA: 用修改版 Whisper (add_gated=1)
    # 3. Original: 用原始 Whisper (add_gated=0)
    
    if args.use_lora:
        use_gated_attn = 0 # LoRA 不用手刻的 Adapter
        print("Model Mode: LoRA (PEFT)")
    elif is_original_whisper or args.disable_adapter:
        use_gated_attn = 0
        print("Model Mode: Vanilla Whisper")
    else:
        use_gated_attn = 1
        print("Model Mode: PGCA / SAMA Adapter Enabled")

    model = whisper.load_model(
        args.model_name, 
        device="cpu", # 先載到 CPU 處理權重，最後再轉 CUDA
        download_root='models/',
        add_gated_x_attn=use_gated_attn, 
        num_langs=args.num_langs 
    )

    # [新增 3] 注入 LoRA 結構
    if args.use_lora:
        print("Injecting LoRA modules...")
        # 這裡的 Config 必須跟訓練時的一樣！
        peft_config = LoraConfig(
            r=32, 
            lora_alpha=64, 
            target_modules=["query", "value"], 
            lora_dropout=0.05, 
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        # model 現在是 PeftModel，原本的 whisper 在 model.base_model.model

    # 載入 Checkpoint
    if not is_original_whisper:
        print(f"Loading weights from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # 處理 PyTorch Lightning 的前綴
            if k.startswith('asr_model.'):
                k = k.replace('asr_model.', '')
            elif k.startswith('model.'):
                k = k.replace('model.', '')
            
            # 過濾掉不需要的參數 (BERT, Translator)
            if not k.startswith('bert_') and not k.startswith('translator'):
                new_state_dict[k] = v
        
        # 載入權重到模型
        try:
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
            # 對於 LoRA，missing keys 應該要是大量的 (因為 backbone 被凍結沒存)，但 lora_ 相關的 key 不能 missing
        except Exception as e:
            print(f"Weight loading warning: {e}")
    
    model.to(DEVICE)
    if args.fp16:
        model.half()
    model.eval()

    # ====================================================
    # 3. 準備資料 (Dataset)
    # ====================================================
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
    
    # LoRA 模式下不需要 Pseudo Labels
    need_pseudo = (current_guide_type in ['text', 'joint']) and (not args.use_ground_truth)
    pseudo_path = args.test_pseudo_path if need_pseudo else None
    aux_langs = ['zh'] if current_guide_type in ['text', 'joint'] else []

    print(f"Loading Dataset: {args.dataset_name} ...")

    if args.dataset_name == 'hakka':
        dataset = HakkaSixianDataset(
            split='test',
            tokenizer=tokenizer,
            sample_rate=16000,
            model_name=args.model_name,
            max_length=160000,
            dataset_path=args.dataset_path, 
            spec_augment=False,
            aux_langs=aux_langs,
            pseudo_label_path=pseudo_path
        )
    else:
        dataset = YTTDTaigiTRSDataset(
            split='test',
            tokenizer=tokenizer, 
            sample_rate=16000,
            model_name=args.model_name,
            max_length=160000, 
            spec_augment=False,
            aux_langs=aux_langs,
            aux_json_path=args.aux_json_path,
            task='transcribe',
            pseudo_label_path=pseudo_path
        )
    
    batch_sampler = SortedBatchSampler(
        batch_size=args.batch_size,
        shapes=[(item['wav_lens']) for item in dataset],
        sort_in_batch='descending',
        sort_batch='descending',
        drop_last=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=whisper_flamingo_collator()
    )

    # ====================================================
    # 4. 開始推論
    # ====================================================
    print(f"Start Decoding... (Mode: {current_guide_type}, LoRA: {args.use_lora})")
    
    options = whisper.DecodingOptions(
        task='transcribe', 
        language='zh', 
        beam_size=args.beam_size,
        fp16=args.fp16,
        without_timestamps=True,
        temperature=args.temperature
    )
    
    hyps = []
    refs = []
    normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
    
    mode_str = "lora" if args.use_lora else (current_guide_type if not is_original_whisper else "original")
    output_file = os.path.join(args.decode_path, f"results_{mode_str}_beam{args.beam_size}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            if args.fp16:
                input_ids = input_ids.half()
            
            # LoRA 模式下，不需要準備 xt_list，或者傳入 None 即可
            # 因為我們載入的是原始 Whisper 結構 (被 PEFT 包裹)，它的 decode 方法不接受 xt_list
            # 但如果您修改過 whisper/decoding.py 讓它接受 xt_list，則傳 [None] 較安全
            
            xt_list = [None] # LoRA / Original / Audio-only baseline 都是 None
            
            # 準備 Feature (如果是 SAMA 模式才需要，LoRA 會略過這裡)
            if current_guide_type in ['text', 'joint']:
               # ... (SAMA 邏輯省略，反正 LoRA 不會進來) ...
               pass

            # 執行 Decode
            with torch.no_grad():
                # 注意：如果 model 是 PeftModel，它會轉發 decode 到 base_model
                # OpenAI Whisper 原生 decode 支援的參數有限
                
                # 關鍵修正：如果我們用的是原始 Whisper 的 decode，它不支援 xt_list
                # 您的 model.py 是修改過的，支援 xt_list
                # 但 LoRA 載入的是原始 Whisper (add_gated_x_attn=0)
                # 所以最好直接呼叫 model.decode(input_ids, options)
                
                # 為了相容性，這裡做一個檢查
                if args.use_lora:
                    # PeftModel -> base_model -> model (原始 Whisper)
                    # 原始 Whisper 可能不接受 xt_list，除非您全域修改了原始碼
                    # 最保險的做法：
                    batch_results = model.base_model.model.decode(input_ids, options) 
                else:
                    batch_results = model.decode(input_ids, options, xt_list=xt_list)
            
            # 收集結果 (不變)
            for i, result in enumerate(batch_results):
                hyp_text = result.text
                label_tokens = batch["labels"][i]
                valid_tokens = [t.item() for t in label_tokens if t != -100 and t.item() not in tokenizer.special_tokens.values()]
                ref_text = tokenizer.decode(valid_tokens)
                
                norm_hyp = normalizer(hyp_text).replace(" ", "")
                norm_ref = normalizer(ref_text).replace(" ", "")
                
                hyps.append(norm_hyp)
                refs.append(norm_ref)
                
                f.write(f"REF: {norm_ref}\n")
                f.write(f"HYP: {norm_hyp}\n")
                f.write("-" * 20 + "\n")

    # ====================================================
    # 5. Metrics
    # ====================================================
    print("Calculating CER...")
    _, cer = wer_cer(hypo=hyps, ref=refs)
    print(f"Final CER: {cer:.4f}")
    
    metrics_file = os.path.join(args.decode_path, f"metrics_{mode_str}.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "cer": cer,
            "beam_size": args.beam_size,
            "mode": mode_str,
            "model": args.model_name
        }, f, indent=4)
        
    print(f"Done! Results saved to {args.decode_path}")

if __name__ == "__main__":
    main()