import argparse
import json
import os
import sys

import numpy as np
import torch
import whisper
from dataset import HakkaSixianDataset, YTTDTaigiTRSDataset
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from utils import wer_cer, whisper_flamingo_collator
from utils_batch_samplers import SortedBatchSampler
from whisper.normalizers.basic import BasicTextNormalizer

"""
python decode_sama_main.py \
    --dataset_name hakka \
    --checkpoint_path models/checkpoints/ft_hakka/whisper_ft_hakka_medium/step-45000-cer=0.0966.ckpt \
    --decode_path decode_results/hakka_fine_tuned \
    --guide_type none \
    --disable_adapter

python decode_sama_main.py \
    --dataset_name taigi \
    --checkpoint_path models/checkpoints/scheme5/text_baseline/step-32000-cer=0.1868.ckpt \
    --decode_path decode_results/taigi_text_baseline_check_gulid_text \
    --guide_type text \
    --batch_size 16

python decode_sama_main.py \
    --dataset_name taigi \
    --checkpoint_path result/checkpoints/scheme5/text_baseline_kd/step-30000-cer=0.2202.ckpt \
    --decode_path decode_results/taigi_text_baseline_kd_check_gulid_text \
    --guide_type text \
    --batch_size 16

python decode_sama_main.py \
    --dataset_name hakka \
    --checkpoint_path result/checkpoints/hakka_scheme5/hakka_text_baseline_kd_sentence/step-17400-cer=0.1829.ckpt \
    --decode_path decode_results/hakka_text_baseline_kd_sentence_check_gulid_text \
    --guide_type text \
    --batch_size 16

"""

# 設定參數解析
parser = argparse.ArgumentParser(description="SAMA-ASR Decoding Script")

# 模型與路徑相關
parser.add_argument('--model_name', default='medium', type=str, help='Whisper model size')
parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the lightning checkpoint (.ckpt). If None, use Original Whisper.')

parser.add_argument('--decode_path', default="decode_results/", help='Path to save the decode results')
parser.add_argument('--lang', default='zh', type=str, help='Target language (zh for Mandarin/Taigi context)')

# 資料集相關
parser.add_argument('--test_pseudo_path', default=None, help='Path to test pseudo labels json (for experiment)')
parser.add_argument('--aux_json_path', default=None, help='Path to aux translations json')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# 實驗設定
parser.add_argument('--guide_type', default='text', type=str, choices=['text', 'audio', 'none', 'joint'], 
                    help='Type of guidance. For Original Whisper, this is ignored (treated as none).')
parser.add_argument('--use_ground_truth', action='store_true', 
                    help='If True, use Ground Truth translation (Topline); otherwise use Pseudo (Experiment)')

# 模型架構設定
parser.add_argument('--num_langs', default=1, type=int, help='Number of guidance inputs supported by the model')

# 解碼設定
parser.add_argument('--beam_size', default=1, type=int, help='Beam size for decoding')
parser.add_argument('--fp16', action='store_true', help='Use FP16 for inference')
parser.add_argument('--temperature', default=0.0, type=float, help='Sampling temperature')
# [新增] 用於載入 Vanilla Whisper Fine-tuning Checkpoint
parser.add_argument('--disable_adapter', action='store_true', 
                    help='If True, force add_gated_x_attn=0 (for Vanilla Fine-tuned models)')

parser.add_argument('--dataset_name', default='taigi', type=str, choices=['taigi', 'hakka'], help='Dataset to decode')
parser.add_argument('--dataset_path', default='./sixian_30h_processed', type=str, help='Path to Hakka dataset (processed disk format)')

args = parser.parse_args()

# 設定 Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE}")

def main():
    os.makedirs(args.decode_path, exist_ok=True)
    
    # [新增] 判斷是否為 Original Whisper 模式
    is_original_whisper = (args.checkpoint_path is None)
    
    if is_original_whisper:
        print("\n" + "="*30)
        print("  Running ORIGINAL WHISPER  ")
        print("="*30 + "\n")
        # 強制將 guide_type 視為 none，避免載入 BERT
        current_guide_type = 'none' 
    else:
        current_guide_type = args.guide_type

    # ====================================================
    # 1. 準備 BERT (Text 或 Joint 模式下需要)
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
    # 2. 載入 Whisper 模型
    # ====================================================
    print(f"Loading Whisper: {args.model_name}")    

    # [修改] 判斷邏輯
    # 如果是 (1) 沒有 checkpoint (OpenAI 原版) 或 (2) 明確指定關閉 Adapter (Vanilla FT)
    # 則將模型架構設為原始架構 (0)
    if is_original_whisper or args.disable_adapter:
        use_gated_attn = 0
        num_langs = 0
        print("Model Mode: Vanilla Whisper (No PGCA/Adapter)")
    else:
        # SAMA-ASR / Baseline: 開啟 Gated X Attn (1)
        use_gated_attn = 1
        num_langs = args.num_langs
        print("Model Mode: PGCA / Adapter Enabled")

    print(f"Initializing model with add_gated_x_attn={use_gated_attn}, num_langs={num_langs}")
    
    # print(f'before whisper.load_model')
    # input("")
    model = whisper.load_model(
        args.model_name, 
        device=DEVICE, 
        download_root='models/',
        add_gated_x_attn=use_gated_attn, # 0 for Original, 1 for SAMA-ASR
        num_langs=num_langs 
    )

    # 載入 Checkpoint (僅在非 Original 模式下)
    if not is_original_whisper:
        print(f"Loading weights from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('asr_model.'):
                k = k.replace('asr_model.', '')
            elif k.startswith('model.'):
                k = k.replace('model.', '')
            if not k.startswith('bert_') and not k.startswith('translator'):
                new_state_dict[k] = v
                
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Weight loading warning: {e}")
    else:
        print("Using Pre-trained OpenAI Whisper weights.")

    if args.fp16:
        model.half()
    model.eval()

    # ====================================================
    # 3. 準備資料
    # ====================================================
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='zh', task='transcribe')
    
    # 只有在 Proposed/Joint 且不是 GT 模式時才需要 pseudo path
    need_pseudo = (current_guide_type in ['text', 'joint']) and (not args.use_ground_truth)
    pseudo_path = args.test_pseudo_path if need_pseudo else None
    
    # 決定 Dataset 是否需要載入輔助語言
    aux_langs = ['zh'] if current_guide_type in ['text', 'joint'] else []

    print(f"Loading Dataset: {args.dataset_name} ...")

    # [修改 3] 根據參數載入對應的 Dataset Class
    if args.dataset_name == 'hakka':
        dataset = HakkaSixianDataset(
            split='test',
            tokenizer=tokenizer,
            sample_rate=16000,
            model_name=args.model_name,
            max_length=160000,
            dataset_path=args.dataset_path, # 傳入客語資料集路徑
            spec_augment=False,
            aux_langs=aux_langs,
            # [修正] 必須補上這行，不然讀不到 Auto Translation
            pseudo_label_path=pseudo_path
        )
    else:
        # 原本的台語 Dataset 邏輯
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
    # print(f'args.batch_size: {args.batch_size}')
    # input("")
    batch_sampler = SortedBatchSampler(
        batch_size=args.batch_size,
        shapes=[(item['wav_lens']) for item in dataset],
        sort_in_batch='descending',
        sort_batch='descending',
        drop_last=False
    )
    
    # Batch Sampler 和 DataLoader 保持不變
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=whisper_flamingo_collator()
    )

    # ====================================================
    # 4. 開始推論 (Decoding Loop)
    # ====================================================
    print(f"Start Decoding... (Mode: {'Original' if is_original_whisper else current_guide_type})")
    
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
    
    # 輸出檔名
    mode_str = "original" if is_original_whisper else args.guide_type
    output_file = os.path.join(args.decode_path, f"results_{mode_str}_beam{args.beam_size}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            if args.fp16:
                input_ids = input_ids.half()
                
            xt_text = None
            xt_audio = None
            text_input_for_log = None

            # 1. 準備 Text Feature
            if current_guide_type in ['text', 'joint']:
                if args.use_ground_truth:
                    raw_translations = batch["translations"]
                    text_input = [t[0] if len(t) > 0 else "" for t in raw_translations]
                else:
                    text_input = batch["pseudo_translations"]
                
                # print(f'text_input: {text_input}')
                # input("")
                text_input_for_log = text_input
                xt = get_bert_embedding(text_input)
                if args.fp16 and xt is not None:
                    xt = xt.half()
                xt_text = xt

            # 2. 準備 Audio Feature
            if current_guide_type in ['audio', 'joint']:
                with torch.no_grad():
                    xt = model.encoder(input_ids)
                    xt_audio = xt

            # 3. 組合 xt_list
            xt_list = []
            if current_guide_type == 'text':
                xt_list = [xt_text]
            elif current_guide_type == 'audio':
                xt_list = [xt_audio]
            elif current_guide_type == 'joint':
                xt_list = [xt_text, xt_audio]
            else:
                # None or Original Whisper
                xt_list = [None]
            
            # 執行 Decode
            with torch.no_grad():
                # 即使是 Original Whisper，我們的 model.py 也支援傳入 xt_list (會被忽略)
                batch_results = model.decode(input_ids, options, xt_list=xt_list)
            
            # 收集結果
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
                
                if text_input_for_log is not None:
                    f.write(f"GUI: {text_input_for_log[i]}\n")
                f.write("-" * 20 + "\n")

    # ====================================================
    # 5. 計算 Metrics (CER)
    # ====================================================
    print("Calculating CER...")
    _, cer = wer_cer(hypo=hyps, ref=refs)
    
    print(f"Final CER: {cer:.4f}")
    
    metrics_file = os.path.join(args.decode_path, f"metrics_{mode_str}.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "cer": cer,
            "beam_size": args.beam_size,
            "guide_type": mode_str,
            "model": args.model_name
        }, f, indent=4)
        
    print(f"Done! Results saved to {args.decode_path}")

if __name__ == "__main__":
    main()