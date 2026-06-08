import os
import sys
import yaml
import types
import torch
from torch import nn
import whisper
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# 引入 PEFT
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Error: 請先安裝 peft 庫 (pip install peft)")
    sys.exit(1)

from utils import (
    whisper_flamingo_collator,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
import wandb
from whisper.normalizers.basic import BasicTextNormalizer
from dataset_subset import YTTDTaigiTRSDataset, HakkaSixianDataset

"""
Usage:
python -u train_sama_lora.py config/scheme5/taigi/proposed.yaml
"""

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class WhisperSAMALoRAModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.lang = lang
        
        # ====================================================
        # 1. 模式設定解析 (保留 SAMA 邏輯)
        # ====================================================
        self.guide_type = getattr(cfg, 'guide_type', 'none') 
        self.use_translation = getattr(cfg, 'use_translation', False)
        self.use_ground_truth = getattr(cfg, 'use_ground_truth', False)

        if self.guide_type == 'text' and not self.use_translation:
            self.guide_type = 'none'

        print(f"=== SAMA + LoRA Configuration ===")
        print(f"Guide Type: {self.guide_type}")
        if self.guide_type in ['text', 'joint']:
            source = "Ground Truth (Topline)" if self.use_ground_truth else "Pseudo Translation"
            print(f"Text Source: {source}")

        # ====================================================
        # 2. 載入 BERT (Text 或 Joint 模式需要)
        # ====================================================
        if self.guide_type in ['text', 'joint']:
            print("Loading Multilingual BERT...")
            self.bert_tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
            self.bert_model = BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
            self.bert_model.eval()
            for p in self.bert_model.parameters():
                p.requires_grad = False
        else:
            self.bert_tokenizer = None
            self.bert_model = None
        
        # ====================================================
        # 3. 載入 ASR (Whisper) 並啟用 SAMA
        # ====================================================
        print(f"Loading Whisper Model: {model_name} with SAMA Adapter")
        self.asr_model = whisper.load_model(
            model_name,
            device='cpu',
            download_root='models/',
            dropout_rate=cfg.dropout_rate,
            add_gated_x_attn=cfg.add_gated_x_attn, # 根據 config 啟用 PGCA Adapter
            num_langs=cfg.num_langs,             
        )

        # 載入 ASR 預訓練權重 (如果有 SAMA 預訓練權重可於此載入)
        if hasattr(cfg, 'pt_ckpt') and cfg.pt_ckpt and cfg.pt_ckpt != '':
            print(f"Loading ASR backbone from: {cfg.pt_ckpt}")
            self.load_asr_checkpoint(cfg.pt_ckpt)

        # ====================================================
        # 4. 注入 LoRA (覆蓋 Attention 與 Backbone FFN)
        # ====================================================
        # OpenAI Whisper 的 Attention Q,V 為 "query", "value"
        # 原始 FFN 為 "mlp.0" (fc1) 與 "mlp.2" (fc2)
        target_modules = ["query", "value"]
        
        peft_config = LoraConfig(
            r=32,                   
            lora_alpha=64,          
            target_modules=target_modules, 
            lora_dropout=0.05,
            bias="none",
            modules_to_save=[],     
        )
        
        print(f"Injecting LoRA into: {target_modules} ...")
        # 這一步會自動凍結 Backbone，並加入 LoRA 權重 (LoRA 權重預設為 requires_grad=True)
        self.asr_model = get_peft_model(self.asr_model, peft_config)

        # ====================================================
        # 5. 強制解凍 SAMA 模組
        # ====================================================
        # peft 把 backbone 凍結了，但我們希望 SAMA 模組與 LoRA 一起被訓練
        sama_train_params = ["gated_x_attn", "attn_gate", "ff", "text_projection", "audio_projection"]
        
        print("Unfreezing SAMA specific modules...")
        unfrozen_sama_count = 0
        for n, p in self.asr_model.named_parameters():
            if any(k in n for k in sama_train_params):
                p.requires_grad = True
                unfrozen_sama_count += 1
                
        print(f"Unfrozen {unfrozen_sama_count} SAMA parameter tensors.")
        self.asr_model.print_trainable_parameters()

        # ====================================================
        # 6. 工具
        # ====================================================
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=lang, task='transcribe')
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def load_asr_checkpoint(self, ckpt_path):
        try:
            asr_ckpt = torch.load(ckpt_path, map_location='cpu')
            asr_state_dict = {}
            src_dict = asr_ckpt['state_dict'] if 'state_dict' in asr_ckpt else asr_ckpt
            
            for k, v in src_dict.items():
                if k.startswith('asr_model.'):
                    clean_k = k.replace('asr_model.', '')
                else:
                    clean_k = k.replace('model.', '')
                
                if not clean_k.startswith('bert_') and not clean_k.startswith('translator'):
                    asr_state_dict[clean_k] = v
            
            msg = self.asr_model.load_state_dict(asr_state_dict, strict=False)
            print(f"ASR weights loaded. Missing keys: {len(msg.missing_keys)}")
        except Exception as e:
            print(f"ASR load warning: {e}")

    def forward(self, x):
        return self.asr_model(x)

    def get_bert_embedding(self, texts):
        if self.guide_type not in ['text', 'joint']:
            return None
        
        if not texts or len(texts) == 0: 
            return None
        
        safe_texts = [t if t.strip() != "" else " " for t in texts]
        tokenized = self.bert_tokenizer(
            safe_texts, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=448
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**tokenized)
            embeddings = outputs.last_hidden_state 

        return embeddings.detach()

    def _prepare_guide_signals(self, batch, asr_audio_feats):
        xt_list_input = [None]
        debug_input_text = ["(None)"] * len(batch["input_ids"])

        text_input = None
        xt_text = None
        if self.guide_type in ['text', 'joint']:
            if self.use_ground_truth:
                raw_translations = batch["translations"]
                text_input = [t[0] if len(t) > 0 else "" for t in raw_translations]
            else:
                text_input = batch["pseudo_translations"]

            xt_text = self.get_bert_embedding(text_input)
            debug_input_text = text_input

        if self.guide_type == 'text':
            xt_list_input = [xt_text]
        elif self.guide_type == 'audio':
            xt_list_input = [asr_audio_feats]
            debug_input_text = ["(Audio Features)"] * len(batch["input_ids"])
        elif self.guide_type == 'joint':
            xt_list_input = [xt_text, asr_audio_feats]

        return xt_list_input, debug_input_text

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        asr_labels = batch["labels"].long()
        asr_dec_inputs = batch["dec_input_ids"].long()
        
        # 透過 base_model.model 存取被 PEFT 包裝的原始 Whisper 結構
        base_model = self.asr_model.base_model.model

        # 1. Audio Features
        asr_audio_feats = base_model.encoder(input_ids)
        
        # 2. Prepare Guide Signals
        xt_list_input, _ = self._prepare_guide_signals(batch, asr_audio_feats)
        
        # 3. Decoder
        logits = base_model.decoder(
            asr_dec_inputs,
            asr_audio_feats,
            xt_list=xt_list_input
        )
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), asr_labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        asr_labels = batch["labels"].long()
        asr_dec_inputs = batch["dec_input_ids"].long()

        base_model = self.asr_model.base_model.model

        # 1. Audio Features
        asr_audio_feats = base_model.encoder(input_ids)
        
        # 2. Prepare Guide Signals
        xt_list_input, debug_text_input = self._prepare_guide_signals(batch, asr_audio_feats)
        
        # 3. Decoder
        logits = base_model.decoder(
            asr_dec_inputs,
            asr_audio_feats,
            xt_list=xt_list_input
        )
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), asr_labels.view(-1))
        
        # Calculate CER
        tokens = torch.argmax(logits, dim=2)
        
        # EOT handling
        eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
        for i in range(eot_find.shape[0]):
            if torch.any(eot_find[i] == 1):
                first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).to(self.device) * eot_find[i], dim=0, keepdim=True)
                tokens[i, torch.arange(eot_find.shape[1]).to(self.device) > first_eot] = self.tokenizer.eot

        target_texts = [self.tokenizer.decode([t for t in l if t != -100 and t not in self.special_token_set]) for l in asr_labels]
        pred_texts = [self.tokenizer.decode([t for t in p if t not in self.special_token_set]) for p in tokens]
        
        norm_refs = [self.text_normalizer(t).replace(" ", "") for t in target_texts]
        norm_preds = [self.text_normalizer(t).replace(" ", "") for t in pred_texts]
        
        _, cer = wer_cer(hypo=norm_preds, ref=norm_refs)
        
        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/cer", cer, on_step=False, prog_bar=True, logger=True, sync_dist=True)
            
        if batch_id == 0:
            print(f"\n=== Validation Samples (SAMA + LoRA, Mode: {self.guide_type}) ===")
            for i in range(min(2, len(norm_refs))):
                print(f"sample [{i}]:")
                if self.guide_type in ['text', 'joint']:
                    print(f"  Guide (ZH):    {debug_text_input[i]}")
                print(f"  Ref:           {norm_refs[i]}")
                print(f"  Hyp:           {norm_preds[i]}")
                print("-" * 20)
            
        return cer

    def configure_optimizers(self):
        # 收集所有 requires_grad=True 的參數 (包含 LoRA 權重與解凍的 SAMA 權重)
        params_to_optimize = [p for p in self.asr_model.parameters() if p.requires_grad]
        
        print(f"Optimizer: Training {len(params_to_optimize)} parameter tensors (LoRA + SAMA).")

        optimizer = AdamW(
            params_to_optimize,
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        pseudo_path = getattr(self.cfg, 'train_pseudo_path', None)
        dataset_name = getattr(self.cfg, 'dataset_name', 'taigi') 
        subset_hours = getattr(self.cfg, 'train_subset_hours', None)
        
        if dataset_name == 'hakka':
            dataset = HakkaSixianDataset(
                split='train',
                tokenizer=self.tokenizer,
                sample_rate=SAMPLE_RATE,
                model_name=self.model_name,
                max_length=self.cfg.audio_max_length,
                dataset_path=getattr(self.cfg, 'dataset_path', './sixian_30h_processed'),
                spec_augment=self.cfg.spec_augment,
                lang=self.cfg.lang,
                aux_langs=self.cfg.aux_langs,
                pseudo_label_path=pseudo_path,
                train_subset_hours=subset_hours
            )
        else:
            dataset = YTTDTaigiTRSDataset(
                split='train',
                tokenizer=self.tokenizer, 
                sample_rate=SAMPLE_RATE,
                model_name=self.model_name,
                max_length=self.cfg.audio_max_length,
                spec_augment=self.cfg.spec_augment,
                aux_langs=self.cfg.aux_langs,
                aux_json_path=self.cfg.aux_json_path,
                task='transcribe',
                pseudo_label_path=pseudo_path,
                train_subset_hours=subset_hours
            )

        batch_sampler = SortedBatchSampler(
            batch_size = self.cfg.batch_size,
            shapes=[(item['wav_lens']) for item in dataset],
            sort_in_batch='descending',
            sort_batch='descending',
            drop_last=True
        )
        if self.cfg.num_devices > 1:
            batch_sampler = DistributedSamplerWrapper(batch_sampler)

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_worker,
            collate_fn=whisper_flamingo_collator()
        )

    def test_dataloader(self):
        pseudo_path = getattr(self.cfg, 'test_pseudo_path', None)
        dataset_name = getattr(self.cfg, 'dataset_name', 'taigi')
        
        if dataset_name == 'hakka':
            dataset = HakkaSixianDataset(
                split='test',
                tokenizer=self.tokenizer,
                sample_rate=SAMPLE_RATE,
                model_name=self.model_name,
                max_length=self.cfg.audio_max_length,
                dataset_path=getattr(self.cfg, 'dataset_path', './sixian_30h_processed'),
                spec_augment=False,
                lang=self.cfg.lang,
                aux_langs=self.cfg.aux_langs,
                pseudo_label_path=pseudo_path
            )
        else:
            dataset = YTTDTaigiTRSDataset(
                split='test',
                tokenizer=self.tokenizer, 
                sample_rate=SAMPLE_RATE,
                model_name=self.model_name,
                max_length=self.cfg.audio_max_length,
                spec_augment=False,
                aux_langs=self.cfg.aux_langs,
                aux_json_path=self.cfg.aux_json_path,
                task='transcribe',
                pseudo_label_path=pseudo_path
            )
        batch_sampler = SortedBatchSampler(
            batch_size = self.cfg.batch_size,
            shapes=[(item['wav_lens']) for item in dataset],
            sort_in_batch='descending',
            sort_batch='descending',
            drop_last=False
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_worker,
            collate_fn=whisper_flamingo_collator()
        )

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print("=== SAMA + LoRA Hybrid Training Script ===")
    print("Config:", cfg)

    wandb.init(
        project="SAMA-ASR", 
        config=cfg,
        name=cfg.train_id,
        # mode="disabled"
    )

    callback_list = setup_logging_and_checkpoint(
        cfg.check_output_dir, cfg.train_name, cfg.train_id,
        "val/cer", cfg.filename
    )

    model = WhisperSAMALoRAModule(cfg, cfg.model_name, cfg.lang)
    wandb_logger = WandbLogger()

    strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"

    trainer = Trainer(
        precision=cfg.precision,
        strategy=strategy,
        accelerator="gpu",
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=wandb_logger,
        callbacks=callback_list,
        num_sanity_val_steps=0, 
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps),
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        sync_batchnorm=True,
    )

    trainer.validate(model=model, dataloaders=model.test_dataloader()) 
    trainer.fit(model, val_dataloaders=model.test_dataloader())

    wandb.finish()