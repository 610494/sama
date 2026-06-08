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
python -u train_sama_main.py config/scheme5/taigi/proposed.yaml
python -u train_sama_main.py config/scheme5/hakka/topline_joint.yaml
python -u train_sama_main.py config/scheme5/taigi/topline_joint.yaml

python -u train_sama_main.py config/audio-text/seamless_all_wago.yaml
"""

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class WhisperScheme5MainModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.lang = lang
        
        # ====================================================
        # 1. 模式設定解析
        # ====================================================
        # guide_type: "text" | "audio" | "none" | "joint"
        self.guide_type = getattr(cfg, 'guide_type', 'none') 
        
        # use_translation: 相容舊 config
        self.use_translation = getattr(cfg, 'use_translation', False)
        
        # use_ground_truth: 控制 Text 相關模式下是用 GT 還是 Pseudo
        self.use_ground_truth = getattr(cfg, 'use_ground_truth', False)

        # 再次確認邏輯一致性 (若為 text 模式但關閉 translation，則轉為 none)
        if self.guide_type == 'text' and not self.use_translation:
            self.guide_type = 'none'

        print(f"=== Model Configuration ===")
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
        # 3. 載入 ASR (Whisper)
        # ====================================================
        print(f"Loading Whisper Model: {model_name}")
        self.asr_model = whisper.load_model(
            model_name,
            device='cpu',
            download_root='models/',
            dropout_rate=cfg.dropout_rate,
            add_gated_x_attn=cfg.add_gated_x_attn, # PGCA Adapter 開關
            num_langs = cfg.num_langs,             # Joint 模式請確保 config 設定為 2
        )

        # 顯式凍結 Whisper Backbone (保留 Adapter 與 Projection Layer)
        # 包含 text_projection 和 audio_projection
        train_params = ["gated_x_attn", "attn_gate", "ff", "text_projection", "audio_projection"]
        
        print("Freezing Whisper backbone, keeping Adapters trainable...")
        for n, p in self.asr_model.named_parameters():
            if any(k in n for k in train_params):
                p.requires_grad = True # Adapter & Projections 要訓練
            else:
                p.requires_grad = False # Backbone 凍結

        # 載入 ASR 預訓練權重 (如果有)
        if hasattr(cfg, 'pt_ckpt') and cfg.pt_ckpt and cfg.pt_ckpt != '':
            print(f"Loading ASR backbone from: {cfg.pt_ckpt}")
            self.load_asr_checkpoint(cfg.pt_ckpt)

        # ====================================================
        # 4. 工具
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
                
                # 過濾掉 BERT 和 Translator 的權重
                if not clean_k.startswith('bert_') and not clean_k.startswith('translator'):
                    asr_state_dict[clean_k] = v
            
            msg = self.asr_model.load_state_dict(asr_state_dict, strict=False)
            print(f"ASR weights loaded. Missing keys: {len(msg.missing_keys)}")
        except Exception as e:
            print(f"ASR load warning: {e}")

    def forward(self, x):
        return self.asr_model(x)

    def get_bert_embedding(self, texts):
        # 只有 text 或 joint 模式才需要執行這裡
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
        """
        根據 guide_type 準備 PGCA 的輸入 (xt_list)
        回傳: xt_list, debug_text_list
        """
        xt_list_input = [None] # Default for 'none' mode
        debug_input_text = ["(None)"] * len(batch["input_ids"])

        # 1. 準備 Text (如果模式包含 text)
        text_input = None
        xt_text = None
        if self.guide_type in ['text', 'joint']:
            if self.use_ground_truth:
                # Topline: use dataset translations
                raw_translations = batch["translations"]
                # print(f'raw_translations: {raw_translations}')
                text_input = [t[0] if len(t) > 0 else "" for t in raw_translations]
            else:
                # Experiment: use pseudo translations
                text_input = batch["pseudo_translations"]

            # print(f'text_input: {text_input}')
            # input("")
            xt_text = self.get_bert_embedding(text_input)
            debug_input_text = text_input

        # 2. 根據模式組合 List
        if self.guide_type == 'text':
            # Text Only Mode
            xt_list_input = [xt_text]

        elif self.guide_type == 'audio':
            # Audio Cross Mode
            xt_list_input = [asr_audio_feats]
            debug_input_text = ["(Audio Features)"] * len(batch["input_ids"])

        elif self.guide_type == 'joint':
            # Joint Mode: [Text, Audio]
            # 注意：順序必須對應 model.py 裡的 gated_x_attn_layers 順序
            xt_list_input = [xt_text, asr_audio_feats]

        # guide_type == 'none' 預設已經處理 (xt_list_input = [None])

        # print(f'xt_list_input: {xt_list_input}, debug_input_text: {debug_input_text}')
        return xt_list_input, debug_input_text

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        asr_labels = batch["labels"].long()
        asr_dec_inputs = batch["dec_input_ids"].long()
        
        # 1. Audio Features (Encoder Output)
        asr_audio_feats = self.asr_model.encoder(input_ids)
        
        # 2. Prepare Guide Signals (Text / Audio / Joint / None)
        xt_list_input, _ = self._prepare_guide_signals(batch, asr_audio_feats)
        
        # 3. Decoder
        logits = self.asr_model.decoder(
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

        # 1. Audio Features
        asr_audio_feats = self.asr_model.encoder(input_ids)
        
        # 2. Prepare Guide Signals
        xt_list_input, debug_text_input = self._prepare_guide_signals(batch, asr_audio_feats)
        
        # 3. Decoder
        logits = self.asr_model.decoder(
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
            print(f"\n=== Validation Samples (Mode: {self.guide_type}) ===")
            for i in range(min(2, len(norm_refs))):
                print(f"sample [{i}]:")
                if self.guide_type in ['text', 'joint']:
                    print(f"  Guide (ZH):    {debug_text_input[i]}")
                print(f"  Ref (Taigi):   {norm_refs[i]}")
                print(f"  Hyp (Taigi):   {norm_preds[i]}")
                print("-" * 20)
            
        return cer

    def configure_optimizers(self):
        train_params = ["gated_x_attn", "attn_gate", "ff", "text_projection", "audio_projection"]

        params_to_optimize = []
        for n, p in self.asr_model.named_parameters():
            if p.requires_grad and any(k in n for k in train_params):
                params_to_optimize.append(p)
        
        print(f"Optimizer: Training {len(params_to_optimize)} parameter tensors.")

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
        # 根據模式決定是否需要 pseudo_label_path
        pseudo_path = getattr(self.cfg, 'train_pseudo_path', None)
        dataset_name = getattr(self.cfg, 'dataset_name', 'taigi') # 預設為 taigi 以相容舊 config
        
        subset_hours = getattr(self.cfg, 'train_subset_hours', None)
        
        if dataset_name == 'hakka':
            dataset = HakkaSixianDataset(
                split='train',
                tokenizer=self.tokenizer,
                sample_rate=SAMPLE_RATE,
                model_name=self.model_name,
                max_length=self.cfg.audio_max_length,
                dataset_path=getattr(self.cfg, 'dataset_path', './sixian_30h_processed'), # 從 config 讀路徑
                spec_augment=self.cfg.spec_augment,
                lang=self.cfg.lang,
                aux_langs=self.cfg.aux_langs,
                # [新增] 必須把 pseudo path 傳進去
                pseudo_label_path=pseudo_path,
                train_subset_hours=subset_hours
            )
        else:
            # 原本的 Taigi Dataset 邏輯
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
                spec_augment=False, # Test 不做 Augmentation
                lang=self.cfg.lang,
                aux_langs=self.cfg.aux_langs,
                # [新增] 必須把 pseudo path 傳進去
                pseudo_label_path=pseudo_path
            )
        else:
            # 原本的 Taigi Dataset
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

    print("=== Scheme 5 Combined Main Script ===")
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

    model = WhisperScheme5MainModule(cfg, cfg.model_name, cfg.lang)
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