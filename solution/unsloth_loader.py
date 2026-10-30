"""unsloth-unblocked branch only: load the model through Unsloth with the base task's LoRA.

Soup's own `backend: unsloth` puts LoRA on all 7 projections (target_modules: auto); here Soup's
transformers path is swapped for Unsloth's FastLanguageModel with the same adapter as base
Soup: q_proj and v_proj, the config's r, alpha, dropout and rslora, and no gradient
checkpointing. Not a submission: the jury's harness bans unsloth (check 2).
"""

BASE_TARGETS = ["q_proj", "v_proj"]  # what Soup's transformers path picks for Qwen2 with "auto"


def patch_unsloth_loader():
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.unsloth import load_model_and_tokenizer

    original = SFTTrainerWrapper._setup_transformers

    def setup(self, cfg, tcfg):
        # only the task's case; anything else (other quantization, vocab expansion) stays Soup's
        if tcfg.quantization != "4bit" or getattr(cfg.data, "add_new_tokens", None) \
                or getattr(cfg.data, "new_special_tokens", None):
            return original(self, cfg, tcfg)
        targets = tcfg.lora.target_modules
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=cfg.base,
            max_seq_length=cfg.data.max_length,
            quantization=tcfg.quantization,
            lora_r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=BASE_TARGETS if targets in ("auto", None) else targets,
            use_rslora=tcfg.lora.use_rslora,
            use_gradient_checkpointing=False,  # base Soup trains without it
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    SFTTrainerWrapper._setup_transformers = setup
