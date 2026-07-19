# Sample datasets

Tiny JSONL fixtures for quick starts and example configs.

| File | Methods |
|------|---------|
| `train.jsonl` | SFT, LoRA, QLoRA, PPO, GRPO |
| `preferences.jsonl` | DPO (`prompt` / `chosen` / `rejected`) |

Point `data.dataset.path` at these files, or replace them with your own data.

Paths in `configs/examples/` use `../../data/...` so they resolve from the example config directory. Scaffolded configs use `./data/...` next to the config file.
