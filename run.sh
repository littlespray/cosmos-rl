source /opt/venv/cosmos_rl/bin/activate
COSMOS_NCCL_TIMEOUT_MS=60000000 COSMOS_GLOO_TIMEOUT=60000 cosmos-rl --config config.toml --port 8000 --rdzv-port 29345