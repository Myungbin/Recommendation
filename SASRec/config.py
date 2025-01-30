class Config:
    dataset: str = "ml-1m"
    train_dir: str = "default"
    batch_size: int = 128
    lr: float = 0.001
    maxlen: int = 200
    hidden_units: int = 50
    num_blocks: int = 2
    num_epochs: int = 1000
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 0.0
    device: str = "cuda"
    inference_only: bool = False
    state_dict_path: str = None


args = Config()