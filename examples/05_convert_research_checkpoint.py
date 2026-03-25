from bgnet import BGNetConfig, convert_research_checkpoint

ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
config = BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256)
# This converter is for encoder / window checkpoints only. Record-level MIL checkpoints with a
# separate mil_head_state are intentionally rejected by the public package.
convert_research_checkpoint("best_model.pt", "./bgnet-converted", config=config)
