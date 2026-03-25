from bgnet import BGNetConfig, convert_research_mil_checkpoint


ch_names = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
]

config = BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256)

convert_research_mil_checkpoint(
    "best_model.pt",
    "./bgnet-tuab-mil",
    config=config,
    label_map={"normal": 0, "abnormal": 1},
    split_manifest_path="split_manifest.json",
)
