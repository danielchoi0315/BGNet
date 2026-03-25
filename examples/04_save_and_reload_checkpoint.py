from bgnet import BGNet

ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
model = BGNet.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256)
model.save_pretrained("./bgnet-demo")
reloaded = BGNet.from_pretrained("./bgnet-demo")
print(type(reloaded).__name__)

