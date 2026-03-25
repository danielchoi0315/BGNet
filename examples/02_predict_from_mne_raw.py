import mne

from bgnet import BGNet

raw = mne.io.read_raw_edf("sample.edf", preload=True)
model = BGNet.from_pretrained("./checkpoint_dir")
print(model.predict(raw))

