import mne

from bgnet import BGNet

raw = mne.io.read_raw_edf("sample.edf", preload=True)
model = BGNet.from_pretrained("./checkpoint_dir")
result = model.predict_raw_full(
    raw,
    window_seconds=10.0,
    stride_seconds=5.0,
    aggregation="mean",
)
print(result.prediction, result.probabilities)
