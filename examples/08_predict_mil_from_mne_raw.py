import mne

from bgnet import BGNetMILModel


raw = mne.io.read_raw_edf("sample.edf", preload=True)
model = BGNetMILModel.from_pretrained("./bgnet-tuab-mil")
result = model.predict_raw_full(raw, window_seconds=10.0, stride_seconds=5.0)
print(result.prediction, result.probabilities, result.attention_weights)
