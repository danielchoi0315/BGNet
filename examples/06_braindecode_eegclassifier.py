from braindecode import EEGClassifier
from bgnet.braindecode import BraindecodeBGNet


ch_names = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
]

module = BraindecodeBGNet(
    n_outputs=2,
    n_chans=len(ch_names),
    sfreq=256,
    channel_names=ch_names,
    preset="clinical",
)

clf = EEGClassifier(
    module,
    criterion="cross_entropy",
    optimizer__lr=1e-3,
    train_split=None,
)

print(clf)
