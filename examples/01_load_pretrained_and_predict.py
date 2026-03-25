from bgnet import BGNet
import numpy as np

x = np.random.randn(1, 19, 2560).astype("float32")
model = BGNet.from_pretrained("./checkpoint_dir")
print(model.predict(x))

