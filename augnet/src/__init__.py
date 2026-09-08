# e3nn (<=0.4.x) torch.load()s cached constants at import; torch>=2.6 defaults
# weights_only=True and rejects them. Every src.* import passes through here, so
# set it before any module pulls in e3nn.
import os as _os

_os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
