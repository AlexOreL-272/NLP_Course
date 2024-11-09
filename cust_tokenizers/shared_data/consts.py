import numpy as np

UNK_TOKEN = "<UNK>"  # unknown token
PAD_TOKEN = "<PAD>"  # padding token
SOW_TOKEN = "<SOW>"  # start of word token
EOW_TOKEN = "<EOW>"  # end of word token

LARGE_INT = ~np.uint64(0)   # large integer. In binary it is 64 ones
