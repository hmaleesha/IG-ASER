import socket

import torch

DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'

host = socket.gethostname()

IEMOCAP_DIR = 'C:\\Users\\75001023\\OneDrive - Murdoch University (1)\\Desktop\\tasks\\experiments_obj1\\emo\\emoDARTS\\prepare-dataset'
ESD_DIR = ''
MSPIMPROV_DIR = ''
MSPPODCAST_DIR = ''
