YOLO v5 requires Pytorch 1.7.0
and yolo v4 needs pytorch 1.4.0

so make two seprate envs for both

in case of yolo-v4 copy my_inference.py file inside the `pytorch-YOLOv4` dir.



# '''
# Important note in case of PyTorch the library will order the GPU ids like following e.g.
# say you have 4 GPUs  and with nvidia-smi you get following output

# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 460.56       Driver Version: 460.56       CUDA Version: 11.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  TITAN RTX           Off  | 00000000:19:00.0 Off |                  N/A |
# | 59%   79C    P2   226W / 280W |  23201MiB / 24220MiB |     88%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
# |   1  TITAN RTX           Off  | 00000000:1A:00.0 Off |                  N/A |
# | 80%   86C    P2   146W / 280W |  23201MiB / 24220MiB |     78%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
# |   2  GeForce GTX TIT...  Off  | 00000000:67:00.0 Off |                  N/A |
# | 57%   84C    P2   131W / 250W |  12068MiB / 12212MiB |     54%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
# |   3  TITAN RTX           Off  | 00000000:68:00.0 Off |                  N/A |
# | 66%   87C    P2   216W / 280W |   2505MiB / 24217MiB |     36%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+

# Now by default you will consider that the RTX GPU ids are 0,1 and 3 and GTX is on id=2 but pytorch
# will order them like follows

# 0  TITAN RTX
# 1  TITAN RTX
# 2  TITAN RTX
# 3  GeForce GTX

# So when you give it the id 2 instead of selecting GTX it'll select the RTX 

# I don't know if its a bug or not but this logic seems to work when you want to use a specific
# GPU withn ID

# '''