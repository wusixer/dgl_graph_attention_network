### Jupyter lab on different platform

- on HPC through qlogin
`qrsh -l h_rt=120000 -pe smp 4 -l h_vmem=128G`, do screen

- on linux workstation

1. set up connection yourself
`JUPYTER_PORT=1234` change jupyter lab render port
`lsof -ti:1234 | xargs kill -9` ssh unbind port
`ssh -N -f -L 1111:127.0.0.1:1234 zach -N -v -v` ssh port forwarding
Then one can go to localhost:1111

2. use scripts written by hpc team
module load PythonDS
sh /usr/prog/scicomp/pythonds/v0.9/bin/local_jlab.sh

