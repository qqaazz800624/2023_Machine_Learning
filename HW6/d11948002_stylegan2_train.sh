stylegan2_pytorch --new \
      --name stylegan_test \
      --data /neodata/ML/hw6_dataset/faces \
      --num-train-steps 300 \
      --results_dir /home/u/qqaazz800624/2023_Machine_Learning/HW6/results \
      --models_dir /home/u/qqaazz800624/2023_Machine_Learning/HW6/models \
      --image-size 64 \
      --network-capacity 64 \
      --multi-gpus 