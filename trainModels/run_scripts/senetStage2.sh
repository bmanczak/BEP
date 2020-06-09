#!/bin/bash
source /mnt/server-home/TUE/20175985/miniconda3/etc/profile.d/conda.sh
conda activate DL

python "/mnt/server-home/TUE/20175985/myTask.py" \
                    --job-dir "/mnt/server-home/TUE/20175985/jobs" \
                    --arch "senet50" \
                    --type "fineTune" \
                    --save-path "senetStage2Model" \
                    --optimizer "Adam" \
                    --path-train "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedFull7Classes.npz" \
                    --path-test "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessedBalanced7Classes3k.npz" \
                    --modelpath "" \
                    --batch-size 32 \
                    --balance-weights True \
                    --num-epochs 35


conda deactivate
exit 0
