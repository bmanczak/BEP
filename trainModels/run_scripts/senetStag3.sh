#!/bin/bash
source /mnt/server-home/TUE/20175985/miniconda3/etc/profile.d/conda.sh
conda activate DL

python "/mnt/server-home/TUE/20175985/myTask.py" \
                    --job-dir "/mnt/server-home/TUE/20175985/jobs" \
                    --arch "senet50" \
                    --type "fineTune" \
                    --save-path "senetStage3Model" \
                    --optimizer "Adam" \
                    --path-test "/mnt/server-home/TUE/20175985/basicRaf/dataNpz/testDataProcessedOriginal.npz" \
                    --path-train "/mnt/server-home/TUE/20175985/basicRaf/dataNpz/trainDataProcessedOriginal.npz" \
                    --modelpath "/mnt/server-home/TUE/20175985/jobs/senetStage3Model" \
                    --batch-size 56 \
                    --balance-weights False \
                    --num-epochs 50


conda deactivate
exit 0
