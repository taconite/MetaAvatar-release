#/bin/bash
SUBJECT_IDX=$1
CLOTH_TYPE=$2
TRAIN_ACT=$3
TEST_ACT=$4
SUBSAMPLING_RATE=$5
EPOCHS=$6
CONFIG=$7
python fine_tune_avatar.py --num-workers 8 --subject-idx ${SUBJECT_IDX} --train-cloth-split ${CLOTH_TYPE} --train-act-split ${TRAIN_ACT} --test-act-split ${TEST_ACT} --subsampling-rate ${SUBSAMPLING_RATE} --test-subsampling-rate 1 --test-start-offset 0 --exp-suffix _subsample-rate${SUBSAMPLING_RATE}_epochs${EPOCHS} --optim-epochs ${EPOCHS} --epochs-per-run 10000 ${CONFIG}
