#/bin/bash
SUBJECT_IDX=$1
CLOTH_TYPE=$2
TRAIN_ACT=$3
SUBSAMPLING_RATE=$5
EPOCHS=$6
CONFIG=$7
python evaluation/eval_chamfer_distance_and_normal_consistency.py --subject-idx ${SUBJECT_IDX} --cloth-split ${CLOTH_TYPE} --act-split ${TRAIN_ACT} --test-subsampling-rate 10 --test-start-offset 5 --exp-suffix _subsample-rate${SUBSAMPLING_RATE}_epochs${EPOCHS} --interpolation ${CONFIG}
