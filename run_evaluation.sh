#/bin/bash
while read params
do
    [ "${params:0:1}" = "#" ] && continue
    bash run_single_evaluation.sh ${params};
done < jobs/splits
