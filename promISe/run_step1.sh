# Set the variables
BENCHMARK=mmlu
AUGMENT_SIZE=50
ROUND=0

# Run the Python script
python extensive_search.py \
    --benchmark ${BENCHMARK} \
    --augment_size ${AUGMENT_SIZE} \
    --round ${ROUND}