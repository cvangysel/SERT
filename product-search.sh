#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source "${SCRIPT_DIR}/scripts/functions.sh"

declare -A BENCHMARKS
BENCHMARKS[home_and_kitchen]="Home_and_Kitchen"
BENCHMARKS[clothing_shoes_and_jewelry]="Clothing_Shoes_and_Jewelry"
BENCHMARKS[pet_supplies]="Pet_Supplies"
BENCHMARKS[sports_and_outdoors]="Sports_and_Outdoors"

AMAZON_PRODUCT_DATA="${1:-}"
check_not_empty "${AMAZON_PRODUCT_DATA}" "path to Amazon product data"
check_directory "${AMAZON_PRODUCT_DATA}"

OUTPUT_DIR="${2:-}"
check_not_empty "${OUTPUT_DIR}" "output directory"
# check_directory_not_exists "${OUTPUT_DIR}"

DEVICE="${3:-cpu}"
check_valid_option "gpu" "cpu" "${DEVICE}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists."
fi

for BENCHMARK in ${!BENCHMARKS[@]}; do
    echo "Processing ${BENCHMARK}."

    BENCHMARK_DIR="$(package_root)/resources/product-search/${BENCHMARK}"
    check_directory "${BENCHMARK_DIR}"

    PRODUCT_LIST="${BENCHMARK_DIR}/product_list"
    check_file "${PRODUCT_LIST}"

    ASSOCS="${BENCHMARK_DIR}/assocs"
    check_file "${ASSOCS}"

    TOPICS="${BENCHMARK_DIR}/topics"
    check_file "${TOPICS}"

    META_GZIP="${AMAZON_PRODUCT_DATA}/meta_${BENCHMARKS[${BENCHMARK}]}.json.gz"
    check_file "${META_GZIP}"

    REVIEWS_GZIP="${AMAZON_PRODUCT_DATA}/reviews_${BENCHMARKS[${BENCHMARK}]}.json.gz"
    check_file "${REVIEWS_GZIP}"

    BENCHMARK_OUTPUT_DIR="${OUTPUT_DIR}/${BENCHMARK}"

    echo
    echo "Creating output directory."
    mkdir -p "${BENCHMARK_OUTPUT_DIR}"

    mkdir -p "${BENCHMARK_OUTPUT_DIR}/logs"

    if [[ ! -d "${BENCHMARK_OUTPUT_DIR}/trec" ]]; then
        echo
        echo "Verifying corpus."
        CORPUS_MD5_FILE="${BENCHMARK_OUTPUT_DIR}/md5"
        md5sum "${META_GZIP}" "${REVIEWS_GZIP}" | awk '{print $1}' > "${CORPUS_MD5_FILE}"

        set +e
        diff "${BENCHMARK_DIR}" "${CORPUS_MD5_FILE}" > "${BENCHMARK_OUTPUT_DIR}/diff"
        set -e

        if [[ -n "$(cat ${BENCHMARK_OUTPUT_DIR}/diff | tr '\n' ' ')" ]]; then
           echo "WARNING: the specified corpus does not match.
                 Results might differ from those published."
           echo

           echo "Diff:"
           cat "${BENCHMARK_OUTPUT_DIR}/diff"
        fi

        echo
        echo "Extracting product descriptions and reviews."

        mkdir -p "${BENCHMARK_OUTPUT_DIR}/trec"

        python $(package_root)/bin/amazon/amazon_products_to_trec.py \
            --loglevel debug \
            "${META_GZIP}" \
            --product_list "${PRODUCT_LIST}" \
            --trectext_out "${BENCHMARK_OUTPUT_DIR}/trec/${BENCHMARK}_meta" \
            &> ${BENCHMARK_OUTPUT_DIR}/logs/amazon_products_to_trec.log

        python $(package_root)/bin/amazon/amazon_reviews_to_trec.py \
            --loglevel error \
            "${REVIEWS_GZIP}" \
            --product_list "${PRODUCT_LIST}" \
            --trectext_out "${BENCHMARK_OUTPUT_DIR}/trec/${BENCHMARK}_reviews" \
            &> ${BENCHMARK_OUTPUT_DIR}/logs/amazon_reviews_to_trec.log
    fi

    export THEANO_FLAGS="mode=FAST_RUN,device=${DEVICE},floatX=float32,blas.ldflags=,nvcc.fastmath=True,warn_float64='warn',allow_gc=False,lib.cnmem=0.80"

    echo
    echo "Constructing LSE model on ${BENCHMARK} collection."

    if [[ ! -f "${BENCHMARK_OUTPUT_DIR}/meta" ]]; then
        # Package the corpus into machine-readable matrices.
        python bin/prepare.py \
            --loglevel info \
            --seed $(date +%s) \
            --assoc_path "${ASSOCS}" \
            --num_workers 2 \
            --overlapping \
            --resample \
            --window_size 4 \
            --no_instance_weights \
            --data_output "${BENCHMARK_OUTPUT_DIR}/data.npz" \
            --meta_output "${BENCHMARK_OUTPUT_DIR}/meta" \
            $(find ${BENCHMARK_OUTPUT_DIR}/trec -type f) \
            &> ${BENCHMARK_OUTPUT_DIR}/logs/prepare.log
    fi

    if [[ ! -d "${BENCHMARK_OUTPUT_DIR}/models" ]]; then
        mkdir -p "${BENCHMARK_OUTPUT_DIR}/models"

        # Train a model.
        python bin/train.py \
            --loglevel info \
            --data "${BENCHMARK_OUTPUT_DIR}/data.npz" \
            --meta "${BENCHMARK_OUTPUT_DIR}/meta" \
            --type vectorspace \
            --iterations 15 \
            --batch_size 4096 \
            --word_representation_size 300 \
            --entity_representation_size 128 \
            --one_hot_classes \
            --num_negative_samples 10 \
            --model_output "${BENCHMARK_OUTPUT_DIR}/models/model" \
            &> ${BENCHMARK_OUTPUT_DIR}/logs/train.log
    fi

    if [[ ! -d "${BENCHMARK_OUTPUT_DIR}/models/runs" ]]; then
        mkdir -p "${BENCHMARK_OUTPUT_DIR}/models/runs"
        for EPOCH in $(seq 0 15); do
            # Query the model.
            python bin/query.py \
                --loglevel info \
                --meta "${BENCHMARK_OUTPUT_DIR}/meta" \
                --model "${BENCHMARK_OUTPUT_DIR}/models/model_${EPOCH}.bin" \
                --topics "${TOPICS}" \
                --top 100 \
                --run_out "${BENCHMARK_OUTPUT_DIR}/models/runs/model_${EPOCH}.run" \
                &> ${BENCHMARK_OUTPUT_DIR}/logs/query.log

            for QREL in "qrel_validation" "qrel_test"; do
                trec_eval -m all_trec \
                    "${BENCHMARK_DIR}/${QREL}" \
                    "${BENCHMARK_OUTPUT_DIR}/models/runs/model_${EPOCH}.run_ef" \
                    > "${BENCHMARK_OUTPUT_DIR}/models/runs/model_${EPOCH}_${QREL}.eval"
            done
        done
    fi

    BEST_VALIDATION_QREL=$(
        awk '/^ndcg_cut_100 .*all/{print $3 " " FILENAME}' ${BENCHMARK_OUTPUT_DIR}/models/runs/*_qrel_validation.eval \
        | sort -g \
        | tail -n 1 \
        | cut -d' ' -f2)
    BEST_TEST_QREL=$(
        echo ${BEST_VALIDATION_QREL} | \
        sed -E 's/qrel_validation/qrel_test/g')

    echo -n "NDCG@100 (validation): " > "${BENCHMARK_OUTPUT_DIR}/results"
    awk '/^ndcg_cut_100 .*all/{print $3}' ${BEST_VALIDATION_QREL} >> "${BENCHMARK_OUTPUT_DIR}/results"
    echo -n "NDCG@100 (test): " >> "${BENCHMARK_OUTPUT_DIR}/results"
    awk '/^ndcg_cut_100 .*all/{print $3}' ${BEST_TEST_QREL} >> "${BENCHMARK_OUTPUT_DIR}/results"

    echo
    cat "${BENCHMARK_OUTPUT_DIR}/results"
done

echo
echo "All done!"
