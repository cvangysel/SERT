#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

convert_topics_to_csv() {
    tr '\n' ' ' | \
    sed  -e 's/<\/top>/\n/g' | \
    sed -e 's/.*<num>.*EX\([0-9]\+\).*\s*<title>\s*\(.*\)\s*<\/title>.*/\1;\2/g'
}

check_installed() {
    command -v $1 >/dev/null 2>&1 || { echo >&2 "Required tool '$1' is not installed. Aborting."; exit 1; }
}

declare -A TREC_ENTERPRISE_TOPICS
TREC_ENTERPRISE_TOPICS[2005]="http://trec.nist.gov/data/enterprise/05/ent05.expert.topics"
TREC_ENTERPRISE_TOPICS[2006]="http://trec.nist.gov/data/enterprise/06/ent06.expert.topics"

declare -A TREC_ENTERPRISE_QRELS
TREC_ENTERPRISE_QRELS[2005]="http://trec.nist.gov/data/enterprise/05/ent05.expert.qrels"
TREC_ENTERPRISE_QRELS[2006]="http://trec.nist.gov/data/enterprise/06/ent06.qrels.expert"

W3C_CORPUS="${1:-}"
OUTPUT_DIR="${2:-}"

W3C_ASSOCS="${SCRIPT_DIR}/aux/w3c-all.assoc"

check_installed "tr"
check_installed "sed"
check_installed "find"
check_installed "md5sum"
check_installed "sort"
check_installed "uniq"
check_installed "curl"
check_installed "python"
check_installed "trec_eval"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists."

    exit -1
fi

echo
echo "Verifying W3C corpus."
CORPUS_HASH=$(find ${W3C_CORPUS} -type f -exec md5sum {} \; | sort -k 34 | md5sum)

if [[ "${CORPUS_HASH}" != fd5f267c8283682d77bf489056206251* ]]; then
    echo "WARNING: the specified W3C corpus does not match.
          Results might differ from those published."
fi

echo
echo "Creating output directory."
mkdir -p "${OUTPUT_DIR}"

echo
echo "Fetching topics and relevance judgments."
for TREC_EDITION in "2005" "2006"; do
    curl -s "${TREC_ENTERPRISE_TOPICS[${TREC_EDITION}]}" | \
    convert_topics_to_csv \
    > "${OUTPUT_DIR}/${TREC_EDITION}.topics.csv"

    curl -s "${TREC_ENTERPRISE_QRELS[${TREC_EDITION}]}" | \
    sort | uniq \
    > "${OUTPUT_DIR}/${TREC_EDITION}.qrel"
done

export THEANO_FLAGS="mode=FAST_RUN,device=cpu,floatX=float32,blas.ldflags=,nvcc.fastmath=True,warn_float64='warn',allow_gc=False,lib.cnmem=0.80"

echo
echo "Constructing log-linear model on W3C collection."

# Package the corpus into machine-readable matrices.
python bin/prepare.py \
    --loglevel error \
    --assoc_path "${W3C_ASSOCS}" \
    --num_workers 8 \
    --data_output "${OUTPUT_DIR}/data.npz" \
    --meta_output "${OUTPUT_DIR}/meta" \
    $(find ${W3C_CORPUS} -type f)

# Train a model.
python bin/train.py \
    --loglevel error \
    --data "${OUTPUT_DIR}/data.npz" \
    --meta "${OUTPUT_DIR}/meta" \
    --model_output "${OUTPUT_DIR}/model"

# Query the model.
python bin/query.py \
    --loglevel error \
    --meta "${OUTPUT_DIR}/meta" \
    --model "${OUTPUT_DIR}/model_1.bin" \
    --topics "${OUTPUT_DIR}/2005.topics.csv" "${OUTPUT_DIR}/2006.topics.csv" \
    --run_out "${OUTPUT_DIR}/run"

echo
echo "Evaluating on TREC Enterprise tracks."
for TREC_EDITION in "2005" "2006"; do
    trec_eval -q -m all_trec \
    "${OUTPUT_DIR}/${TREC_EDITION}.qrel" \
    "${OUTPUT_DIR}/run_ef" \
    > "${OUTPUT_DIR}/run_ef.${TREC_EDITION}.eval"

    echo -n "${TREC_EDITION} Enterprise Track: "
    for MEASURE in "ndcg" "map" "recip_rank" "P_5"; do
        VALUE=$(cat "${OUTPUT_DIR}/run_ef.${TREC_EDITION}.eval" |
                awk "/^${MEASURE} .*all/{print \$3}")

        echo -n "${MEASURE}=${VALUE}; "
    done

    echo
done
