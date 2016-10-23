#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source "${SCRIPT_DIR}/scripts/functions.sh"

convert_topics_to_csv() {
    tr '\n' ' ' | \
    sed  -e 's/<\/top>/\n/g' | \
    sed -e 's/.*<num>.*EX\([0-9]\+\).*\s*<title>\s*\(.*\)\s*<\/title>.*/\1;\2/g'
}

declare -A TREC_ENTERPRISE_TOPICS
TREC_ENTERPRISE_TOPICS[2005]="http://trec.nist.gov/data/enterprise/05/ent05.expert.topics"
TREC_ENTERPRISE_TOPICS[2006]="http://trec.nist.gov/data/enterprise/06/ent06.expert.topics"

declare -A TREC_ENTERPRISE_QRELS
TREC_ENTERPRISE_QRELS[2005]="http://trec.nist.gov/data/enterprise/05/ent05.expert.qrels"
TREC_ENTERPRISE_QRELS[2006]="http://trec.nist.gov/data/enterprise/06/ent06.qrels.expert"

W3C_CORPUS="${1:-}"
check_not_empty "${W3C_CORPUS}" "path to W3C corpus"
check_directory "${W3C_CORPUS}"

OUTPUT_DIR="${2:-}"
check_not_empty "${OUTPUT_DIR}" "output directory"
check_directory_not_exists "${OUTPUT_DIR}"

DEVICE="${3:-cpu}"
check_valid_option "gpu" "cpu" "${DEVICE}"

W3C_ASSOCS="${SCRIPT_DIR}/resources/expert-finding/w3c-all.assoc"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists."

    exit -1
fi

echo
echo "Creating output directory."
mkdir -p "${OUTPUT_DIR}"

echo
echo "Verifying W3C corpus."
CORPUS_MD5_FILE="${OUTPUT_DIR}/W3C-local.md5"
directory_md5sum ${W3C_CORPUS} > "${CORPUS_MD5_FILE}"

diff "${SCRIPT_DIR}/resources/expert-finding/W3C.md5" "${CORPUS_MD5_FILE}" > "${OUTPUT_DIR}/diff"

if [[ -n "$(cat ${OUTPUT_DIR}/diff)" ]]; then
    echo "WARNING: the specified W3C corpus does not match.
          Results might differ from those published."
    echo

    echo "Diff:"
    cat "${OUTPUT_DIR}/diff"
fi

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

export THEANO_FLAGS="mode=FAST_RUN,device=${DEVICE},floatX=float32,blas.ldflags=,nvcc.fastmath=True,warn_float64='warn',allow_gc=False,lib.cnmem=0.80"

echo
echo "Constructing log-linear model on W3C collection."

# Package the corpus into machine-readable matrices.
python bin/prepare.py \
    --loglevel info \
    --seed $(date +%s) \
    --assoc_path "${W3C_ASSOCS}" \
    --num_workers 16 \
    --data_output "${OUTPUT_DIR}/data.npz" \
    --meta_output "${OUTPUT_DIR}/meta" \
    $(find ${W3C_CORPUS} -type f) \
    >& "${OUTPUT_DIR}/prepare.log"

# Train a model.
python bin/train.py \
    --loglevel info \
    --data "${OUTPUT_DIR}/data.npz" \
    --meta "${OUTPUT_DIR}/meta" \
    --type loglinear \
    --batch_size 1024 \
    --word_representation_size 300 \
    --model_output "${OUTPUT_DIR}/model" \
    >& "${OUTPUT_DIR}/train.log"

# Query the model.
python bin/query.py \
    --loglevel info \
    --meta "${OUTPUT_DIR}/meta" \
    --model "${OUTPUT_DIR}/model_1.bin" \
    --topics "${OUTPUT_DIR}/2005.topics.csv" "${OUTPUT_DIR}/2006.topics.csv" \
    --run_out "${OUTPUT_DIR}/run" \
    >& "${OUTPUT_DIR}/query.log"

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
