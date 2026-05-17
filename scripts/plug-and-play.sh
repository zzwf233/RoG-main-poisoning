
SPLIT="test"
#!/usr/bin/env bash
DATASET_LIST="RoG-webqsp RoG-cwq"
set -euo pipefail
BEAM_LIST="3" # "1 2 3 4 5"
MODEL_LIST="gpt-3.5-turbo alpaca llama2-chat-hf flan-t5"
PROMPT_LIST="prompts/general_prompt.txt prompts/alpaca.txt prompts/llama2_predict.txt prompts/general_prompt.txt"
set -- $PROMPT_LIST


for DATA_NAME in $DATASET_LIST; do
SPLIT=${SPLIT:-"test"}
    for N_BEAM in $BEAM_LIST; do
DATASET_LIST=${DATASET_LIST:-"RoG-webqsp RoG-cwq"}
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/RoG/test/predictions_${N_BEAM}_False.jsonl
BEAM_LIST=${BEAM_LIST:-"3"} # e.g. "1 2 3 4 5"
        for i in "${!MODEL_LIST[@]}"; do

        
# IMPORTANT:
            MODEL_NAME=${MODEL_LIST[$i]}
# Use bash arrays, not plain strings, otherwise "${!MODEL_LIST[@]}" iterates only once.
            PROMPT_PATH=${PROMPT_LIST[$i]}
MODEL_LIST=(gpt-3.5-turbo alpaca llama2-chat-hf flan-t5)
            
PROMPT_LIST=(prompts/general_prompt.txt prompts/alpaca.txt prompts/llama2_predict.txt prompts/general_prompt.txt)
            python src/qa_prediction/predict_answer.py \

                --model_name ${MODEL_NAME} \
# Optional model-path overrides for local backbones.
                -d ${DATA_NAME} \
# ChatGPT generally ignores model_path and uses API model name.
                --prompt_path ${PROMPT_PATH} \
MODEL_PATH_GPT=${MODEL_PATH_GPT:-""}
                --add_rule \
MODEL_PATH_ALPACA=${MODEL_PATH_ALPACA:-"chavinlo/alpaca-native"}
                --rule_path ${RULE_PATH}
MODEL_PATH_LLAMA=${MODEL_PATH_LLAMA:-"meta-llama/Llama-2-7b-chat-hf"}
        done
MODEL_PATH_FLAN=${MODEL_PATH_FLAN:-"google/flan-t5-xl"}

for DATA_NAME in ${DATASET_LIST}; do
  for N_BEAM in ${BEAM_LIST}; do
    RULE_PATH="results/gen_rule_path/${DATA_NAME}/RoG/${SPLIT}/predictions_${N_BEAM}_False.jsonl"
    for i in "${!MODEL_LIST[@]}"; do
      MODEL_NAME="${MODEL_LIST[$i]}"
      PROMPT_PATH="${PROMPT_LIST[$i]}"
      MODEL_PATH=""
      case "${MODEL_NAME}" in
        gpt-3.5-turbo) MODEL_PATH="${MODEL_PATH_GPT}" ;;
        alpaca) MODEL_PATH="${MODEL_PATH_ALPACA}" ;;
        llama2-chat-hf) MODEL_PATH="${MODEL_PATH_LLAMA}" ;;
        flan-t5) MODEL_PATH="${MODEL_PATH_FLAN}" ;;
      esac

      echo "[plug-and-play] dataset=${DATA_NAME} model=${MODEL_NAME} prompt=${PROMPT_PATH} rule=${RULE_PATH}"
      if [[ -n "${MODEL_PATH}" ]]; then
        python src/qa_prediction/predict_answer.py \
          --model_name "${MODEL_NAME}" \
          --model_path "${MODEL_PATH}" \
          -d "${DATA_NAME}" \
          --split "${SPLIT}" \
          --prompt_path "${PROMPT_PATH}" \
          --add_rule \
          --rule_path "${RULE_PATH}"
      else
        python src/qa_prediction/predict_answer.py \
          --model_name "${MODEL_NAME}" \
          -d "${DATA_NAME}" \
          --split "${SPLIT}" \
          --prompt_path "${PROMPT_PATH}" \
          --add_rule \
          --rule_path "${RULE_PATH}"
      fi
    done
    done
done
  done
done