#!/bin/bash

# ==============================================================================
# 1. Environment & Paths Configuration
# ==============================================================================

PROJ_ROOT=$(pwd)

WEIGHTS_DIR="${PROJ_ROOT}/weights" 
OUTPUT_ROOT="${PROJ_ROOT}/output"


PYTHON_BIN="python3"

export HF_HOME="${PROJ_ROOT}/huggingface"
# export HF_TOKEN="your_token_here" 

# ==============================================================================
# 2. Model & Inference Hyperparameters
# ==============================================================================

INFINITY_MODEL_PATH="${WEIGHTS_DIR}/infinity_2b_reg.pth"
VAE_PATH="${WEIGHTS_DIR}/infinity_vae_d32reg.pth"
TEXT_ENCODER_CKPT="${WEIGHTS_DIR}/flan-t5-xl"
MASK2FORMER_PATH="${WEIGHTS_DIR}" 

PN="1M"
MODEL_TYPE="infinity_2b"
VAE_TYPE=32
TEXT_CHANNELS=2048
CHECKPOINT_TYPE='torch'

CFG=4
TAU=1
CFG_INSERTION_LAYER=0
REWRITE_PROMPT=0

USE_SCALE_SCHEDULE_EMBEDDING=0
USE_BIT_LABEL=1
ROPE2D_NORMALIZED_BY_HW=2
ADD_LVL_EMBEDING_ONLY_FIRST_BLOCK=1
ROPE2D_EACH_SA_LAYER=1
APPLY_SPATIAL_PATCHIFY=0

# === VAR-Scaling ===
ENABLE_SCALING=1
SCALING_LAYER_IDX=3
SCALING_NUM_SAMPLES=3000

SUB_FIX="cfg${CFG}_tau${TAU}_scaling${ENABLE_SCALING}"

# ==============================================================================
# 3. Evaluation Function
# ==============================================================================

test_gen_eval() {
    echo "========================================================"
    echo "Running GenEval..."
    echo "Model: ${MODEL_TYPE}"
    echo "Scaling Enabled: ${ENABLE_SCALING}"
    echo "========================================================"

    local OUT_DIR="${OUTPUT_ROOT}/gen_eval_${SUB_FIX}_rewrite${REWRITE_PROMPT}"
    mkdir -p "${OUT_DIR}/images"
    mkdir -p "${OUT_DIR}/results"

    ${PYTHON_BIN} evaluation/gen_eval/infer4eval.py \
    --cfg ${CFG} \
    --tau ${TAU} \
    --pn ${PN} \
    --model_path "${INFINITY_MODEL_PATH}" \
    --vae_type ${VAE_TYPE} \
    --vae_path "${VAE_PATH}" \
    --add_lvl_embeding_only_first_block ${ADD_LVL_EMBEDING_ONLY_FIRST_BLOCK} \
    --use_bit_label ${USE_BIT_LABEL} \
    --model_type ${MODEL_TYPE} \
    --rope2d_each_sa_layer ${ROPE2D_EACH_SA_LAYER} \
    --rope2d_normalized_by_hw ${ROPE2D_NORMALIZED_BY_HW} \
    --use_scale_schedule_embedding ${USE_SCALE_SCHEDULE_EMBEDDING} \
    --checkpoint_type ${CHECKPOINT_TYPE} \
    --text_encoder_ckpt "${TEXT_ENCODER_CKPT}" \
    --text_channels ${TEXT_CHANNELS} \
    --apply_spatial_patchify ${APPLY_SPATIAL_PATCHIFY} \
    --cfg_insertion_layer ${CFG_INSERTION_LAYER} \
    --outdir "${OUT_DIR}/images" \
    --rewrite_prompt ${REWRITE_PROMPT} \
    --enable_scaling ${ENABLE_SCALING} \
    --scaling_layer_idx ${SCALING_LAYER_IDX} \
    --scaling_num_samples ${SCALING_NUM_SAMPLES}

    local M2F_CONFIG="evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    
    ${PYTHON_BIN} evaluation/gen_eval/evaluate_images.py "${OUT_DIR}/images" \
    --outfile "${OUT_DIR}/results/det.jsonl" \
    --model-config "${M2F_CONFIG}" \
    --model-path "${MASK2FORMER_PATH}"

    ${PYTHON_BIN} evaluation/gen_eval/summary_scores.py "${OUT_DIR}/results/det.jsonl" > "${OUT_DIR}/results/res.txt"
    
    echo "--------------------------------------------------------"
    echo "Evaluation Results:"
    cat "${OUT_DIR}/results/res.txt"
    echo "--------------------------------------------------------"
}

# ==============================================================================
# 4. Execution
# ==============================================================================

test_gen_eval