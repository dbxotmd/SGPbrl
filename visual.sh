# #!/usr/bin/env bash
# set -e

# # GPU 설정
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 실험 설정
# seeds=(0 42 1234 5678 9876)
# latents=(16 32)
# hiddens=(32 64 128)
# topkps=(10)
# methods=(negative_distance exponential_decay gaussian_kernel cosine_similarity normalized_distance potential_based)

# # 시작 메시지
# echo "=== Visualization Stage: Parallel train_offline_visualization runs (up to ${num_gpus} at once) ==="

# pids=()
# job_idx=0

# for env in hopper walker; do
#   if [ "$env" = "hopper" ]; then
#     env_name="hopper-medium-expert-v2"
#   else
#     env_name="walker2d-medium-expert-v2"
#   fi

#   for seed in "${seeds[@]}"; do
#     for ld in "${latents[@]}"; do
#       for hd in "${hiddens[@]}"; do
#         for tk in "${topkps[@]}"; do
#           for method in "${methods[@]}"; do

#             ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}"

#             # 체크포인트 없으면 건너뜀
#             if [ ! -d "$ckpt_dir" ]; then
#               echo "⚡ SKIP Visualization (missing ckpt): ${ckpt_dir}"
#               continue
#             fi

#             # round-robin GPU 할당
#             gpu="${gpus[$((job_idx % num_gpus))]}"
#             echo "▶ Visualize: env=${env}, seed=${seed}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method} on GPU${gpu}"

#             CUDA_VISIBLE_DEVICES=${gpu} \
#               python train_offline_visualization_d4rl.py \
#                 --comment "${env}_${ld}_${hd}_${tk}_${method}" \
#                 --eval_interval 5000 \
#                 --env_name "${env_name}" \
#                 --config configs/mujoco_config.py \
#                 --eval_episodes 10 \
#                 --use_reward_model True \
#                 --model_type PrefTransformer \
#                 --ckpt_dir "${ckpt_dir}" \
#                 --seed ${seed} \
#                 --latent_dim ${ld} \
#                 --hidden_dim ${hd} \
#                 --topkp ${tk} \
#                 --method ${method} &

#             pids+=($!)
#             job_idx=$((job_idx+1))

#             # 동시 실행 개수 제한
#             if (( ${#pids[@]} >= num_gpus )); then
#               wait -n
#             fi

#           done
#         done
#       done
#     done
#   done
# done

# # 남은 작업 대기
# wait
# echo "✅ Visualization complete: all jobs done."


# #!/usr/bin/env bash
# set -e

# # GPU 설정
# gpus=(0 1 2 3)
# num_gpus=${#gpus[@]}

# # 실험 설정
# seeds=(0 42 1234 5678 9876)
# latents=(16 32)
# hiddens=(32 64 128)
# topkps=(10)
# methods=(negative_distance exponential_decay gaussian_kernel cosine_similarity normalized_distance potential_based)
# weights=(0.1 0.5 1.0 2.0 5.0)

# echo "=== Visualization Stage: Serial train_offline_visualization runs (one at a time) ==="

# job_idx=0

# for env in hopper walker; do
#   if [ "$env" = "hopper" ]; then
#     env_name="hopper-medium-expert-v2"
#   else
#     env_name="walker2d-medium-expert-v2"
#   fi

#   for seed in "${seeds[@]}"; do
#     for ld in "${latents[@]}"; do
#       for hd in "${hiddens[@]}"; do
#         for tk in "${topkps[@]}"; do
#           for method in "${methods[@]}"; do
#             for weight in "${weights[@]}"; do

#               ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}"

#               if [ ! -d "$ckpt_dir" ]; then
#                 echo "⚡ SKIP Visualization (missing ckpt): ${ckpt_dir}"
#                 continue
#               fi

#               gpu="${gpus[$((job_idx % num_gpus))]}"
#               echo "▶ [${job_idx}] Visualize: env=${env}, seed=${seed}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu}"

#               MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=${gpu} \
#                 python train_offline_visualization_d4rl.py \
#                   --comment "${env}_${ld}_${hd}_${tk}_${method}_${weight}" \
#                   --eval_interval 5000 \
#                   --env_name "${env_name}" \
#                   --config configs/mujoco_config.py \
#                   --eval_episodes 10 \
#                   --use_reward_model True \
#                   --model_type PrefTransformer \
#                   --ckpt_dir "${ckpt_dir}" \
#                   --seed ${seed} \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --topkp ${tk} \
#                   --method ${method} \
#                   --shaping_weight ${weight}

#               job_idx=$((job_idx + 1))

#             done
#           done
#         done
#       done
#     done
#   done
# done

# echo "✅ All serial visualization jobs complete."

# #!/usr/bin/env bash
# set -e

# # GPU 설정
# gpus=(0 1 2 )
# num_gpus=${#gpus[@]}

# # 공통 실험 설정
# seeds=(0 42 1234)
# topkps=(10)
# methods=(negative_distance exponential_decay gaussian_kernel cosine_similarity normalized_distance potential_based)
# weights=(0.1 0.5 1.0 2.0 5.0)

# # 실행할 작업을 추적하기 위한 인덱스
# job_idx=0

# # 시각화 실행 로직을 함수로 정의
# run_visualizations() {
#   # 함수로 배열을 전달받기 위해 nameref(-n) 사용
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2

#   echo "=== Visualization Stage Start: latents=(${latents_ref[@]}), hiddens=(${hiddens_ref[@]}) ==="

#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#     else
#       env_name="walker2d-medium-expert-v2"
#     fi

#     for seed in "${seeds[@]}"; do
#       for ld in "${latents_ref[@]}"; do
#         for hd in "${hiddens_ref[@]}"; do
#           for tk in "${topkps[@]}"; do
#             for method in "${methods[@]}"; do
#               for weight in "${weights[@]}"; do

#                 ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}"

#                 if [ ! -d "$ckpt_dir" ]; then
#                   echo "⚡ SKIP Visualization (missing ckpt): ${ckpt_dir}"
#                   continue
#                 fi

#                 gpu="${gpus[$((job_idx % num_gpus))]}"
#                 echo "▶ [${job_idx}] Visualize: env=${env}, seed=${seed}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu}"

#                 # xvfb-run을 사용하여 가상 스크린에서 파이썬 스크립트 실행
#                 MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=${gpu} \
#                   xvfb-run -s "-screen 0 1920x1080x24" \
#                   python train_offline_visualization_d4rl.py \
#                     --comment "${env}_${ld}_${hd}_${tk}_${method}_${weight}" \
#                     --eval_interval 5000 \
#                     --env_name "${env_name}" \
#                     --config configs/mujoco_config.py \
#                     --eval_episodes 10 \
#                     --use_reward_model True \
#                     --model_type PrefTransformer \
#                     --ckpt_dir "${ckpt_dir}" \
#                     --seed ${seed} \
#                     --latent_dim ${ld} \
#                     --hidden_dim ${hd} \
#                     --topkp ${tk} \
#                     --method ${method} \
#                     --shaping_weight ${weight}

#                 job_idx=$((job_idx + 1))

#               done
#             done
#           done
#         done
#       done
#     done
#   done
# }

# # --- 1차 실험 실행 ---
# # latents=(16 32), hiddens=(32 64 128) 설정으로 실행
# latents_batch1=(16 32)
# hiddens_batch1=(32 64 128)
# run_visualizations latents_batch1 hiddens_batch1

# # --- 2차 실험 실행 ---
# # latents=(12), hiddens=(750) 설정으로 이어서 실행
# latents_batch2=(12)
# hiddens_batch2=(750)
# run_visualizations latents_batch2 hiddens_batch2

# echo "✅ All serial visualization jobs complete."

# #!/usr/bin/env bash
# set -e

# # GPU 설정
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 공통 실험 설정
# seeds=(0 42 1234)
# topkps=(10)
# methods=(negative_distance exponential_decay gaussian_kernel cosine_similarity normalized_distance potential_based)
# weights=(0.1 0.5 1.0 2.0 5.0)
# state_actions=(True False)

# # 실행할 작업을 추적하기 위한 인덱스
# job_idx=0

# # 시각화 실행 로직을 함수로 정의
# run_visualizations() {
#   # 함수로 배열을 전달받기 위해 nameref(-n) 사용
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2

#   echo "=== Visualization Stage Start: latents=(${latents_ref[@]}), hiddens=(${hiddens_ref[@]}) ==="

#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#     else
#       # 기존 스크립트와 동일하게 expert 사용 (필요시 replay로 변경)
#       env_name="walker2d-medium-expert-v2"
#     fi

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do
#               for method in "${methods[@]}"; do
#                 for weight in "${weights[@]}"; do

#                   # state_action을 경로에 포함
#                   ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}_${sa}"

#                   if [ ! -d "$ckpt_dir" ]; then
#                     echo "⚡ SKIP Visualization (missing ckpt): ${ckpt_dir}"
#                     continue
#                   fi

#                   gpu="${gpus[$((job_idx % num_gpus))]}"
#                   echo "▶ [${job_idx}] Visualize: env=${env}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu}"

#                   # xvfb-run을 사용하여 가상 스크린에서 파이썬 스크립트 실행
#                   MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=${gpu} \
#                     xvfb-run -s "-screen 0 1920x1080x24" \
#                     python train_offline_visualization_d4rl.py \
#                       --comment "${env}_${ld}_${hd}_${tk}_${sa}_${method}_${weight}" \
#                       --eval_interval 5000 \
#                       --env_name "${env_name}" \
#                       --config configs/mujoco_config.py \
#                       --eval_episodes 10 \
#                       --use_reward_model True \
#                       --model_type PrefTransformer \
#                       --ckpt_dir "${ckpt_dir}" \
#                       --seed ${seed} \
#                       --latent_dim ${ld} \
#                       --hidden_dim ${hd} \
#                       --topkp ${tk} \
#                       --state_action ${sa} \
#                       --method ${method} \
#                       --shaping_weight ${weight}

#                   job_idx=$((job_idx + 1))

#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# }

# # --- 1차 실험 실행 ---
# # latents=(16 32), hiddens=(32 64 128) 설정으로 실행
# latents_batch1=(16 32)
# hiddens_batch1=(32 64 128)
# run_visualizations latents_batch1 hiddens_batch1

# # --- 2차 실험 실행 ---
# # latents=(12), hiddens=(750) 설정으로 이어서 실행
# latents_batch2=(12)
# hiddens_batch2=(750)
# run_visualizations latents_batch2 hiddens_batch2

# echo "✅ All serial visualization jobs complete."

# #!/usr/bin/env bash
# set -e

# # GPU 설정
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 공통 실험 설정
# seeds=(0 42 1234)
# topkps=(10)
# methods=(negative_distance exponential_decay gaussian_kernel cosine_similarity normalized_distance potential_based)
# weights=(0.1 0.5 1.0 2.0 5.0)
# state_actions=(True False)

# job_idx=0

# run_visualizations() {
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2

#   echo "=== Visualization Stage Start: latents=(${latents_ref[@]}), hiddens=(${hiddens_ref[@]}) ==="

#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#     else
#       env_name="walker2d-medium-expert-v2"
#     fi

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do
#               for method in "${methods[@]}"; do
#                 for weight in "${weights[@]}"; do

#                   ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}_${sa}"

#                   if [ ! -d "$ckpt_dir" ]; then
#                     echo "⚡ SKIP Visualization (missing ckpt): ${ckpt_dir}"
#                     continue
#                   fi

#                   gpu="${gpus[$((job_idx % num_gpus))]}"
#                   echo "▶ [${job_idx}] Visualize: env=${env}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu}"

#                   MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=${gpu} \
#                     xvfb-run -a --server-args="-screen 0 1920x1080x24 -nolisten tcp -noreset" \
#                     python train_offline_visualization_d4rl.py \
#                       --comment "${env}_${ld}_${hd}_${tk}_${sa}_${method}_${weight}" \
#                       --eval_interval 5000 \
#                       --env_name "${env_name}" \
#                       --config configs/mujoco_config.py \
#                       --eval_episodes 10 \
#                       --use_reward_model True \
#                       --model_type PrefTransformer \
#                       --ckpt_dir "${ckpt_dir}" \
#                       --seed ${seed} \
#                       --latent_dim ${ld} \
#                       --hidden_dim ${hd} \
#                       --topkp ${tk} \
#                       --state_action ${sa} \
#                       --method ${method} \
#                       --shaping_weight ${weight} &  # <-- 백그라운드 실행

#                   job_idx=$((job_idx + 1))

#                   # 3개씩 실행 후 동기화
#                   if (( job_idx % num_gpus == 0 )); then
#                     wait
#                   fi

#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# }

# # 1차 실행
# latents_batch1=(16 32)
# hiddens_batch1=(32 64 128)
# run_visualizations latents_batch1 hiddens_batch1

# # 2차 실행
# latents_batch2=(12)
# hiddens_batch2=(750)
# run_visualizations latents_batch2 hiddens_batch2

# # 마지막 남은 작업 대기
# wait

# echo "✅ All parallel visualization jobs complete."
# #!/usr/bin/env bash
# set -e

# # =========================
# # 기본 설정
# # =========================
# # 사용할 GPU 리스트
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 공통 실험 설정
# seeds=(0)
# topkps=(10)
# methods=(negative_distance gaussian_kernel cosine_similarity normalized_distance potential_based)
# weights=(0.1 0.5 1.0 2.0)
# state_actions=(False True)

# # EGL로 렌더링 (X 서버 불필요)
# export MUJOCO_GL=egl

# # 병렬 실행 관리용 인덱스
# job_idx=0

# # =========================
# # 함수: 시각화 실행
# # =========================
# run_visualizations() {
#   # 배열 참조 (nameref)
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2

#   echo "=== Visualization Stage Start: latents=(${latents_ref[@]}), hiddens=(${hiddens_ref[@]}) ==="

#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#     else
#       env_name="walker2d-medium-replay-v2"
#     fi

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do
#               for method in "${methods[@]}"; do
#                 for weight in "${weights[@]}"; do

#                   ckpt_dir="./logs/tb/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}_${sa}"

#                   if [ ! -d "$ckpt_dir" ]; then
#                     echo "⚡ SKIP Visualization (missing ckpt): ${ckpt_dir}"
#                     continue
#                   fi

#                   gpu="${gpus[$((job_idx % num_gpus))]}"

#                   echo "▶ [${job_idx}] Visualize: env=${env}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu}"

#                   # 핵심 변경: xvfb-run 제거, EGL 사용
#                   CUDA_VISIBLE_DEVICES=${gpu} \
#                   python train_offline_visualization_d4rl.py \
#                     --comment "${env}_${ld}_${hd}_${tk}_${sa}_${method}_${weight}" \
#                     --eval_interval 5000 \
#                     --env_name "${env_name}" \
#                     --config configs/mujoco_config.py \
#                     --eval_episodes 10 \
#                     --use_reward_model True \
#                     --model_type PrefTransformer \
#                     --ckpt_dir "${ckpt_dir}" \
#                     --seed ${seed} \
#                     --latent_dim ${ld} \
#                     --hidden_dim ${hd} \
#                     --topkp ${tk} \
#                     --state_action ${sa} \
#                     --method ${method} \
#                     --shaping_weight ${weight} &

#                   job_idx=$((job_idx + 1))

#                   # GPU 수(num_gpus) 만큼 동시에 돌리고 동기화
#                   if (( job_idx % num_gpus == 0 )); then
#                     wait
#                   fi

#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# }

# # =========================
# # 배치 1
# # =========================
# latents_batch2=(12)
# hiddens_batch2=(750)
# run_visualizations latents_batch2 hiddens_batch2

# # =========================
# # 배치 2
# # =========================
# latents_batch1=(16 32)
# hiddens_batch1=(32 64 128)
# run_visualizations latents_batch1 hiddens_batch1
# # 남은 작업 대기
# wait

# echo "✅ All parallel visualization jobs complete."

#!/usr/bin/env bash
set -e

# =========================
# 기본 설정
# =========================
# 사용할 GPU 리스트
gpus=(0 1 2)
num_gpus=${#gpus[@]}

# 공통 실험 설정
seeds=(0 42 1234)
topkps=(10)
methods=(cosine_similarity)
weights=(1.0)
state_actions=(False)

# EGL로 렌더링 (X 서버 불필요)
export MUJOCO_GL=egl

# 결과 루트 및 고정된 러닝 설정(로그 경로 구성용)
OUTPUT_ROOT="/home/hgmin/SGPbrl/logs/tb"

# 병렬 실행 관리용 인덱스
job_idx=0

# =========================
# 함수: 시각화 실행
# =========================
run_visualizations() {
  # 배열 참조 (nameref)
  local -n latents_ref=$1
  local -n hiddens_ref=$2

  echo "=== Visualization Stage Start: latents=(${latents_ref[@]}), hiddens=(${hiddens_ref[@]}) ==="

  for env in hopper walker; do
    if [ "$env" = "hopper" ]; then
      env_name="hopper-medium-expert-v2"
    else
      env_name="walker2d-medium-replay-v2"
    fi

    for sa in "${state_actions[@]}"; do
      for seed in "${seeds[@]}"; do
        for ld in "${latents_ref[@]}"; do
          for hd in "${hiddens_ref[@]}"; do
            for tk in "${topkps[@]}"; do
              for method in "${methods[@]}"; do
                for weight in "${weights[@]}"; do

                  # (기존) 체크포인트 경로: 파이썬 스크립트 인자로 계속 사용
                  ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}_${sa}"

                  exp_name="${env}_${ld}_${hd}_${tk}_${sa}_${method}_${weight}"
                  result_base_dir="${OUTPUT_ROOT}/${env_name}/reward_True_PrefTransformer/${exp_name}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}/visual"
                  echo "${result_base_dir}"


                  # 이미 결과가 존재하면 스킵 (타임스탬프 하위가 하나라도 있으면 스킵)
                  if compgen -G "${result_base_dir}/*" > /dev/null; then
                    echo "⏭️  SKIP Visualization (already done): ${result_base_dir}"
                    continue
                  fi

                  gpu="${gpus[$((job_idx % num_gpus))]}"

                  echo "▶ [${job_idx}] Visualize: env=${env}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu}"
                  CUDA_VISIBLE_DEVICES=${gpu} \
                  python train_offline_visualization_d4rl_extra.py \
                    --comment "${env}_${ld}_${hd}_${tk}_${sa}_${method}_${weight}" \
                    --eval_interval 5000 \
                    --env_name "${env_name}" \
                    --config configs/mujoco_config.py \
                    --eval_episodes 10 \
                    --use_reward_model True \
                    --model_type PrefTransformer \
                    --ckpt_dir "${ckpt_dir}" \
                    --seed ${seed} \
                    --latent_dim ${ld} \
                    --hidden_dim ${hd} \
                    --topkp ${tk} \
                    --state_action=${sa} \
                    --method ${method} \
                    --shaping_weight ${weight} &

                  job_idx=$((job_idx + 1))

                  # GPU 수(num_gpus) 만큼 동시에 돌리고 동기화
                  if (( job_idx % num_gpus == 0 )); then
                    wait
                  fi

                done
              done
            done
          done
        done
      done
    done
  done
}


# =========================
# 배치 2
# =========================
latents_batch1=(16)
hiddens_batch1=(32)
run_visualizations latents_batch1 hiddens_batch1

# 남은 작업 대기
wait
echo "✅ All parallel visualization jobs complete."
