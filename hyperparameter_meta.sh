# #!/usr/bin/env bash
# set -e

# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # latents=(16 32)
# # hiddens=(32 64 128)
# seeds=(0 42 1234 5678 9876)
# latents=(12)
# hiddens=(750)
# topkps=(10)


# ####################################
# # 1) CVAE 보상 모델 학습 (병렬 실행)
# ####################################
# echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="
# pids=()
# job_idx=0

# for env in hopper walker; do
#   # 맵핑
#   if [ "$env" = "hopper" ]; then
#     env_name="hopper-medium-expert-v2"; num_query=100
#   else
#     env_name="walker2d-medium-replay-v2"; num_query=500
#   fi

#   for seed in "${seeds[@]}"; do
#     for ld in "${latents[@]}"; do
#       for hd in "${hiddens[@]}"; do
#         for tk in "${topkps[@]}"; do

#           cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}.pkl"
          
#           if [ -f "$cvae_file" ]; then
#             echo "⚡ SKIP CVAE: ${cvae_file}"
#             continue
#           fi

#           gpu="${gpus[$((job_idx % num_gpus))]}"
#           echo "▶ CVAE train: env=${env}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk} on GPU${gpu}"
#           CUDA_VISIBLE_DEVICES=${gpu} \
#             python -m JaxPref.new_preference_reward_main \
#               --use_human_label True \
#               --comment ${env} \
#               --env ${env_name} \
#               --logging.output_dir './logs/pref_reward' \
#               --batch_size 256 \
#               --num_query ${num_query} \
#               --query_len 100 \
#               --n_epochs 10000 \
#               --skip_flag 0 \
#               --seed ${seed} \
#               --model_type PrefTransformer \
#               --latent_dim ${ld} \
#               --hidden_dim ${hd} \
#               --topkp ${tk} &

#           pids+=($!)
#           job_idx=$((job_idx+1))

#           # 4개 이상 올라가면 하나 끝날 때까지 대기
#           if (( ${#pids[@]} >= num_gpus )); then
#             wait -n
#             # (원하면 여기서 끝난 PID를 pids 배열에서 제거)
#           fi

#         done
#       done
#     done
#   done
# done

# # 남은 CVAE 잡 대기
# wait
# echo "✅ Stage 1 complete: all CVAE models trained."

####################################
# 2) RL 학습 (병렬, 최대 4개 동시)
####################################
# echo "=== Stage 2: Running all RL experiments ==="
# pids=()
# rl_job_idx=0
# (아래 주석 해제해서 쓰세요)
#
# for env in hopper walker; do
#   if [ "$env" = "hopper" ]; then
#     env_name="hopper-medium-expert-v2"
#   else
#     env_name="walker2d-medium-replay-v2"
#   fi
#
#   for seed in "${seeds[@]}"; do
#     for ld in "${latents[@]}"; do
#       for hd in "${hiddens[@]}"; do
#         for tk in "${topkps[@]}"; do
#           ckpt_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}"
#
#           for method in "${methods[@]}"; do
#             for weight in "${weights[@]}"; do
#               # 이미 결과 있으면 건너뛰기
#               rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${env}/${seed}/${ld}_${hd}_${tk}/${method}/${weight}"
#               if [ -d "$rl_dir" ]; then
#                 echo "⚡ SKIP RL: ${rl_dir}"
#                 continue
#               fi
#
#               gpu_rl="${gpus[$((rl_job_idx % num_gpus))]}"
#               echo "▶ RL run: env=${env}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"
#               CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                 python train_offline.py \
#                   --comment ${env} \
#                   --eval_interval 5000 \
#                   --env_name ${env_name} \
#                   --config configs/mujoco_config.py \
#                   --eval_episodes 10 \
#                   --use_reward_model True \
#                   --model_type PrefTransformer \
#                   --ckpt_dir "${ckpt_dir}" \
#                   --seed ${seed} \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --shaping_weight ${weight} \
#                   --method ${method} &
#
#               pids+=($!)
#               rl_job_idx=$((rl_job_idx+1))
#
#               if (( ${#pids[@]} >= num_gpus )); then
#                 wait -n
#               fi
#
#             done
#           done
#
#         done
#       done
#     done
#   done
# done
#
# wait
# echo "✅ Stage 2 complete: all RL experiments done."

# # --- 1) CVAE 보상 모델 학습 (병렬 실행) with state_action flag ---
# #!/usr/bin/env bash
# set -e

# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 공통 실험 설정
# seeds=(0 42 1234)
# topkps=(10)
# state_actions=(True False)

# # Batch별 latent & hidden 설정
# latents_batch1=(16 32)
# hiddens_batch1=(32 64 128)
# latents_batch2=(12)
# hiddens_batch2=(750)

# echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="
# pids=()
# job_idx=0

# # Batch 실행 함수 정의: latents & hiddens 배열을 인자로 받아 모든 조합 실행
# run_batch() {
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2

#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#       num_query=100
#     else
#       env_name="walker2d-medium-replay-v2"
#       num_query=500
#     fi

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do

#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#               if [ -f "$cvae_file" ]; then
#                 echo "⚡ SKIP CVAE (exists): ${cvae_file}"
#                 continue
#               fi

#               gpu="${gpus[$((job_idx % num_gpus))]}"
#               echo "▶ CVAE train: env=${env}, state_action=${sa}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk} on GPU${gpu}"

#               CUDA_VISIBLE_DEVICES=${gpu} \
#                 python -m JaxPref.new_preference_reward_main \
#                   --use_human_label True \
#                   --comment ${env} \
#                   --env ${env_name} \
#                   --logging.output_dir './logs/pref_reward' \
#                   --batch_size 256 \
#                   --num_query ${num_query} \
#                   --query_len 100 \
#                   --n_epochs 10000 \
#                   --skip_flag 0 \
#                   --seed ${seed} \
#                   --model_type PrefTransformer \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --state_action=${sa} \
#                   --topkp ${tk} &

#               pids+=("$!")
#               job_idx=$((job_idx+1))

#               # num_gpus 이상 실행 중이면 하나 끝날 때까지 대기
#               if (( ${#pids[@]} >= num_gpus )); then
#                 wait -n
#               fi

#             done
#           done
#         done
#       done
#     done
#   done
# }

# # 1차 Batch 실행: latent=(16,32), hidden=(32,64,128)
# run_batch latents_batch1 hiddens_batch1
# # 2차 Batch 실행: latent=(12), hidden=(750)
# run_batch latents_batch2 hiddens_batch2

# # 남은 CVAE 잡 대기
# wait

# echo "✅ Stage 1 complete: all CVAE models trained with state_action flag."


# #!/usr/bin/env bash
# set -e

# # GPU 설정
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 공통 실험 설정
# seeds=(0 42 1234 5678 9876)
# topkps=(10)
# state_actions=(True)

# # Batch별 latent & hidden 설정
# latents_batch1=(16)
# hiddens_batch1=(32)

# echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="
# pids=()
# job_idx=0

# # Batch 실행 함수 정의: latents & hiddens 배열을 인자로 받아 모든 조합 실행
# run_batch() {
#   local -n latents_ref=$1

#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#       num_query=100
#     else
#       env_name="walker2d-medium-replay-v2"
#       num_query=500
#     fi

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do

#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#               if [ -f "$cvae_file" ]; then
#                 echo "⚡ SKIP CVAE (exists): ${cvae_file}"
#                 continue
#               fi

#               gpu="${gpus[$((job_idx % num_gpus))]}"
#               echo "▶ CVAE train: env=${env}, state_action=${sa}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk} on GPU${gpu}"

#               # 학습 프로세스를 백그라운드에서 실행하고 로그 기록
#               CUDA_VISIBLE_DEVICES=${gpu} \
#                 python -m JaxPref.new_preference_reward_main \
#                   --use_human_label True \
#                   --comment ${env} \
#                   --env ${env_name} \
#                   --logging.output_dir './logs/pref_reward' \
#                   --batch_size 256 \
#                   --num_query ${num_query} \
#                   --query_len 100 \
#                   --n_epochs 10000 \
#                   --skip_flag 0 \
#                   --seed ${seed} \
#                   --model_type PrefTransformer \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --state_action=${sa} \
#                   --topkp ${tk} \
#                 2>&1 | tee ./logs/error_log_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.txt &

#               pids+=("$!")
#               job_idx=$((job_idx+1))

#               # num_gpus 이상 실행 중이면 하나 끝날 때까지 대기
#               if (( ${#pids[@]} >= num_gpus )); then
#                 wait -n
#               fi

#             done
#           done
#         done
#       done
#     done
#   done

#   # 모든 프로세스가 종료될 때까지 기다림
#   for pid in "${pids[@]}"; do
#     wait "$pid" && echo "Process $pid completed successfully."
#   done
# }

# # 1차 Batch 실행: latent=(16,32), hidden=(32,64,128)
# run_batch latents_batch1 hiddens_batch1
# wait # 첫 번째 배치가 끝날 때까지 기다린 후

# # 2차 Batch 실행: latent=(12), hidden=(750)
# run_batch latents_batch2 hiddens_batch2

# # 남은 CVAE 잡 대기
# wait

# echo "✅ Stage 1 complete: all CVAE models trained with state_action flag."

# #!/usr/bin/env bash
# set -Eeuo pipefail

# # ===== 공용 설정 =====
# gpus=(0 1 2)                # 사용할 실제 GPU 인덱스
# num_gpus=${#gpus[@]}

# # "사실상 비어있음" 판정 기준(필요시 조정)
# MAX_PROCS_PER_GPU=1         # GPU당 동시에 몇 개까지 올릴지 (1 권장)
# MIN_FREE_MEM_MB=4000        # 이 이상 여유 메모리가 있어야 "비어있음"으로 간주
# POLL_SEC=5                  # 빈 GPU를 찾을 때 재시도 간격(초)

# # ===== 실험 설정 =====
# seeds=(0 42 1234 5678 9876)
# topkps=(10)
# state_actions=(True)

# # 배치 구성
# latents_batch2=(16)
# hiddens_batch2=(32)

# # RL 관련 하이퍼파라미터
# methods=( cosine_similarity )
# weights=(1.0)

# # ===== 내부 상태(건들 필요 없음) =====
# declare -A gpu_pids   # gpu_pids[<gpu_index>] = "pid1 pid2 ..."
# declare -A gpu_load   # gpu_load[<gpu_index>] = <현재 개수>
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# have_nvsmi=1
# command -v nvidia-smi >/dev/null 2>&1 || have_nvsmi=0

# # 현재 우리 스크립트가 올린 작업 중 종료된 것 청소
# gc_finished() {
#   for gpu in "${gpus[@]}"; do
#     local alive=""
#     local count=0
#     for pid in ${gpu_pids[$gpu]}; do
#       if kill -0 "$pid" 2>/dev/null; then
#         alive+="$pid "
#         count=$((count+1))
#       fi
#     done
#     gpu_pids[$gpu]="$alive"
#     gpu_load[$gpu]=$count
#   done
# }

# # 실제 GPU 메모리 사용량 조회 (실패 시 큰 수 반환)
# mem_used_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 999999
#   else
#     # nvidia-smi 없으면 메모리 체크 못하므로 "사용중"으로 간주해서 스택 방지
#     echo 999999
#   fi
# }

# # 해당 GPU가 "지금 올려도 되는지" 판단 (동시 수, 여유 메모리 둘 다 확인)
# gpu_is_ok() {
#   local id="$1"
#   local load="${gpu_load[$id]:-0}"
#   if (( load >= MAX_PROCS_PER_GPU )); then
#     return 1
#   fi
#   # 메모리 기준 확인(여유가 MIN_FREE_MEM_MB 이상인지)
#   if (( have_nvsmi == 1 )); then
#     local used
#     used="$(mem_used_mb "$id")"
#     # total을 모르면 used 절대값으로 판단하기 어려우니 "used <= 임계"로 간단히 판정
#     # 여기서는 여유 >= MIN_FREE_MEM_MB <=> used <= (total - MIN_FREE_MEM_MB) 가 더 정확하지만
#     # total 호출을 줄이기 위해 used 절대값 임계로 운용해도 실무에선 충분.
#     # 필요하면 total도 쿼리해서 더 정밀하게 만들 수 있음.
#     if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1000 )); then
#       # used가 매우 적으면 비어있다고 간주(약식). 더 엄격히 하려면 total도 조회해서 비교하세요.
#       return 0
#     fi
#     # 간단 모드: used 기준 말고 프로세스 동시수만 신뢰하려면 여기서 return 0 하세요.
#     # return 0
#     return 1
#   else
#     # nvidia-smi 없으면 동시수 제약만 신뢰
#     return 0
#   fi
# }

# # 빈 슬롯이 날 때까지 기다렸다가, 사용할 GPU 인덱스를 반환
# wait_for_free_gpu() {
#   while true; do
#     gc_finished
#     for id in "${gpus[@]}"; do
#       if gpu_is_ok "$id"; then
#         echo "$id"
#         return 0
#       fi
#     done
#     sleep "$POLL_SEC"
#   done
# }

# # 전체 작업이 끝날 때까지 블록
# wait_all() {
#   while true; do
#     gc_finished
#     local total=0
#     for id in "${gpus[@]}"; do
#       total=$(( total + ${gpu_load[$id]:-0} ))
#     done
#     (( total == 0 )) && break
#     sleep "$POLL_SEC"
#   done
# }

# # 비정상 종료 시 자식 정리
# cleanup() {
#   for id in "${gpus[@]}"; do
#     for pid in ${gpu_pids[$id]}; do
#       kill -TERM "$pid" 2>/dev/null || true
#     done
#   done
# }
# trap cleanup INT TERM

# # ===== 배치 2 → 배치 1 순서로 실행 =====
# for batch in 2 1; do
#   if [ "$batch" = "2" ]; then
#     current_latents=( "${latents_batch2[@]}" )
#     current_hiddens=( "${hiddens_batch2[@]}" )
#     echo "=== 🚀 Run BATCH-2 (latents=${current_latents[*]}, hiddens=${current_hiddens[*]}) ==="
#   else
#     current_latents=( "${latents_batch1[@]}" )
#     current_hiddens=( "${hiddens_batch1[@]}" )
#     echo "=== 🚀 Run BATCH-1 (latents=${current_latents[*]}, hiddens=${current_hiddens[*]}) ==="
#   fi

#   # 환경 루프
#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#     else
#       env_name="walker2d-medium-replay-v2"
#     fi

#     for seed in "${seeds[@]}"; do
#       for sa in "${state_actions[@]}"; do
#         for ld in "${current_latents[@]}"; do
#           for hd in "${current_hiddens[@]}"; do
#             for tk in "${topkps[@]}"; do

#               # (CVAE 학습 경로/파일)
#               cvae_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}_${sa}"
#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"

#               # CVAE 없으면 스킵
#               if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
#                 echo "⏭️  SKIP RL: CVAE not found: $cvae_file or $cvae_dir"
#                 continue
#               fi

#               for method in "${methods[@]}"; do
#                 for weight in "${weights[@]}"; do

#                   rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${env}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
#                   if [ -d "$rl_dir" ]; then
#                     echo "⚡ SKIP RL (exists): ${rl_dir}"
#                     continue
#                   fi

#                   # >>>>>> 여기서 "빈 GPU"가 날 때까지 대기하고 선택 <<<<<<
#                   gpu_rl="$(wait_for_free_gpu)"
#                   echo "▶ RL run: [BATCH-${batch}] env=${env}, seed=${seed}, state_action=${sa}, ld=${ld}, hd=${hd}, topkp=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"

#                   CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                     python train_offline.py \
#                       --comment ${env} \
#                       --eval_interval 5000 \
#                       --env_name ${env_name} \
#                       --config configs/mujoco_config.py \
#                       --eval_episodes 10 \
#                       --use_reward_model True \
#                       --model_type PrefTransformer \
#                       --ckpt_dir "${cvae_dir}" \
#                       --seed ${seed} \
#                       --latent_dim ${ld} \
#                       --hidden_dim ${hd} \
#                       --state_action=${sa} \
#                       --topkp ${tk} \
#                       --shaping_weight ${weight} \
#                       --method ${method} &

#                   pid=$!
#                   gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#                   gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#                 done
#               done

#             done
#           done
#         done
#       done
#     done
#   done

#   # 배치 단위로 싹 마무리 기다림(안전)
#   wait_all
#   echo "✅ BATCH-${batch} complete."
# done

# echo "🎉 All batches complete."


# #!/usr/bin/env bash
# set -Eeuo pipefail

# # ===== 공용 설정 =====
# gpus=(0 1 2)                # 사용할 실제 GPU 인덱스
# num_gpus=${#gpus[@]}

# # "사실상 비어있음" 판정 기준(필요시 조정)
# MAX_PROCS_PER_GPU=1         # GPU당 동시에 몇 개까지 올릴지 (1 권장)
# MIN_FREE_MEM_MB=4000        # 이 이상 여유 메모리가 있어야 "비어있음"으로 간주
# POLL_SEC=5                  # 빈 GPU를 찾을 때 재시도 간격(초)

# # ===== 실험 설정 =====
# seeds=(0 42 1234 5678 9876)
# topkps=(10)
# state_actions=(True)

# # 배치 구성
# latents_batch2=(16)
# hiddens_batch2=(32)

# # RL 관련 하이퍼파라미터
# methods=(negative_distance gaussian_kernel cosine_similarity normalized_distance potential_based)
# weights=(0.1 0.5 1.0 -0.1 -0.5 -1.0)

# # ===== 내부 상태(건들 필요 없음) =====
# declare -A gpu_pids   # gpu_pids[<gpu_index>] = "pid1 pid2 ..."
# declare -A gpu_load   # gpu_load[<gpu_index>] = <현재 개수>
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# have_nvsmi=1
# command -v nvidia-smi >/dev/null 2>&1 || have_nvsmi=0

# # 현재 우리 스크립트가 올린 작업 중 종료된 것 청소
# gc_finished() {
#   for gpu in "${gpus[@]}"; do
#     local alive=""
#     local count=0
#     for pid in ${gpu_pids[$gpu]}; do
#       if kill -0 "$pid" 2>/dev/null; then
#         alive+="$pid "
#         count=$((count+1))
#       fi
#     done
#     gpu_pids[$gpu]="$alive"
#     gpu_load[$gpu]=$count
#   done
# }

# # 실제 GPU 메모리 사용량 조회 (실패 시 큰 수 반환)
# mem_used_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 999999
#   else
#     # nvidia-smi 없으면 메모리 체크 못하므로 "사용중"으로 간주해서 스택 방지
#     echo 999999
#   fi
# }

# # 해당 GPU가 "지금 올려도 되는지" 판단 (동시 수, 여유 메모리 둘 다 확인)
# gpu_is_ok() {
#   local id="$1"
#   local load="${gpu_load[$id]:-0}"
#   if (( load >= MAX_PROCS_PER_GPU )); then
#     return 1
#   fi
#   # 메모리 기준 확인(여유가 MIN_FREE_MEM_MB 이상인지)
#   if (( have_nvsmi == 1 )); then
#     local used
#     used="$(mem_used_mb "$id")"
#     # total을 모르면 used 절대값으로 판단하기 어려우니 "used <= 임계"로 간단히 판정
#     # 여기서는 여유 >= MIN_FREE_MEM_MB <=> used <= (total - MIN_FREE_MEM_MB) 가 더 정확하지만
#     # total 호출을 줄이기 위해 used 절대값 임계로 운용해도 실무에선 충분.
#     # 필요하면 total도 쿼리해서 더 정밀하게 만들 수 있음.
#     if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1000 )); then
#       # used가 매우 적으면 비어있다고 간주(약식). 더 엄격히 하려면 total도 조회해서 비교하세요.
#       return 0
#     fi
#     # 간단 모드: used 기준 말고 프로세스 동시수만 신뢰하려면 여기서 return 0 하세요.
#     # return 0
#     return 1
#   else
#     # nvidia-smi 없으면 동시수 제약만 신뢰
#     return 0
#   fi
# }

# # 빈 슬롯이 날 때까지 기다렸다가, 사용할 GPU 인덱스를 반환
# wait_for_free_gpu() {
#   while true; do
#     gc_finished
#     for id in "${gpus[@]}"; do
#       if gpu_is_ok "$id"; then
#         echo "$id"
#         return 0
#       fi
#     done
#     sleep "$POLL_SEC"
#   done
# }

# # 전체 작업이 끝날 때까지 블록
# wait_all() {
#   while true; do
#     gc_finished
#     local total=0
#     for id in "${gpus[@]}"; do
#       total=$(( total + ${gpu_load[$id]:-0} ))
#     done
#     (( total == 0 )) && break
#     sleep "$POLL_SEC"
#   done
# }

# # 비정상 종료 시 자식 정리
# cleanup() {
#   for id in "${gpus[@]}"; do
#     for pid in ${gpu_pids[$id]}; do
#       kill -TERM "$pid" 2>/dev/null || true
#     done
#   done
# }
# trap cleanup INT TERM

# # ===== 배치 2 → 배치 1 순서로 실행 =====
# for batch in 2 1; do
#   if [ "$batch" = "2" ]; then
#     current_latents=( "${latents_batch2[@]}" )
#     current_hiddens=( "${hiddens_batch2[@]}" )
#     echo "=== 🚀 Run BATCH-2 (latents=${current_latents[*]}, hiddens=${current_hiddens[*]}) ==="
#   else
#     current_latents=( "${latents_batch1[@]}" )
#     current_hiddens=( "${hiddens_batch1[@]}" )
#     echo "=== 🚀 Run BATCH-1 (latents=${current_latents[*]}, hiddens=${current_hiddens[*]}) ==="
#   fi

#   # 환경 루프
#   for env in hopper walker; do
#     if [ "$env" = "hopper" ]; then
#       env_name="hopper-medium-expert-v2"
#     else
#       env_name="walker2d-medium-replay-v2"
#     fi

#     for seed in "${seeds[@]}"; do
#       for sa in "${state_actions[@]}"; do
#         for ld in "${current_latents[@]}"; do
#           for hd in "${current_hiddens[@]}"; do
#             for tk in "${topkps[@]}"; do

#               # (CVAE 학습 경로/파일)
#               cvae_dir="./logs/pref_reward/${env_name}/PrefTransformer/${env}/s${seed}/${ld}_${hd}_${tk}_${sa}"
#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"

#               # CVAE 없으면 스킵
#               if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
#                 echo "⏭️  SKIP RL: CVAE not found: $cvae_file or $cvae_dir"
#                 continue
#               fi

#               for method in "${methods[@]}"; do
#                 for weight in "${weights[@]}"; do

#                   rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${env}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
#                   if [ -d "$rl_dir" ]; then
#                     echo "⚡ SKIP RL (exists): ${rl_dir}"
#                     continue
#                   fi

#                   # >>>>>> 여기서 "빈 GPU"가 날 때까지 대기하고 선택 <<<<<<
#                   gpu_rl="$(wait_for_free_gpu)"
#                   echo "▶ RL run: [BATCH-${batch}] env=${env}, seed=${seed}, state_action=${sa}, ld=${ld}, hd=${hd}, topkp=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"

#                   CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                     python train_offline.py \
#                       --comment ${env} \
#                       --eval_interval 5000 \
#                       --env_name ${env_name} \
#                       --config configs/mujoco_config.py \
#                       --eval_episodes 10 \
#                       --use_reward_model True \
#                       --model_type PrefTransformer \
#                       --ckpt_dir "${cvae_dir}" \
#                       --seed ${seed} \
#                       --latent_dim ${ld} \
#                       --hidden_dim ${hd} \
#                       --state_action=${sa} \
#                       --topkp ${tk} \
#                       --shaping_weight ${weight} \
#                       --method ${method} &

#                   pid=$!
#                   gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#                   gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#                 done
#               done

#             done
#           done
#         done
#       done
#     done
#   done

#   # 배치 단위로 싹 마무리 기다림(안전)
#   wait_all
#   echo "✅ BATCH-${batch} complete."
# done

# echo "🎉 All batches complete."

# # #!/usr/bin/env bash
# set -Eeuo pipefail

# # =========================
# # 공용 설정
# # =========================
# gpus=(0 2)
# num_gpus=${#gpus[@]}

# # 실험 설정
# seeds=(0 42 1234 5678 9876)
# topkps=(10)
# state_actions=(True)

# # --- 사용 배치(배치2 제거) ---
# latents=(16)
# hiddens=(32)

# # --- RL 하이퍼파라미터 ---
# methods=(cosine_similarity)
# weights=(1.0)

# # 4개 환경
# envs=(
#   "hopper-medium-expert-v2"
#   "hopper-medium-replay-v2"
#   "walker2d-medium-expert-v2"
#   "walker2d-medium-replay-v2"
# )

# mkdir -p ./logs ./logs/pref_reward

# # =========================
# # Stage 1: CVAE 학습
# # =========================
# echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="

# job_idx=0
# run_cvae() {
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2
#   local pids=()

#   for env_name in "${envs[@]}"; do
#     # expert/replay에 따라 num_query 설정
#     if [[ "$env_name" == *"expert"* ]]; then
#       num_query=100
#     else
#       num_query=500
#     fi
#     comment="${env_name%%-*}"  # hopper / walker2d

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do
#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#               if [ -f "$cvae_file" ]; then
#                 echo "⚡ SKIP CVAE (exists): ${cvae_file}"
#                 continue
#               fi

#               gpu="${gpus[$((job_idx % num_gpus))]}"
#               echo "▶ CVAE train: env=${env_name}, num_query=${num_query}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk}, sa=${sa} on GPU${gpu}"

#               CUDA_VISIBLE_DEVICES=${gpu} \
#                 python -m JaxPref.new_preference_reward_main \
#                   --use_human_label True \
#                   --comment "${comment}" \
#                   --env "${env_name}" \
#                   --logging.output_dir './logs/pref_reward' \
#                   --batch_size 256 \
#                   --num_query ${num_query} \
#                   --query_len 100 \
#                   --n_epochs 10000 \
#                   --skip_flag 0 \
#                   --seed ${seed} \
#                   --model_type PrefTransformer \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --state_action=${sa} \
#                   --topkp ${tk} \
#                 2>&1 | tee "./logs/error_log_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.txt" &

#               pids+=("$!")
#               job_idx=$((job_idx+1))

#               # 동시에 num_gpus개까지만 실행
#               if (( ${#pids[@]} >= num_gpus )); then
#                 wait -n || true
#               fi
#             done
#           done
#         done
#       done
#     done
#   done

#   # 배치에서 띄운 모든 프로세스 대기
#   for pid in "${pids[@]}"; do
#     wait "$pid" && echo "Process $pid completed successfully." || echo "Process $pid exited non-zero."
#   done
# }

# run_cvae latents hiddens
# echo "✅ Stage 1 complete: all CVAE models trained."

# # =========================
# # Stage 2: RL 학습 (GPU 슬롯 감시, 고침)
# # =========================
# echo "=== Stage 2: Offline RL trainings with GPU slot watcher ==="

# # 내부 상태
# declare -A gpu_pids   # 각 GPU에 내가 띄운 PID 목록(공백 분리 문자열)
# declare -A gpu_load   # 각 GPU에 현재 내 PID 수
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# MAX_PROCS_PER_GPU=1
# POLL_SEC=5
# BACKOFF_SEC=1
# MEM_LIMIT_MB=1000   # 필요시 조정

# have_nvsmi=1
# if ! command -v nvidia-smi >/dev/null 2>&1; then
#   have_nvsmi=0
# fi

# gc_finished() {
#   for gpu in "${gpus[@]}"; do
#     local alive=""
#     local count=0
#     # 내 PID 중 살아있는 것만 유지
#     for pid in ${gpu_pids[$gpu]}; do
#       if kill -0 "$pid" 2>/dev/null; then
#         alive+="$pid "
#         count=$((count+1))
#       fi
#     done
#     gpu_pids[$gpu]="$alive"
#     gpu_load[$gpu]=$count
#   done
# }

# mem_used_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
#   else
#     echo 0
#   fi
# }

# gpu_is_ok() {
#   local id="$1"
#   local load="${gpu_load[$id]:-0}"

#   # 슬롯 비어있으면(내 PID가 0개면) 메모리 체크를 건너뛴다 → 멈춤 방지 핵심
#   if (( load == 0 )); then
#     return 0
#   fi

#   # 슬롯이 남아있는 경우에만(추가로 같은 GPU에 태울 때만) 메모리 확인
#   if (( load < MAX_PROCS_PER_GPU )); then
#     if (( have_nvsmi == 1 )); then
#       local used
#       used="$(mem_used_mb "$id")"
#       [[ "$used" =~ ^[0-9]+$ ]] || used=0
#       (( used <= MEM_LIMIT_MB )) && return 0 || return 1
#     else
#       return 0
#     fi
#   fi

#   # 이미 꽉 찼으면 불가
#   return 1
# }

# wait_for_free_gpu() {
#   local spins=0
#   while true; do
#     gc_finished
#     for id in "${gpus[@]}"; do
#       if gpu_is_ok "$id"; then
#         echo "$id"
#         return 0
#       fi
#     done
#     ((spins++))
#     if (( spins % 12 == 0 )); then
#       # 1분마다 상태 로그
#       echo "[watcher] waiting for free GPU... loads: $(for id in "${gpus[@]}"; do echo -n "GPU${id}=${gpu_load[$id]} "; done)"
#     fi
#     sleep "$POLL_SEC"
#   done
# }

# wait_all() {
#   while true; do
#     gc_finished
#     local total=0
#     for id in "${gpus[@]}"; do
#       total=$(( total + ${gpu_load[$id]:-0} ))
#     done
#     (( total == 0 )) && break
#     sleep "$POLL_SEC"
#   done
# }

# cleanup() {
#   echo "Cleanup: terminating child RL processes..."
#   for id in "${gpus[@]}"; do
#     for pid in ${gpu_pids[$id]}; do
#       kill -TERM "$pid" 2>/dev/null || true
#     done
#   done
# }
# trap cleanup INT TERM

# # RL 실행
# for env_name in "${envs[@]}"; do
#   comment="${env_name%%-*}"  # hopper / walker2d
#   for seed in "${seeds[@]}"; do
#     for sa in "${state_actions[@]}"; do
#       for ld in "${latents[@]}"; do
#         for hd in "${hiddens[@]}"; do
#           for tk in "${topkps[@]}"; do

#             # CVAE 산출물 확인(파일 또는 디렉토리)
#             cvae_dir="./logs/pref_reward/${env_name}/PrefTransformer/${comment}/s${seed}/${ld}_${hd}_${tk}_${sa}"
#             cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#             if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
#               echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $cvae_dir"
#               continue
#             fi

#             for method in "${methods[@]}"; do
#               for weight in "${weights[@]}"; do
#                 rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${comment}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
#                 if [ -d "$rl_dir" ]; then
#                   echo "⚡ SKIP RL (exists): ${rl_dir}"
#                   continue
#                 fi

#                 gpu_rl="$(wait_for_free_gpu)"
#                 echo "▶ RL run: env=${env_name}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"

#                 CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                   python train_offline.py \
#                     --comment "${comment}" \
#                     --eval_interval 5000 \
#                     --env_name "${env_name}" \
#                     --config configs/mujoco_config.py \
#                     --eval_episodes 10 \
#                     --use_reward_model True \
#                     --model_type PrefTransformer \
#                     --ckpt_dir "${cvae_dir}" \
#                     --seed ${seed} \
#                     --latent_dim ${ld} \
#                     --hidden_dim ${hd} \
#                     --state_action=${sa} \
#                     --topkp ${tk} \
#                     --shaping_weight ${weight} \
#                     --method ${method} &

#                 pid=$!
#                 gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#                 gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#                 # 아주 짧은 백오프(로그 정돈, 과한 스핀 방지)
#                 sleep "$BACKOFF_SEC"
#               done
#             done

#           done
#         done
#       done
#     done
#   done
# done

# wait_all
# # echo "🎉 All stages complete."
# # #!/usr/bin/env bash
# set -Eeuo pipefail

# # =========================
# # 공용 설정
# # =========================
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 실험 설정
# # seeds=(0 42 1234 5678 9876)
# topkps=(10)
# state_actions=(False)

# # --- 사용 배치(배치2 제거) ---
# latents=(16)
# hiddens=(32)

# # # --- RL 하이퍼파라미터 ---
# # methods=(cosine_similarity)
# # weights=(1.0)

# # 4개 환경
# # envs=(
# #   "hopper-medium-expert-v2"
# #   "hopper-medium-replay-v2"
# #   "walker2d-medium-expert-v2"
# #   "walker2d-medium-replay-v2"
# # )

# #loss function & weight
# seeds=(0 42 1234)
# methods=(negative_distance  cosine_similarity  potential_based)
# weights=(0.1 0.5 1.0 -0.1 -0.5 -1.0)
# envs=(
#   "hopper-medium-expert-v2"
#   "walker2d-medium-replay-v2"
# )

# #topk
# # topkps=(10 20 -10 -20)

# mkdir -p ./logs ./logs/pref_reward

# # =========================
# # Stage 1: CVAE 학습
# # =========================
# echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="

# job_idx=0
# run_cvae() {
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2
#   local pids=()

#   for env_name in "${envs[@]}"; do
#     # expert/replay에 따라 num_query 설정
#     if [[ "$env_name" == *"expert"* ]]; then
#       num_query=100
#     else
#       num_query=500
#     fi
#     comment="${env_name%%-*}"  # hopper / walker2d

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do
#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#               if [ -f "$cvae_file" ]; then
#                 echo "⚡ SKIP CVAE (exists): ${cvae_file}"
#                 continue
#               fi

#               gpu="${gpus[$((job_idx % num_gpus))]}"
#               echo "▶ CVAE train: env=${env_name}, num_query=${num_query}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk}, sa=${sa} on GPU${gpu}"

#               CUDA_VISIBLE_DEVICES=${gpu} \
#                 python -m JaxPref.new_preference_reward_main \
#                   --use_human_label True \
#                   --comment "${comment}" \
#                   --env "${env_name}" \
#                   --logging.output_dir './logs/pref_reward' \
#                   --batch_size 256 \
#                   --num_query ${num_query} \
#                   --query_len 100 \
#                   --n_epochs 10000 \
#                   --skip_flag 0 \
#                   --seed ${seed} \
#                   --model_type PrefTransformer \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --state_action=${sa} \
#                   --topkp ${tk} \
#                 2>&1 | tee "./logs/error_log_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.txt" &

#               pids+=("$!")
#               job_idx=$((job_idx+1))

#               # 동시에 num_gpus개까지만 실행
#               if (( ${#pids[@]} >= num_gpus )); then
#                 wait -n || true
#               fi
#             done
#           done
#         done
#       done
#     done
#   done

#   # 배치에서 띄운 모든 프로세스 대기
#   for pid in "${pids[@]}"; do
#     wait "$pid" && echo "Process $pid completed successfully." || echo "Process $pid exited non-zero."
#   done
# }

# run_cvae latents hiddens
# echo "✅ Stage 1 complete: all CVAE models trained."

# # =========================
# # Stage 2: RL 학습 (GPU 슬롯 감시, 고침)
# # =========================
# echo "=== Stage 2: Offline RL trainings with GPU slot watcher ==="

# # 내부 상태
# declare -A gpu_pids   # 각 GPU에 내가 띄운 PID 목록(공백 분리 문자열)
# declare -A gpu_load   # 각 GPU에 현재 내 PID 수
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# MAX_PROCS_PER_GPU=1
# POLL_SEC=5
# BACKOFF_SEC=1
# MEM_LIMIT_MB=1000   # 필요시 조정

# have_nvsmi=1
# if ! command -v nvidia-smi >/dev/null 2>&1; then
#   have_nvsmi=0
# fi

# gc_finished() {
#   for gpu in "${gpus[@]}"; do
#     local alive=""
#     local count=0
#     # 내 PID 중 살아있는 것만 유지
#     for pid in ${gpu_pids[$gpu]}; do
#       if kill -0 "$pid" 2>/dev/null; then
#         alive+="$pid "
#         count=$((count+1))
#       fi
#     done
#     gpu_pids[$gpu]="$alive"
#     gpu_load[$gpu]=$count
#   done
# }

# mem_used_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
#   else
#     echo 0
#   fi
# }

# gpu_is_ok() {
#   local id="$1"
#   local load="${gpu_load[$id]:-0}"

#   # 슬롯 비어있으면(내 PID가 0개면) 메모리 체크를 건너뛴다 → 멈춤 방지 핵심
#   if (( load == 0 )); then
#     return 0
#   fi

#   # 슬롯이 남아있는 경우에만(추가로 같은 GPU에 태울 때만) 메모리 확인
#   if (( load < MAX_PROCS_PER_GPU )); then
#     if (( have_nvsmi == 1 )); then
#       local used
#       used="$(mem_used_mb "$id")"
#       [[ "$used" =~ ^[0-9]+$ ]] || used=0
#       (( used <= MEM_LIMIT_MB )) && return 0 || return 1
#     else
#       return 0
#     fi
#   fi

#   # 이미 꽉 찼으면 불가
#   return 1
# }

# wait_for_free_gpu() {
#   local spins=0
#   while true; do
#     gc_finished
#     for id in "${gpus[@]}"; do
#       if gpu_is_ok "$id"; then
#         echo "$id"
#         return 0
#       fi
#     done
#     ((spins++))
#     if (( spins % 12 == 0 )); then
#       # 1분마다 상태 로그
#       echo "[watcher] waiting for free GPU... loads: $(for id in "${gpus[@]}"; do echo -n "GPU${id}=${gpu_load[$id]} "; done)"
#     fi
#     sleep "$POLL_SEC"
#   done
# }

# wait_all() {
#   while true; do
#     gc_finished
#     local total=0
#     for id in "${gpus[@]}"; do
#       total=$(( total + ${gpu_load[$id]:-0} ))
#     done
#     (( total == 0 )) && break
#     sleep "$POLL_SEC"
#   done
# }

# cleanup() {
#   echo "Cleanup: terminating child RL processes..."
#   for id in "${gpus[@]}"; do
#     for pid in ${gpu_pids[$id]}; do
#       kill -TERM "$pid" 2>/dev/null || true
#     done
#   done
# }
# trap cleanup INT TERM

# # RL 실행
# for env_name in "${envs[@]}"; do
#   comment="${env_name%%-*}"  # hopper / walker2d
#   for seed in "${seeds[@]}"; do
#     for sa in "${state_actions[@]}"; do
#       for ld in "${latents[@]}"; do
#         for hd in "${hiddens[@]}"; do
#           for tk in "${topkps[@]}"; do

#             # CVAE 산출물 확인(파일 또는 디렉토리)
#             cvae_dir="./logs/pref_reward/${env_name}/PrefTransformer/${comment}/s${seed}/${ld}_${hd}_${tk}_${sa}"
#             cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#             if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
#               echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $cvae_dir"
#               continue
#             fi

#             for method in "${methods[@]}"; do
#               for weight in "${weights[@]}"; do
#                 rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${comment}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
#                 if [ -d "$rl_dir" ]; then
#                   echo "⚡ SKIP RL (exists): ${rl_dir}"
#                   continue
#                 fi

#                 gpu_rl="$(wait_for_free_gpu)"
#                 echo "▶ RL run: env=${env_name}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"

#                 CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                   python train_offline.py \
#                     --comment "${comment}" \
#                     --eval_interval 5000 \
#                     --env_name "${env_name}" \
#                     --config configs/mujoco_config.py \
#                     --eval_episodes 10 \
#                     --use_reward_model True \
#                     --model_type PrefTransformer \
#                     --ckpt_dir "${cvae_dir}" \
#                     --seed ${seed} \
#                     --latent_dim ${ld} \
#                     --hidden_dim ${hd} \
#                     --state_action=${sa} \
#                     --topkp ${tk} \
#                     --shaping_weight ${weight} \
#                     --method ${method} &

#                 pid=$!
#                 gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#                 gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#                 # 아주 짧은 백오프(로그 정돈, 과한 스핀 방지)
#                 sleep "$BACKOFF_SEC"
#               done
#             done

#           done
#         done
#       done
#     done
#   done
# done

# wait_all
# echo "🎉 All stages complete."

# #!/usr/bin/env bash
# set -Eeuo pipefail

# # =========================
# # 공용 설정
# # =========================
# gpus=(0 1 2)
# num_gpus=${#gpus[@]}

# # 실험 설정
# # seeds=(0 42 1234 5678 9876)
# topkps=(10)
# state_actions=(False)

# # --- 사용 배치(배치2 제거) ---
# latents=(16)
# hiddens=(32)

# # # --- RL 하이퍼파라미터 ---
# # methods=(cosine_similarity)
# # weights=(1.0)

# # 4개 환경
# # envs=(
# #   "hopper-medium-expert-v2"
# #   "hopper-medium-replay-v2"
# #   "walker2d-medium-expert-v2"
# #   "walker2d-medium-replay-v2"
# # )

# #loss function & weight
# seeds=(0 42 1234)
# methods=(negative_distance  cosine_similarity  potential_based)
# weights=(0.1 0.5 1.0 -0.1 -0.5 -1.0)
# envs=(
#   "hopper-medium-expert-v2"
#   "walker2d-medium-replay-v2"
# )

# #topk
# # topkps=(10 20 -10 -20)

# mkdir -p ./logs ./logs/pref_reward

# # =========================
# # Stage 1: CVAE 학습
# # =========================
# echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="

# job_idx=0
# run_cvae() {
#   local -n latents_ref=$1
#   local -n hiddens_ref=$2
#   local pids=()

#   for env_name in "${envs[@]}"; do
#     # expert/replay에 따라 num_query 설정
#     if [[ "$env_name" == *"expert"* ]]; then
#       num_query=100
#     else
#       num_query=500
#     fi
#     comment="${env_name%%-*}"  # hopper / walker2d

#     for sa in "${state_actions[@]}"; do
#       for seed in "${seeds[@]}"; do
#         for ld in "${latents_ref[@]}"; do
#           for hd in "${hiddens_ref[@]}"; do
#             for tk in "${topkps[@]}"; do
#               cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#               if [ -f "$cvae_file" ]; then
#                 echo "⚡ SKIP CVAE (exists): ${cvae_file}"
#                 continue
#               fi

#               gpu="${gpus[$((job_idx % num_gpus))]}"
#               echo "▶ CVAE train: env=${env_name}, num_query=${num_query}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk}, sa=${sa} on GPU${gpu}"

#               CUDA_VISIBLE_DEVICES=${gpu} \
#                 python -m JaxPref.new_preference_reward_main \
#                   --use_human_label True \
#                   --comment "${comment}" \
#                   --env "${env_name}" \
#                   --logging.output_dir './logs/pref_reward' \
#                   --batch_size 256 \
#                   --num_query ${num_query} \
#                   --query_len 100 \
#                   --n_epochs 10000 \
#                   --skip_flag 0 \
#                   --seed ${seed} \
#                   --model_type PrefTransformer \
#                   --latent_dim ${ld} \
#                   --hidden_dim ${hd} \
#                   --state_action=${sa} \
#                   --topkp ${tk} \
#                 2>&1 | tee "./logs/error_log_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.txt" &

#               pids+=("$!")
#               job_idx=$((job_idx+1))

#               # 동시에 num_gpus개까지만 실행
#               if (( ${#pids[@]} >= num_gpus )); then
#                 wait -n || true
#               fi
#             done
#           done
#         done
#       done
#     done
#   done

#   # 배치에서 띄운 모든 프로세스 대기
#   for pid in "${pids[@]}"; do
#     wait "$pid" && echo "Process $pid completed successfully." || echo "Process $pid exited non-zero."
#   done
# }

# run_cvae latents hiddens
# echo "✅ Stage 1 complete: all CVAE models trained."

# # =========================
# # Stage 2: RL 학습 (GPU 슬롯 감시, 고침)
# # =========================
# echo "=== Stage 2: Offline RL trainings with GPU slot watcher ==="

# # 내부 상태
# declare -A gpu_pids   # 각 GPU에 내가 띄운 PID 목록(공백 분리 문자열)
# declare -A gpu_load   # 각 GPU에 현재 내 PID 수
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# MAX_PROCS_PER_GPU=1
# POLL_SEC=5
# BACKOFF_SEC=1
# MEM_LIMIT_MB=1000   # (기존 변수 유지; 아래 free 기준이 우선)

# # === 추가: 전체 프로세스/메모리 기준 ===
# MAX_PROCS_TOTAL_PER_GPU=3   # (나+타인) 이 이상이면 대기
# MIN_FREE_MB=6000            # free 메모리가 이 이상이어야 시작

# have_nvsmi=1
# if ! command -v nvidia-smi >/dev/null 2>&1; then
#   have_nvsmi=0
# fi

# gc_finished() {
#   for gpu in "${gpus[@]}"; do
#     local alive=""
#     local count=0
#     # 내 PID 중 살아있는 것만 유지
#     for pid in ${gpu_pids[$gpu]}; do
#       if kill -0 "$pid" 2>/dev/null; then
#         alive+="$pid "
#         count=$((count+1))
#       fi
#     done
#     gpu_pids[$gpu]="$alive"
#     gpu_load[$gpu]=$count
#   done
# }

# mem_used_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
#   else
#     echo 0
#   fi
# }

# # === 추가: free 메모리, 전체 프로세스 수 조회 ===
# mem_free_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
#   else
#     echo 0
#   fi
# }

# proc_count_total() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     # 해당 GPU에 매달린 compute 프로세스 수(나+타인)
#     nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' | wc -l | tr -d ' '
#   else
#     echo 0
#   fi
# }

# gpu_is_ok() {
#   local id="$1"
#   local load="${gpu_load[$id]:-0}"

#   # (변경) 내 PID가 0개여도 무조건 시스템 상태 확인
#   if (( have_nvsmi == 1 )); then
#     local free total_procs
#     free="$(mem_free_mb "$id")"
#     total_procs="$(proc_count_total "$id")"
#     [[ "$free" =~ ^[0-9]+$ ]] || free=0
#     [[ "$total_procs" =~ ^[0-9]+$ ]] || total_procs=0

#     # 전체 프로세스 상한
#     if (( total_procs >= MAX_PROCS_TOTAL_PER_GPU )); then
#       return 1
#     fi

#     # 여유 메모리 기준
#     if (( free < MIN_FREE_MB )); then
#       return 1
#     fi

#     # 내 동시 슬롯 확인
#     if (( load < MAX_PROCS_PER_GPU )); then
#       return 0
#     else
#       return 1
#     fi
#   else
#     # nvidia-smi 없으면 기존 내 슬롯 기준만
#     if (( load < MAX_PROCS_PER_GPU )); then
#       return 0
#     else
#       return 1
#     fi
#   fi
# }

# wait_for_free_gpu() {
#   local spins=0
#   while true; do
#     gc_finished
#     for id in "${gpus[@]}"; do
#       if gpu_is_ok "$id"; then
#         echo "$id"
#         return 0
#       fi
#     done
#     ((spins++))
#     if (( spins % 12 == 0 )); then
#       # 1분마다 상태 로그
#       echo "[watcher] waiting for free GPU... $(for id in "${gpus[@]}"; do \
#         fm=$(mem_free_mb "$id"); tp=$(proc_count_total "$id"); echo -n "GPU${id}:free=${fm}MB procs=${tp} my=${gpu_load[$id]} | "; done)"
#     fi
#     sleep "$POLL_SEC"
#   done
# }

# wait_all() {
#   while true; do
#     gc_finished
#     local total=0
#     for id in "${gpus[@]}"; do
#       total=$(( total + ${gpu_load[$id]:-0} ))
#     done
#     (( total == 0 )) && break
#     sleep "$POLL_SEC"
#   done
# }

# cleanup() {
#   echo "Cleanup: terminating child RL processes..."
#   for id in "${gpus[@]}"; do
#     for pid in ${gpu_pids[$id]}; do
#       kill -TERM "$pid" 2>/dev/null || true
#     done
#   done
# }
# trap cleanup INT TERM

# # RL 실행
# for env_name in "${envs[@]}"; do
#   comment="${env_name%%-*}"  # hopper / walker2d
#   for seed in "${seeds[@]}"; do
#     for sa in "${state_actions[@]}"; do
#       for ld in "${latents[@]}"; do
#         for hd in "${hiddens[@]}"; do
#           for tk in "${topkps[@]}"; do

#             # CVAE 산출물 확인(파일 또는 디렉토리)
#             cvae_dir="./logs/pref_reward/${env_name}/PrefTransformer/${comment}/s${seed}/${ld}_${hd}_${tk}_${sa}"
#             cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#             if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
#               echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $cvae_dir"
#               continue
#             fi

#             for method in "${methods[@]}"; do
#               for weight in "${weights[@]}"; do
#                 rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${comment}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
#                 if [ -d "$rl_dir" ]; then
#                   echo "⚡ SKIP RL (exists): ${rl_dir}"
#                   continue
#                 fi

#                 gpu_rl="$(wait_for_free_gpu)"
#                 echo "▶ RL run: env=${env_name}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"

#                 CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                   python train_offline.py \
#                     --comment "${comment}" \
#                     --eval_interval 5000 \
#                     --env_name "${env_name}" \
#                     --config configs/mujoco_config.py \
#                     --eval_episodes 10 \
#                     --use_reward_model True \
#                     --model_type PrefTransformer \
#                     --ckpt_dir "${cvae_dir}" \
#                     --seed ${seed} \
#                     --latent_dim ${ld} \
#                     --hidden_dim ${hd} \
#                     --state_action=${sa} \
#                     --topkp ${tk} \
#                     --shaping_weight ${weight} \
#                     --method ${method} &

#                 pid=$!
#                 gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#                 gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#                 # 아주 짧은 백오프(로그 정돈, 과한 스핀 방지)
#                 sleep "$BACKOFF_SEC"
#               done
#             done

#           done
#         done
#       done
#     done
#   done
# done

# wait_all
# echo "🎉 All stages complete."
#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# 공용 설정
# =========================
gpus=(0 1 2)
num_gpus=${#gpus[@]}

# 실험 설정
# seeds=(0 42 1234 5678 9876)
topkps=(10)
state_actions=(False)

# --- 사용 배치(배치2 제거) ---
latents=(16)
hiddens=(32)

# --- RL 하이퍼파라미터 ---
methods=(cosine_similarity)
weights=(1.0)

# 4개 환경
# envs=(
#   "hopper-medium-expert-v2"
#   "hopper-medium-replay-v2"
#   "walker2d-medium-expert-v2"
#   "walker2d-medium-replay-v2"
# )

#loss function & weight
seeds=(0 42 1234 5678 9876)
# methods=(negative_distance  cosine_similarity  potential_based)
# weights=(0.1 0.5 1.0 -0.1 -0.5 -1.0)
envs=(
  "plate-slide-v2"
  "drawer-open-v2"
)

#topk
topkps=(10)

mkdir -p ./logs ./logs/pref_reward

# =========================
# Stage 1: CVAE 학습
# =========================
echo "=== Stage 1: Parallel CVAE trainings (up to ${num_gpus} at once) ==="

job_idx=0
run_cvae() {
  local -n latents_ref=$1
  local -n hiddens_ref=$2
  local pids=()

  for env_name in "${envs[@]}"; do
    # expert/replay에 따라 num_query 설정
    comment="${env_name%%-*}"  # hopper / walker2d

    for sa in "${state_actions[@]}"; do
      for seed in "${seeds[@]}"; do
        for ld in "${latents_ref[@]}"; do
          for hd in "${hiddens_ref[@]}"; do
            for tk in "${topkps[@]}"; do
              cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
              if [ -f "$cvae_file" ]; then
                echo "⚡ SKIP CVAE (exists): ${cvae_file}"
                continue
              fi

              gpu="${gpus[$((job_idx % num_gpus))]}"
              echo "▶ CVAE train: env=${env_name}, seed=${seed}, ld=${ld}, hd=${hd}, topkp=${tk}, sa=${sa} on GPU${gpu}"

              CUDA_VISIBLE_DEVICES=${gpu} \
                python -m JaxPref.new_preference_reward_main_meta_world \
                  --use_human_label True \
                  --comment "${comment}" \
                  --env "${env_name}" \
                  --logging.output_dir './logs/pref_reward' \
                  --batch_size 256 \
                  --query_len 100 \
                  --n_epochs 10000 \
                  --skip_flag 0 \
                  --seed ${seed} \
                  --model_type PrefTransformer \
                  --latent_dim ${ld} \
                  --hidden_dim ${hd} \
                  --state_action=${sa} \
                  --topkp ${tk} \
                2>&1 | tee "./logs/error_log_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.txt" &

              pids+=("$!")
              job_idx=$((job_idx+1))

              # 동시에 num_gpus개까지만 실행
              if (( ${#pids[@]} >= num_gpus )); then
                wait -n || true
              fi
            done
          done
        done
      done
    done
  done

  # 배치에서 띄운 모든 프로세스 대기
  for pid in "${pids[@]}"; do
    wait "$pid" && echo "Process $pid completed successfully." || echo "Process $pid exited non-zero."
  done
}

run_cvae latents hiddens
echo "✅ Stage 1 complete: all CVAE models trained."

# =========================
# Stage 2: RL 학습 (GPU 슬롯 감시, 고침)
# =========================
# echo "=== Stage 2: Offline RL trainings with GPU slot watcher ==="

# # 내부 상태
# declare -A gpu_pids   # 각 GPU에 내가 띄운 PID 목록(공백 분리 문자열)
# declare -A gpu_load   # 각 GPU에 현재 내 PID 수
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# MAX_PROCS_PER_GPU=1
# POLL_SEC=5
# BACKOFF_SEC=1
# MEM_LIMIT_MB=1000   # (기존 변수 유지; 아래 free 기준이 우선)

# # === 추가: 전체 프로세스/메모리 기준 ===
# MAX_PROCS_TOTAL_PER_GPU=3   # (나+타인) 이 이상이면 대기
# MIN_FREE_MB=6000            # free 메모리가 이 이상이어야 시작

# have_nvsmi=1
# if ! command -v nvidia-smi >/dev/null 2>&1; then
#   have_nvsmi=0
# fi

# # --- 여기만 핵심 수정: 종료/좀비 PID를 즉시 회수해서 슬롯 해제 ---
# gc_finished() {
#   for gpu in "${gpus[@]}"; do
#     local alive=""
#     local count=0
#     for pid in ${gpu_pids[$gpu]}; do
#       # ps가 있으면 상태 확인 (Z=좀비). 없거나 사라졌으면 wait로 즉시 회수.
#       if ps -p "$pid" -o stat= >/dev/null 2>&1; then
#         st="$(ps -p "$pid" -o stat= | awk '{print $1}')"
#         if [[ "$st" == Z* ]]; then
#           # 좀비: 부모가 wait 해줘야 완전히 사라짐(즉시 반환)
#           wait "$pid" 2>/dev/null || true
#         else
#           # 여전히 실행 중
#           alive+="$pid "
#           count=$((count+1))
#         fi
#       else
#         # 이미 종료: wait로 회수(즉시 반환)
#         wait "$pid" 2>/dev/null || true
#       fi
#     done
#     gpu_pids[$gpu]="$alive"
#     gpu_load[$gpu]=$count
#   done
# }

# mem_used_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
#   else
#     echo 0
#   fi
# }

# # === 추가: free 메모리, 전체 프로세스 수 조회 ===
# mem_free_mb() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     nvidia-smi -i "$id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
#   else
#     echo 0
#   fi
# }

# proc_count_total() {
#   local id="$1"
#   if (( have_nvsmi == 1 )); then
#     # 해당 GPU에 매달린 compute 프로세스 수(나+타인)
#     nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' | wc -l | tr -d ' '
#   else
#     echo 0
#   fi
# }

# gpu_is_ok() {
#   local id="$1"
#   local load="${gpu_load[$id]:-0}"

#   # (변경) 내 PID가 0개여도 무조건 시스템 상태 확인
#   if (( have_nvsmi == 1 )); then
#     local free total_procs
#     free="$(mem_free_mb "$id")"
#     total_procs="$(proc_count_total "$id")"
#     [[ "$free" =~ ^[0-9]+$ ]] || free=0
#     [[ "$total_procs" =~ ^[0-9]+$ ]] || total_procs=0

#     # 전체 프로세스 상한
#     if (( total_procs >= MAX_PROCS_TOTAL_PER_GPU )); then
#       return 1
#     fi

#     # 여유 메모리 기준
#     if (( free < MIN_FREE_MB )); then
#       return 1
#     fi

#     # 내 동시 슬롯 확인
#     if (( load < MAX_PROCS_PER_GPU )); then
#       return 0
#     else
#       return 1
#     fi
#   else
#     # nvidia-smi 없으면 기존 내 슬롯 기준만
#     if (( load < MAX_PROCS_PER_GPU )); then
#       return 0
#     else
#       return 1
#     fi
#   fi
# }

# wait_for_free_gpu() {
#   local spins=0
#   while true; do
#     gc_finished
#     for id in "${gpus[@]}"; do
#       if gpu_is_ok "$id"; then
#         echo "$id"
#         return 0
#       fi
#     done
#     ((spins++))
#     if (( spins % 12 == 0 )); then
#       # 1분마다 상태 로그
#       echo "[watcher] waiting for free GPU... $(for id in "${gpus[@]}"; do \
#         fm=$(mem_free_mb "$id"); tp=$(proc_count_total "$id"); echo -n "GPU${id}:free=${fm}MB procs=${tp} my=${gpu_load[$id]} | "; done)"
#     fi
#     sleep "$POLL_SEC"
#   done
# }

# wait_all() {
#   while true; do
#     gc_finished
#     local total=0
#     for id in "${gpus[@]}"; do
#       total=$(( total + ${gpu_load[$id]:-0} ))
#     done
#     (( total == 0 )) && break
#     sleep "$POLL_SEC"
#   done
# }

# cleanup() {
#   echo "Cleanup: terminating child RL processes..."
#   for id in "${gpus[@]}"; do
#     for pid in ${gpu_pids[$id]}; do
#       kill -TERM "$pid" 2>/dev/null || true
#       # 종료 신호 후 즉시 회수 시도
#       wait "$pid" 2>/dev/null || true
#     done
#   done
# }
# trap cleanup INT TERM

# # RL 실행
# for env_name in "${envs[@]}"; do
#   comment="${env_name%%-*}"  # hopper / walker2d
#   for seed in "${seeds[@]}"; do
#     for sa in "${state_actions[@]}"; do
#       for ld in "${latents[@]}"; do
#         for hd in "${hiddens[@]}"; do
#           for tk in "${topkps[@]}"; do

#             # CVAE 산출물 확인(파일 또는 디렉토리)
#             cvae_dir="./logs/pref_reward/${env_name}/PrefTransformer/${comment}/s${seed}/${ld}_${hd}_${tk}_${sa}"
#             cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
#             if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
#               echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $cvae_dir"
#               continue
#             fi

#             for method in "${methods[@]}"; do
#               for weight in "${weights[@]}"; do
#                 rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${comment}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
#                 if [ -d "$rl_dir" ]; then
#                   echo "⚡ SKIP RL (exists): ${rl_dir}"
#                   continue
#                 fi

#                 gpu_rl="$(wait_for_free_gpu)"
#                 echo "▶ RL run: env=${env_name}, seed=${seed}, sa=${sa}, ld=${ld}, hd=${hd}, tk=${tk}, method=${method}, weight=${weight} on GPU${gpu_rl}"

#                 CUDA_VISIBLE_DEVICES=${gpu_rl} \
#                   python train_offline.py \
#                     --comment "${comment}" \
#                     --eval_interval 5000 \
#                     --env_name "${env_name}" \
#                     --config configs/mujoco_config.py \
#                     --eval_episodes 10 \
#                     --use_reward_model True \
#                     --model_type PrefTransformer \
#                     --ckpt_dir "${cvae_dir}" \
#                     --seed ${seed} \
#                     --latent_dim ${ld} \
#                     --hidden_dim ${hd} \
#                     --state_action=${sa} \
#                     --topkp ${tk} \
#                     --shaping_weight ${weight} \
#                     --method ${method} &

#                 pid=$!
#                 gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#                 gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#                 # 아주 짧은 백오프(로그 정돈, 과한 스핀 방지)
#                 sleep "$BACKOFF_SEC"
#               done
#             done

#           done
#         done
#       done
#     done
#   done
# done

# wait_all
# echo "🎉 All stages complete."

# =========================
# Stage 2: RL 학습 (작업 큐 스케줄러)
# =========================
echo "=== Stage 2: Offline RL trainings with a GPU job dispatcher ==="

# 내부 상태
declare -A gpu_pids   # 각 GPU에 내가 띄운 PID 목록(공백 분리 문자열)
declare -A gpu_load   # 각 GPU에 현재 내 PID 수
for gpu in "${gpus[@]}"; do
  gpu_pids[$gpu]=""
  gpu_load[$gpu]=0
done

MAX_PROCS_PER_GPU=1        # 각 GPU에 동시에 올릴 내 프로세스 수
POLL_SEC=5                 # 폴링 주기
BACKOFF_SEC=1              # 프로세스 띄운 직후 약간 대기
MAX_PROCS_TOTAL_PER_GPU=3  # (나+타인) 전체 프로세스 수 상한
MIN_FREE_MB=6000           # 시작 전 요구되는 free 메모리(MB)

have_nvsmi=1
if ! command -v nvidia-smi >/dev/null 2>&1; then
  have_nvsmi=0
fi

gc_finished() {
  # 종료/좀비를 회수해 슬롯 해제
  for gpu in "${gpus[@]}"; do
    local alive=""
    local count=0
    for pid in ${gpu_pids[$gpu]}; do
      if ps -p "$pid" -o stat= >/dev/null 2>&1; then
        st="$(ps -p "$pid" -o stat= | awk '{print $1}')"
        if [[ "$st" == Z* ]]; then
          wait "$pid" 2>/dev/null || true
        else
          alive+="$pid "
          count=$((count+1))
        fi
      else
        wait "$pid" 2>/dev/null || true
      fi
    done
    gpu_pids[$gpu]="$alive"
    gpu_load[$gpu]=$count
  done
}

mem_free_mb() {
  local id="$1"
  if (( have_nvsmi == 1 )); then
    nvidia-smi -i "$id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0
  else
    echo 0
  fi
}

proc_count_total() {
  local id="$1"
  if (( have_nvsmi == 1 )); then
    nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

gpu_is_ok() {
  local id="$1"
  local load="${gpu_load[$id]:-0}"

  if (( have_nvsmi == 1 )); then
    local free total_procs
    free="$(mem_free_mb "$id")"
    total_procs="$(proc_count_total "$id")"
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    [[ "$total_procs" =~ ^[0-9]+$ ]] || total_procs=0

    (( total_procs < MAX_PROCS_TOTAL_PER_GPU )) || return 1
    (( free >= MIN_FREE_MB )) || return 1
    (( load < MAX_PROCS_PER_GPU )) && return 0 || return 1
  else
    (( load < MAX_PROCS_PER_GPU )) && return 0 || return 1
  fi
}

running_total() {
  local total=0
  for id in "${gpus[@]}"; do
    total=$(( total + ${gpu_load[$id]:-0} ))
  done
  echo "$total"
}

cleanup() {
  echo "Cleanup: terminating child RL processes..."
  for id in "${gpus[@]}"; do
    for pid in ${gpu_pids[$id]}; do
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  done
}
trap cleanup INT TERM

# -------------------------
# 1) 작업 큐 구성 (필요한 것만 enqueue)
# -------------------------
jobs=()

for env_name in "${envs[@]}"; do
  comment="${env_name%%-*}"  # hopper / walker2d
  for seed in "${seeds[@]}"; do
    for sa in "${state_actions[@]}"; do
      for ld in "${latents[@]}"; do
        for hd in "${hiddens[@]}"; do
          for tk in "${topkps[@]}"; do

            # CVAE 산출물 확인
            cvae_dir="./logs/pref_reward/metaworld/PrefTransformer/${comment}/s${seed}${ld}_${hd}_${tk}_${sa}"
            cvae_file="./subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
            if [ ! -f "$cvae_file" ] && [ ! -d "$cvae_dir" ]; then
              echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $cvae_dir"
              continue
            fi

            for method in "${methods[@]}"; do
              for weight in "${weights[@]}"; do
                rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${comment}/${seed}/subgoal_vae_${env_name}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${method}/${weight}"
                if [ -d "$rl_dir" ]; then
                  echo "⚡ SKIP enqueue (already done): ${rl_dir}"
                  continue
                fi

                # 실제 실행 커맨드(한 줄)로 큐에 넣기
                jobs+=( \
"python meta_world_train_offline.py \
  --comment ${comment} \
  --eval_interval 5000 \
  --env_name ${env_name} \
  --config configs/mujoco_config.py \
  --eval_episodes 10 \
  --use_reward_model True \
  --model_type PrefTransformer \
  --ckpt_dir ${cvae_dir} \
  --seed ${seed} \
  --latent_dim ${ld} \
  --hidden_dim ${hd} \
  --state_action=${sa} \
  --topkp ${tk} \
  --shaping_weight ${weight} \
  --method ${method}" )
              done
            done

          done
        done
      done
    done
  done
done

echo "🧾 Enqueued jobs: ${#jobs[@]}"

# -------------------------
# 2) 디스패처 루프
# -------------------------
next_job_idx=0
total_jobs=${#jobs[@]}

while : ; do
  gc_finished

  # 모든 잡을 배치했고, 실행 중도 없으면 종료
  if (( next_job_idx >= total_jobs )) && (( $(running_total) == 0 )); then
    break
  fi

  # 가능한 GPU에 새 잡 투입
  for id in "${gpus[@]}"; do
    (( next_job_idx < total_jobs )) || break
    if gpu_is_ok "$id"; then
      cmd="${jobs[$next_job_idx]}"
      echo "▶ RL run on GPU${id}: ${cmd}"
      CUDA_VISIBLE_DEVICES=${id} bash -c "$cmd" &
      pid=$!
      gpu_pids[$id]="${gpu_pids[$id]} $pid"
      gpu_load[$id]=$(( ${gpu_load[$id]:-0} + 1 ))
      next_job_idx=$(( next_job_idx + 1 ))
      sleep "$BACKOFF_SEC"
    fi
  done

  # 상태 로그(1분마다)
  if (( RANDOM % 12 == 0 )); then
    status=""
    for id in "${gpus[@]}"; do
      fm=$(mem_free_mb "$id"); tp=$(proc_count_total "$id")
      status+="GPU${id}:free=${fm}MB procs=${tp} my=${gpu_load[$id]} | "
    done
    echo "[dispatcher] ${status}"
  fi

  sleep "$POLL_SEC"
done

echo "🎉 Stage 2 complete: all RL jobs finished."
echo "🎉 All stages complete."

