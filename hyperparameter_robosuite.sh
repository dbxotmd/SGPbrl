#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# 공용 설정 (로보수이트용)
# =========================
gpus=(0 1 2)
num_gpus=${#gpus[@]}

# 실험 설정 (요청: latent_dim/hidden_dim 유지)
seeds=(0 42 1234)
# seeds=(0)
latents=(16)
hiddens=(32)
state_actions=(False)   # 필요 시 False도 추가

# 로보수이트 환경 / 데이터셋 타입
# envs=("Can" "Lift")
# dtypes=("mh" "ph")
envs=("Can")
dtypes=("mh")
# --- RL 하이퍼파라미터 ---
methods=(cosine_similarity)
weights=(1.0)
topkps=(20 -20)
robosuite_dataset_path="/home/hgmin/.robomimic/datasets"

# query_len 규칙: mh=50, ph=100 (요청사항)
declare -A qlen_map=( ["mh"]=100 ["ph"]=50 )
# declare -A qlen_map=( ["mh"]=50 ["ph"]=100 ) ##결과를 보기 위해 현재 query length값을 반대로 해서 실험 돌리는 중

# num_query는 예시 값으로 설정 (쉽게 수정 가능)
#   - mh: 500 (당신 예시와 동일)
#   - ph: 100   (예시 중 Lift-ph에서 0 사용)
declare -A nquery_map=( ["mh"]=500 ["ph"]=100 )

# 공용 러닝 파라미터
batch_size=256
n_epochs=10000
skip_flag=0
model_type="PrefTransformer"

# RL (robosuite_train_offline.py) 설정
eval_interval=100000
eval_episodes=10
rl_config="configs/adroit_config.py"

log_root="./logs/pref_reward"

mkdir -p "${log_root}"

# =========================
# Stage 1: 보상모델(CVAE/PrefTransformer) 학습
# =========================
echo "=== Stage 1: Parallel Robosuite PrefTransformer trainings (up to ${num_gpus} at once) ==="

job_idx=0
pids=()

run_stage1() {
  local -n envs_ref=$1
  local -n dtypes_ref=$2
  local -n latents_ref=$3
  local -n hiddens_ref=$4

  for env_name in "${envs_ref[@]}"; do
    # 소문자 코멘트 태그 (디렉토리 정리용)
    env_tag="$(echo "${env_name}" | tr '[:upper:]' '[:lower:]')"

    for dtype in "${dtypes_ref[@]}"; do
      qlen="${qlen_map[$dtype]}"
      num_query="${nquery_map[$dtype]}"
      comment="${env_tag}-${dtype}"

      for sa in "${state_actions[@]}"; do
        for seed in "${seeds[@]}"; do
          for ld in "${latents_ref[@]}"; do
            for hd in "${hiddens_ref[@]}"; do
              for tk in "${topkps[@]}"; do
                # 산출물(체크포인트) 디렉토리 (robosuite 예시 경로 패턴)
                ckpt_dir="${log_root}/${env_name}/${model_type}/${comment}/s${seed}/${ld}_${hd}_${tk}_${sa}"
                cvae_file="./subgoal_vae_${env_name}_${dtype}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
                if [ -f "$cvae_file" ] && [ -d "$ckpt_dir" ]; then
                  echo "⚡ SKIP Stage1 (exists): ${ckpt_dir} and ${cvae_file}"
                  continue
                fi

                gpu="${gpus[$((job_idx % num_gpus))]}"
                echo "▶ Stage1 train: env=${env_name}, dtype=${dtype}, qlen=${qlen}, num_query=${num_query}, seed=${seed}, ld=${ld}, hd=${hd}, sa=${sa} tk=${tk} on GPU${gpu}"

                CUDA_VISIBLE_DEVICES=${gpu} \
                  python -m JaxPref.new_preference_reward_main \
                    --use_human_label True \
                    --comment "${comment}" \
                    --robosuite True \
                    --robosuite_dataset_type "${dtype}" \
                    --robosuite_dataset_path "${robosuite_dataset_path}" \
                    --env "${env_name}" \
                    --logging.output_dir "${log_root}" \
                    --batch_size "${batch_size}" \
                    --num_query "${num_query}" \
                    --query_len "${qlen}" \
                    --n_epochs "${n_epochs}" \
                    --skip_flag "${skip_flag}" \
                    --seed "${seed}" \
                    --model_type "${model_type}" \
                    --latent_dim "${ld}" \
                    --hidden_dim "${hd}" \
                    --state_action="${sa}" \
                    --topkp ${tk} \
                  2>&1 | tee "./logs/stage1_${env_name}_${dtype}_s${seed}_${ld}_${hd}_${sa}.log" &

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
  done

  # 띄운 모든 프로세스 대기
  for pid in "${pids[@]}"; do
    wait "$pid" && echo "Process $pid completed successfully." || echo "Process $pid exited non-zero."
  done
}

run_stage1 envs dtypes latents hiddens
echo "✅ Stage 1 complete: all Robosuite reward models trained."

# # =========================
# # Stage 2: Offline RL (robosuite_train_offline.py) + GPU 슬롯 감시
# # =========================
# echo "=== Stage 2: Robosuite RL trainings with GPU slot watcher ==="

# declare -A gpu_pids
# declare -A gpu_load
# for gpu in "${gpus[@]}"; do
#   gpu_pids[$gpu]=""
#   gpu_load[$gpu]=0
# done

# MAX_PROCS_PER_GPU=1
# POLL_SEC=5
# BACKOFF_SEC=1
# MEM_LIMIT_MB=1000

# have_nvsmi=1
# if ! command -v nvidia-smi >/dev/null 2>&1; then
#   have_nvsmi=0
# fi

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

#   if (( load == 0 )); then
#     return 0
#   fi
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
#   env_tag="$(echo "${env_name}" | tr '[:upper:]' '[:lower:]')"
#   for dtype in "${dtypes[@]}"; do
#     comment="${env_tag}-${dtype}"
#     for seed in "${seeds[@]}"; do
#       for sa in "${state_actions[@]}"; do
#         for tk in "${topkps[@]}"; do
#           # Stage1 산출물 경로
#           ckpt_dir="${log_root}/${env_name}/${model_type}/${comment}/s${seed}/16_32_${tk}_${sa}"
#           cvae_file="./subgoal_vae_${env_name}_${dtype}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
          
#           if [ ! -f "$cvae_file" ] && [ ! -d "$ckpt_dir" ]; then
#             echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $ckpt_dir"
#             continue
#           fi
          
#           rl_dir="./logs/tb/${env_name}/reward_True_PrefTransformer/${comment}/${seed}/subgoal_vae_${env_name}_${dtype}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl/${methods}/${weights}"
#           echo "${rl_dir}"
#           if [ -d "$rl_dir" ]; then
#             echo "⚡ SKIP RL (exists): ${rl_dir}"
#             continue
#           fi

#           gpu_rl="$(wait_for_free_gpu)"
#           echo "▶ RL run: env=${env_name}, dtype=${dtype}, seed=${seed} ,sa=${sa} on GPU${gpu_rl}"

#           CUDA_VISIBLE_DEVICES=${gpu_rl} \
#             python robosuite_train_offline.py \
#               --comment "${comment}" \
#               --eval_interval "${eval_interval}" \
#               --env_name "${env_name}" \
#               --robosuite_dataset_type "${dtype}" \
#               --robosuite_dataset_path "${robosuite_dataset_path}" \
#               --config "${rl_config}" \
#               --eval_episodes "${eval_episodes}" \
#               --use_reward_model True \
#               --model_type "${model_type}" \
#               --ckpt_dir "${ckpt_dir}" \
#               --seed "${seed}" \
#               --latent_dim "${ld}" \
#               --hidden_dim "${hd}" \
#               --state_action=${sa} \
#               --topkp ${tk} \
#               --shaping_weight ${weights} \
#               --method ${methods} &

#           pid=$!
#           gpu_pids[$gpu_rl]="${gpu_pids[$gpu_rl]} $pid"
#           gpu_load[$gpu_rl]=$(( ${gpu_load[$gpu_rl]:-0} + 1 ))

#           sleep "$BACKOFF_SEC"
#         done
#       done
#     done
#   done
# done

# wait_all
# echo "🎉 All stages complete."

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
  env_tag="$(echo "${env_name}" | tr '[:upper:]' '[:lower:]')"
  for dtype in "${dtypes[@]}"; do
    comment="${env_tag}-${dtype}"
    for seed in "${seeds[@]}"; do
      for sa in "${state_actions[@]}"; do
        for tk in "${topkps[@]}"; do
          # Stage1 산출물 경로
          ckpt_dir="${log_root}/${env_name}/${model_type}/${comment}/s${seed}/16_32_${tk}_${sa}"
          cvae_file="./subgoal_vae_${env_name}_${dtype}_${seed}_${ld}_${hd}_${tk}_${sa}.pkl"
          
          if [ ! -f "$cvae_file" ] && [ ! -d "$ckpt_dir" ]; then
            echo "⏭️  SKIP RL: CVAE not found: $cvae_file OR $ckpt_dir"
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
"python robosuite_train_offline.py \
              --comment "${comment}" \
              --eval_interval "${eval_interval}" \
              --env_name "${env_name}" \
              --robosuite_dataset_type "${dtype}" \
              --robosuite_dataset_path "${robosuite_dataset_path}" \
              --config "${rl_config}" \
              --eval_episodes "${eval_episodes}" \
              --use_reward_model True \
              --model_type "${model_type}" \
              --ckpt_dir "${ckpt_dir}" \
              --seed "${seed}" \
              --latent_dim "${ld}" \
              --hidden_dim "${hd}" \
              --state_action=${sa} \
              --topkp ${tk} \
              --shaping_weight ${weights} \
              --method ${methods}" )
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

