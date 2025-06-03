# sweep over small_model, main_model, thinking_mode, prompt_mode for thinking mode 2
for small_model in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"
do
    for main_model in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"
    do
        for thinking_mode in 2
        do
            for prompt_mode in "seconds" "relative_size" "tokens"
            do
                sbatch eval_script.slurm $thinking_mode "$main_model" "$small_model" "$prompt_mode"
            done
        done
    done
done

# sweep over main_model for thinking mode 0 and 1
for main_model in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"
do
    for thinking_mode in 0 1
    do
        for prompt_mode in "seconds" "relative_size" "tokens"
        do
            sbatch eval_script.slurm $thinking_mode "$main_model" "none" "$prompt_mode"
        done
    done
done

