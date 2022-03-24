
for model_name in resnet50 resnet101
    do
    echo "\n\n Using $model_name"
    #python xla_explore.py --only-train --jit-compile-flag --create-model $model_name
    python xla_explore.py --only-train --create-model $model_name --dir-path ~/Optimization/Dataset

    for batch in 1 4 8 16 32 64 128 256
        do 
        echo "\n\nXLA Enabled Runs, batch = $batch"
        python xla_explore.py --jit-compile-flag --infe-batch $batch --create-model $model_name --dir-path ~/Optimization/Dataset
        done

    for batch in 1 4 8 16 32 64 128 256
        do 
        echo "\n\nXLA Disabled Runs, batch = $batch"
        python xla_explore.py --infe-batch $batch --create-model $model_name --dir-path ~/Optimization/Dataset
        done
done