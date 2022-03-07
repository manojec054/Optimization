
for model_name in resnet50 vgg16 ownmodel resnet101
    do
    echo "\n\n Using $model_name"
    python tvm_explore.py --create-model $model_name --infe-batch 64 --only-train

    for batch in 1 4 8 16 32 64 128 256
        do 
        echo "\n\nTVM Enabled Runs, batch = $batch"
        python tvm_explore.py --create-model $model_name --infe-batch $batch --tvm
        done

    for batch in 1 4 8 16 32 64 128 256
        do 
        echo "\n\nTVM Disabled Runs, batch = $batch"
        python tvm_explore.py --create-model $model_name --infe-batch $batch --tvm
        done
done