export TVM_HOME=~/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}


for batch in 1 4 8 16 32 64
    do 
    echo "\n\nTVM Enabled Runs, batch = $batch"
    python tvm_explore.py --batch $batch
    done
