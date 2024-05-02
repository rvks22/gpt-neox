docker run \
    --runtime=nvidia \
    --rm -it \
    -u 1008 \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e TRANSFORMERS_CACHE=/workspace/.cache \
    -e PYTHONPATH=/best-download \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --mount type=bind,src=$PWD,dst=/gpt-neox \
    -v $(pwd):/workspace \
    -v /code/pythia/pythia_pile_idxmaps:/pythia_pile_idxmaps \
    -v /code/pythia/gpt-neox/20B_tokenizer.json:/workspace/20B_tokenizer.json \
    -v /code/pythia/.cache:/.cache \
    pythia:latest bash
# python deepy.py train.py pythia-160m-deduped.yml  2>&1 | tee output.txt

    # -v /code/pythia_tests/pythia/pythia_pile_idxmaps:/fsx/pile_deduped \
    # -v /code/pythia_tests/20B_tokenizer.json:/fsx/pile/20B_tokenizer.json \
    # -v /code/pythia_tests/pytorch_cache:/.cache \

