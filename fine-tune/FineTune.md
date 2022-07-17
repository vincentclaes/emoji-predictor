# Finte Tune with CLIP

    pip install huggingface-hub
    huggingface-cli login
    huggingface-cli repo create emoji-predictor
    
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
    sudo yum install git-lfs -y
    git lfs install

    mkdir hf-repo && cd $_
    git clone https://huggingface.co/vincentclaes/emoji-predictor
    cd emoji-predictor
    git lfs track "*tfevents*"
    
    export MODEL_DIR="./hf-repo/emoji-predictor
    ln -s /home/sagemaker-user/transformers/examples/research_projects/jax-projects/hybrid_clip/run_hybrid_clip.py run_hybrid_clip.py