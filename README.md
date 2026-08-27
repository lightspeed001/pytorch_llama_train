# pytorch_llama_train
## Training TinyLlama-1.1B on Colab and k8s on T4 GPU
#### Minimal example using Hugging Face _transformers_ and _accelerate_ for memory efficiency


### Key Concepts :robot:
  
  >  * Free tiers (Kaggle/Colab). Suitable for small models (<1.5B params) with optimizations (LoRA, quantisation, gradient accumulation).
  >  * GPU clusters: Necessary for larger models, longer training or multi-GPU setups.
  >  * Alternatives: Consider cloud credits or academic grants for longer projects.
  
### Troubleshooting :warning:
1.  __Out of Memory (OOM) Errors__:
  >  * Reduce ``` per_device_train_batch_size ``` (eg. to 2)
  >  * Increase ``` gradient_accumulation_steps ``` (eg. to 8)
  >  * Use ``` max_length=250 ``` instead of 512.
2. __Checkpoint Saving__
  >  * If Colab disconnects, save the checkpoints frequently (``` save_steps=500 ```).
3. __Dataset Too Large__:
  >  * Use ``` streaming=True ``` in ``` load_dataset ``` to load data on-the-fly:

    ```python
            dataset = load_dataset("tiny_shakespeare", split="train", streaming=True)

     ```

### Building and Pushing the Image :hammer_and_wrench:
1. Build the Docker image:
  ```bash
        docker build -t ${DOCKER_REGISTRY}/tinyllama-trainer:latest .
        
   ```
2. Push to a registry:
  ```bash
        docker push ${DOCKER_REGISTRY}/tinyllama-trainer:latest
        
   ```
3. Deploy to k3s:
  ```bash
        kubectl apply -f pod.yaml
        
   ```

### Monitoring :eyeglasses:
  >  * Check logs:
    ```bash
          kubectl logs -f tinyllama-trainer
          
     ```
     
  >  * Describe the pod for events:
   ```bash
        kubectl describe pod tinyllama-trainer
        
    ```
