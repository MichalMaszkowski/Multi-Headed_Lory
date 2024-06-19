# Things to do:
- prepare plain datasets, dataloaders (Wiki) for training and evaluation
- write positional encoding function
- create a transformer block for language moddeling (causal mask) with placeholder for (MoE) FFNs
- implement simple MoE (done - see plain_MOE notebook)
- implement Lory
- implement MH-MoE
- implement MH-Lory
- write code for controlling randomness (setting seeds)
- write training loop
- find appropriate hyperparameters and train each of the above (IMPORTANT: save model weights)
- evaluate each of the above
- measure performance
- print nice graphs visualizing training loss against epochs, eval loss, performance

# Nice to have (do it only when all above are done or about to be done):
- implement similar-documents stitching technique thah is used in Lory for training dataset
- train all on the above dataset and see whether training or validation improves
