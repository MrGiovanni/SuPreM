# Train on AbdomenAtlas 1.0 using multiple GPUs
# for backbone in segresnet unet swinunetr; do sbatch --error=logs/$backbone.1.0.multigpu.out --output=logs/$backbone.1.0.multigpu.out shell_scripts/train_abdomenatlas1.0_multigpu.sh $backbone; done

# Train on AbdomenAtlas 1.0 using single GPU
# for backbone in segresnet unet swinunetr; do sbatch --error=logs/$backbone.1.0.singlegpu.out --output=logs/$backbone.1.0.singlegpu.out shell_scripts/train_abdomenatlas1.0_singlegpu.sh $backbone; done

# Train on AbdomenAtlas 1.1 using multiple GPUs
# for backbone in segresnet unet swinunetr; do sbatch --error=logs/$backbone.1.1.multigpu.out --output=logs/$backbone.1.1.multigpu.out shell_scripts/train_abdomenatlas1.1_multigpu.sh $backbone; done

# Train on AbdomenAtlas 1.1 using single GPU
for backbone in segresnet unet swinunetr; do sbatch --error=logs/$backbone.1.1.singlegpu.out --output=logs/$backbone.1.1.singlegpu.out shell_scripts/train_abdomenatlas1.1_singlegpu.sh $backbone; done