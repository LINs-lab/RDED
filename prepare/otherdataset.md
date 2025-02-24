## Prepare Other Datasets

To ensure consistency across datasets, we standardized the folder naming convention for all datasets except TinyImageNet. Each class folder is named using a five-digit extension of its class ID. For example, if `class_id=0` corresponds to the first class in ImageNet-1K, "tench" (乌鳢), the folder containing images for this class is named `00000`. This naming convention allows for seamless integration with our pre-trained models and code implementation, ensuring that users can accurately execute our RDED.
