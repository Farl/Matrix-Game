```
python3 -m venv venv_matrix_game/
source venv_matrix_game/bin/activate
```

這個只需要做一次
```
pip install -r requirements_macos_m2.txt
```

```
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python3 inference.py \
--config_path configs/inference_yaml/inference_universal.yaml \
--checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
--img_path demo_images/deemo2/statue.jpg \
--output_folder outputs \
--num_output_frames 30 \
--seed 42 \
--pretrained_model_path Matrix-Game-2.0
```
```
TORCH_COMPILE_DISABLE=1 python3 inference_streaming.py \
--config_path configs/inference_yaml/inference_universal.yaml \
--checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
--pretrained_model_path Matrix-Game-2.0 \
--output_folder outputs \
--max_num_output_frames 3
```
```
demo_images/deemo2/statue.jpg
demo_images/universal/0000.png
demo_images/universal/0001.png
```