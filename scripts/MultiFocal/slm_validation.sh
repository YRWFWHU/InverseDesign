#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/../.."


echo "第一步：生成多焦点目标图像"
python target_generator/MultiFocal.py \
  --image_size 1024 \
  --start_depth 180000 \
  --end_depth 220000 \
  --step 400 \
  --unit um \
  --focus_start_depth 190000 \
  --focus_end_depth 210000 \
  --focus_radius 3 \
  --output_dir slm_validation

echo "第二步：运行相位优化"
python main.py \
  --target_dir target/MultiFocal/slm_validation \
  --propagation_funtion default \
  --phase_profile_mask rect \
  --dx 6.4e-6 \
  --wavelength 520e-9 \
  --device cuda:1 \
  --iterations 300 \
  --learning_rate 1e-1

echo "完成！结果保存在 target/MultiFocal/slm_validation 目录"
