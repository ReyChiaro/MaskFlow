# QwenImage MaskFlow：Poisson 开/关的完整生成轨迹对比

开启 `pipeline.velocity_visualization_dir` 后，`eval_step` 完整运行两条独立演化的生成轨迹：Poisson OFF 与 Poisson ON。源图、mask、文本、CFG、scheduler、初始噪声完全相同，唯一实验变量是推理中的 Poisson 修正。训练过程不变，两边使用同一组模型权重；这不是训练时开/关 Poisson 的消融。

## 比较方式

只编码一次条件、采样一次噪声，然后复制为两份初始状态。每个 timestep 都进行：

1. OFF 在自身状态 `x_off` 上执行各 CFG 分支，得到 `v_off = CFG(model(x_off, sigma))`。
2. ON 在自身状态 `x_on` 上重新执行各 CFG 分支；若 sigma 落入 `poisson_steps`，将干净 latent 估计经 Poisson 修正后换算为速度，得到 `v_on`。
3. 两边分别调用同一 scheduler 的无状态 `step`，包括各自的背景投影，更新各自的状态。
4. 从第一步到最后一步均执行上述过程。选择可视化步骤仅控制大数组的导出，不会跳过未选步骤的任意一条轨迹。

默认线性路径中，单次 Poisson 修正仍使用 `x0 = x_sigma - sigma*v`。但本图比较的是 **`v_on(x_on, sigma) - v_off(x_off, sigma)`**，包含此前累积的状态差异与当前 Poisson 修正，不能再解释成同一状态下的局部修正量。

ON 只在配置的 `poisson_steps` 区间生效，OFF 始终禁用。某一步不在区间内，两条轨迹仍可能因更早的 Poisson 更新而不同。起点 latent 完全一致；第一步使用的速度则可以因 Poisson 修正而不同。`enable_poisson_infer` 决定常规 `output` 返回哪条轨迹；只要诊断开启且记录步骤非空，就会同时运行 ON/OFF 两条。

模型需像常规推理一样处于 eval 模式；项目的 inference/evaluate 入口已设置。两条轨迹在每一步依次执行，保持完整 batch 的形状，适用于当前无状态的 `MaskFlowScheduler`。两条轨迹不会重新拼成一个更大的 batch。

## 可视化

三栏为 **Poisson OFF / Poisson ON / Velocity difference**：

- 左右两条速度场叠在同一张淡化源图上，便于空间对齐。源图是固定参考，不是当前解码结果。
- 箭头是解包后的 latent 通道向量的二维投影，**不是光流或物体移动方向**。Qwen 自身 `_unpack_latents` 恢复 `[B,C,H_latent,W_latent]`，不把 packed token 的前两个特征当成图像轴。
- 两条轨迹的所有选中步骤共同拟合不去均值的 SVD 投影。零速度仍为零箭头，基底符号固定，所有步骤共用基底、箭头尺度、色标。
- 箭头长度表示投影范数，颜色表示全通道速度 RMS；第三栏是 `sqrt(mean_c((v_on-v_off)^2))`。投影保留能量写在图注中。
- 实线为软 mask 的 0.5 等值线，虚线为 Poisson 支持区域 `mask>0` 的边界。显示尺度按所记录步骤的统一 p99 固定并截断极端值，原始值完整保存。不同样本各自拟合，不能直接跨样本比较箭头方向或色彩。

速度记录在 CFG 和可选 Poisson 之后、scheduler 更新之前，表示 `dx/dsigma`。去噪时 sigma 递减，积分增量与速度符号相反。

页面下方补充：

- **逐步速度差异曲线**：全部推理步骤的全局 RMS，不受字段导出步数限制。
- **累计状态差异曲线**：每步积分及背景投影之后的 `RMS(x_on_next-x_off_next)`。它反映实际轨迹的分离，而不是调度器随后可能覆盖的背景速度误差。
- **最终生成图**：两条轨迹各自完整解码，按相同 `enable_pixel_blend` 设置处理后并排展示。像素 blend 会影响最后图像中可见的差异，配置同时写入元数据。

## 使用

在原来的单图推理或 `evaluate.py` 命令中追加：

```bash
pipeline.velocity_visualization_dir=outputs/velocity \
'pipeline.velocity_visualization_steps=[1,10,20,30,40,50]'
```

显式步号必须在本次推理步数内；省略 steps 时自动选至多 6 个均匀步骤。例如：

```bash
python inference.py \
  input.source=/path/to/source.png \
  input.mask=/path/to/mask.png \
  'input.prompt=Replace the marked object with a vase' \
  pipeline.velocity_visualization_dir=outputs/velocity
```

`generate()` 同样走这个入口，直接设置 pipeline 的上述属性即可。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `velocity_visualization_dir` | `null` | 关闭；设置路径开启完整轨迹对比 |
| `velocity_visualization_steps` | `null` | 默认选至多 6 步；显式列表从 1 开始；`[]` 同时关闭记录和额外轨迹 |
| `velocity_visualization_max_samples` | `1` | 每批导出的前几个样本；两条轨迹仍各自运行完整 batch |
| `velocity_visualization_grid` | `24` | 长边上至多约 24 个箭头 |

开启后额外执行一条完整去噪轨迹和一次最终 VAE 解码，去噪前向次数为原来的两倍。条件编码复用；两个 latent 状态保留在设备上。所选步骤的速度与状态复制到 CPU，推理结束后绘图和压缩保存。默认关闭时仅运行原来选定的一条轨迹。

返回值保持 `mask`、`output`；开启比较时额外返回完整 batch 的 `output_poisson_off` 和 `output_poisson_on`。常规 `output` 按 `enable_poisson_infer` 选择，便于保持现有保存流程。

## 导出与数值定义

每个 batch/rank 使用独立目录，每个样本保存：

- `index.html`：离线时间步滑块、播放按钮、完整轨迹曲线及最终两张生成图。
- `step-XXXX.png`：指定步骤的三栏速度图。
- `trajectory.png`：全部步骤的速度差异和累计状态差异曲线。
- `final-poisson-off.png` / `final-poisson-on.png`：最终输出图像，按常规 PNG 保存时截断到 `[0,1]`。
- `fields.npz`：`velocity_off/on`、`state_off/on`、`projected_off/on`、`rms_off/on`、`rms_delta`、`state_delta_rms`、`basis`、`mask`、`steps`、`sigmas`。
- `metadata.json`：schema 版本 2，比较类型、参数、投影/尺度、所选步骤区域统计以及**全部步骤**的 `history`。

NPZ 的 `state_off/on` 是当前 sigma 下、更新前的状态。`history.next_state_delta_rms` 是更新后的状态差异，对应该行 `next_sigma`，最后一行对应 sigma=0 的最终 latent。`history.velocity_delta_rms` 是当前 sigma 下两条速度差异的全局 RMS。

区域统计为编辑核心 `m>=0.95`、软边界 `0<m<0.95`、背景 `m==0` 内逐点通道 RMS 的均值；空区域保存 null。区域内 RMS 均值与 history 的全局 RMS 定义不同。全黑、全白 mask 均可导出。

旧版本的 `before/after` 数组代表同一状态的局部修正，新版以 `velocity_off/on` 显式区分完整轨迹语义。旧导出不会被覆盖，也不应与新版混用。

多卡各 rank 单独保存；FSDP 共享 batch 时会有重复诊断。`max_samples` 只限制每批导出的样本数量，不限制整次评估的样本总数或额外推理计算量。

## 无模型排版示例与验证

```bash
python -m scripts.demo_velocity_visualization --output outputs/velocity-demo
python -m unittest discover -s tests -p 'test_velocity_visualization.py' -v
```

示例使用依赖当前状态的解析速度、现有 Poisson 求解器和 MaskFlowScheduler，完整演化两条合成 latent 轨迹，仅用于排版与语义验证，不代表 Qwen 推理或真实生成图。

CPU 测试将双轨迹推理与各自独立执行的 ON/OFF 推理逐步比较，验证每步状态、速度、更新状态以及最终输出一致；同时覆盖软 mask、非线性背景轨迹、四分支 CFG、批量 seed、pixel blend、稀疏记录、Poisson 区间之外的累计差异，以及未生效时两条轨迹完全一致。
