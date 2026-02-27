## Pre Trained Model Results

26th Feb 2026

### Device = GPU
 - **Device Name**: Tesla T4
 - **Driver Version**: 580.82.07 
 - **CUDA Version**: 13.0
 - **Total Memory**: 14912MB

#### Dataset
- **Dataset**: CIFAR-10
- **Size**: Transformed(32x32 -> 224x224)

#### Model
- **Model**: MobileNetV2
- **Pretrained**: True
- **Input**: 1280
- **Output**: 10
- **Loss**: Cross Entropy Loss
- **Optimizer**: Adam

#### Training Results
- **Batch size**: 64
- **Epoch**: 5

```
Epoch 1, Loss: 0.52 
Epoch 2, Loss: 0.34 
Epoch 3, Loss: 0.29
Epoch 4, Loss: 0.25
Epoch 5, Loss: 0.22
```
- **Time**: 23m 58.6s

#### Tests Results
- **Test Accuracy**: 89.84%
- **Time**: 26.2s

#### Measurements
- **Size**: 8MB
- **Total params**: 2236682M 
- **Trainable params**: 2236682M 
- **FLOPs**: 312926016
- **Latency(Dummy Data)**: 6.99ms

