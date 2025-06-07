# Quick Start Guide

This guide will help you quickly deploy the multilingual-e5-large-instruct model on your Qualcomm SM8650 (Snapdragon 8 Gen 3) Android device.

## Prerequisites Checklist

- [ ] Ubuntu 20.04 LTS x64 host system
- [ ] Android device with SM8650 chipset
- [ ] USB debugging enabled on Android device
- [ ] At least 16GB RAM on host system
- [ ] 20GB+ free disk space

## Step 1: Environment Setup (15 minutes)

### Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential cmake git wget curl
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y android-tools-adb

# Install Android NDK
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
unzip android-ndk-r25c-linux.zip
sudo mv android-ndk-r25c /opt/
export ANDROID_NDK=/opt/android-ndk-r25c
```

### Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Download Qualcomm SDK (10 minutes)

```bash
# Run the SDK setup script
./scripts/download_qnn_sdk.sh

# Follow the interactive prompts to:
# 1. Download SDK from Qualcomm Developer Portal
# 2. Extract and configure the SDK
# 3. Setup environment variables
```

## Step 3: Build ExecutorCH (20 minutes)

```bash
# Clone ExecutorCH
git clone https://github.com/pytorch/executorch.git
cd executorch

# Set environment variables
export EXECUTORCH_ROOT=$(pwd)
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/<version>
source ~/.qnn_env

# Build for host (AOT compilation)
mkdir build_x86_64 && cd build_x86_64
cmake .. -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=${QNN_SDK_ROOT}
cmake --build . -t "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j8

# Install Python APIs
cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-*-linux-gnu.so \
      $EXECUTORCH_ROOT/backends/qualcomm/python
cp -f backends/qualcomm/PyQnnWrapperAdaptor.cpython-*-linux-gnu.so \
      $EXECUTORCH_ROOT/backends/qualcomm/python

# Build for Android
cd $EXECUTORCH_ROOT
mkdir build_android && cd build_android
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$PWD \
    -DEXECUTORCH_BUILD_SDK=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -B$PWD

cmake --build $PWD -j8 --target install

# Build executor runner
cmake ../examples/qualcomm \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -Bexamples/qualcomm

cmake --build examples/qualcomm -j8
```

## Step 4: Export Model (10 minutes)

```bash
# Return to project directory
cd /path/to/open-timeloop

# Export the model with quantization
python scripts/export_e5_model.py \
    --quantize \
    --validate \
    --output_path e5_model_qnn.pte \
    --device SM8650

# Expected output: e5_model_qnn.pte (~140MB quantized)
```

## Step 5: Deploy to Device (5 minutes)

```bash
# Connect your Android device via USB
# Ensure USB debugging is enabled

# Check device connection
adb devices

# Deploy model to device
./scripts/deploy_model.sh \
    --model e5_model_qnn.pte \
    --benchmark

# Expected output:
# - Model deployment successful
# - Inference time: ~15-25ms per forward pass
# - Throughput: ~40-65 sequences/second
```

## Step 6: Test Inference (2 minutes)

```bash
# Run inference examples
python scripts/inference_example.py \
    --model e5_model_qnn.pte \
    --device \
    --examples similarity multilingual

# Test semantic similarity
python scripts/inference_example.py \
    --model e5_model_qnn.pte \
    --device \
    --examples benchmark
```

## Quick Verification

### Expected Performance Metrics

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| Model Size | ~140MB (quantized) | ~560MB (full precision) |
| Inference Time | 15-25ms | Per forward pass |
| Memory Usage | ~560MB | Peak during inference |
| Throughput | 40-65 seq/sec | Batch size 1 |
| Accuracy | >95% similarity | Compared to original |

### Test Commands

```bash
# Quick device info
./scripts/deploy_model.sh --info

# Verify model accuracy
python scripts/inference_example.py --host --examples similarity
python scripts/inference_example.py --device --examples similarity

# Performance comparison
python scripts/inference_example.py --host --examples benchmark
python scripts/inference_example.py --device --examples benchmark
```

## Troubleshooting

### Common Issues

1. **"QNN SDK not found"**
   ```bash
   source ~/.qnn_env
   echo $QNN_SDK_ROOT
   ```

2. **"Device not detected"**
   ```bash
   adb devices
   # Enable USB debugging in Developer Options
   ```

3. **"Model export failed"**
   ```bash
   # Check EXECUTORCH_ROOT
   echo $EXECUTORCH_ROOT
   # Verify Python APIs are installed
   ls $EXECUTORCH_ROOT/backends/qualcomm/python/
   ```

4. **"Low inference performance"**
   ```bash
   # Check NPU utilization
   adb shell "cat /proc/cpuinfo | grep -i hexagon"
   # Verify libraries are loaded correctly
   ./scripts/deploy_model.sh --info
   ```

### Performance Optimization

1. **Reduce Memory Usage**
   - Use smaller batch sizes
   - Enable gradient checkpointing
   - Use INT8 quantization

2. **Improve Throughput**
   - Batch multiple sequences
   - Use dynamic shapes efficiently
   - Optimize sequence lengths

3. **Debug Performance**
   ```bash
   # Profile model execution
   python scripts/inference_example.py --device --examples benchmark
   
   # Check operator delegation
   # Look for "Hexagon" in logs during inference
   ```

## Next Steps

1. **Integrate into Android App**
   - See Android integration examples in README.md
   - Use JNI bindings for native integration

2. **Optimize for Production**
   - Fine-tune quantization parameters
   - Implement dynamic batching
   - Add error handling and fallbacks

3. **Scale Deployment**
   - Test on multiple device models
   - Implement A/B testing framework
   - Monitor performance metrics

## Support

- **Documentation**: See full README.md for detailed information
- **Issues**: Check troubleshooting section in README.md
- **Performance**: Run benchmark scripts for detailed metrics
- **Community**: PyTorch ExecutorCH GitHub discussions

## Estimated Total Time

- **First-time setup**: ~60 minutes
- **Subsequent deployments**: ~5 minutes
- **Model updates**: ~15 minutes

Success! Your multilingual E5 model is now running on Qualcomm SM8650 with Hexagon NPU acceleration. 