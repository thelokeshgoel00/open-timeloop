# Multilingual E5 Large Instruct on Qualcomm SM8650

This project provides a complete guide and implementation for porting the [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) text embedding model to run on Qualcomm Android mobile phones with the SM8650 (Snapdragon 8 Gen 3) chipset, leveraging Hexagon processor acceleration through PyTorch ExecutorCH and Qualcomm AI Engine Direct.

## Overview

The multilingual-e5-large-instruct model is a 560M parameter XLM-RoBERTa-based embedding model with:
- 24 layers and 1024 embedding dimensions
- Support for 94+ languages
- Instruction-tuned for various text tasks
- Maximum sequence length of 512 tokens

## Architecture Compatibility

The SM8650 (Snapdragon 8 Gen 3) is explicitly supported by PyTorch ExecutorCH's Qualcomm AI Engine Direct backend for Hexagon NPU acceleration.

## Prerequisites

### Hardware Requirements
- Android smartphone with SM8650 (Snapdragon 8 Gen 3) chipset
- ADB connectivity enabled
- At least 4GB available storage

### Host System Requirements
- Ubuntu 20.04 LTS x64 (required for Qualcomm SDK)
- Python 3.8+ 
- GCC 9.4+
- Android NDK 25c+
- 16GB+ RAM for model compilation

### Software Dependencies
- [Qualcomm AI Engine Direct SDK v2.12.0+](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- PyTorch ExecutorCH
- Hugging Face Transformers
- ONNX Runtime (optional)

## Development Approaches

### Option 1: GitHub Actions + Local Deployment (Recommended)

This hybrid approach leverages CI/CD for automated model building while handling device deployment locally:

**✅ What GitHub Actions Handles:**
- Model download and quantization (automated)
- PyTorch ExecutorCH compilation (cached)
- Cross-compilation for Android ARM64
- Artifact generation and validation
- Performance metrics reporting

**🔧 What Requires Local Setup:**
- Physical device connection and deployment
- Qualcomm SDK manual download (license agreement)
- On-device inference and benchmarking

**Quick Setup:**
```bash
# 1. Fork this repository and enable GitHub Actions

# 2. Setup local environment for device deployment
./scripts/setup_local_dev.sh

# 3. Trigger model build (or push code to trigger automatically)
# GitHub Actions will build and provide downloadable artifacts

# 4. Download pre-built model and deploy
gh run download --pattern 'e5-model-deployment-*'
./deploy_local.sh
```

### Option 2: Full Local Development

For complete control or offline development:

## Installation Guide

### 1. Setup Environment

```bash
# Set environment variables
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/<version>
export ANDROID_NDK=/path/to/android-ndk-r25c
export EXECUTORCH_ROOT=/path/to/executorch

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
```

### 2. Install PyTorch ExecutorCH

```bash
# Clone ExecutorCH repository
git clone https://github.com/pytorch/executorch.git
cd executorch

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 3. Build ExecutorCH with Qualcomm Backend

```bash
cd $EXECUTORCH_ROOT

# Setup build directory
mkdir build_x86_64 && cd build_x86_64

# Configure CMake
cmake .. \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=${QNN_SDK_ROOT}

# Build AOT components
cmake --build . -t "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j8

# Install Python APIs
cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so \
      $EXECUTORCH_ROOT/backends/qualcomm/python
cp -f backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so \
      $EXECUTORCH_ROOT/backends/qualcomm/python
```

### 4. Build Android Runtime

```bash
cd $EXECUTORCH_ROOT
mkdir build_android && cd build_android

# Build runtime for Android
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$PWD \
    -DEXECUTORCH_BUILD_SDK=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -B$PWD

cmake --build $PWD -j16 --target install

# Build executor runner
cmake ../examples/qualcomm \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -Bexamples/qualcomm

cmake --build examples/qualcomm -j16
```

## Model Conversion Process

### 1. Export Model from Hugging Face

See `scripts/export_e5_model.py` for the complete implementation.

### 2. Key Considerations

#### Memory Optimization
- The model requires ~2.2GB for full precision inference
- Use quantization (INT8) to reduce memory footprint to ~560MB
- Enable dynamic shapes for variable sequence lengths

#### Instruction Format
The model requires specific instruction formatting:
```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
```

#### Model Architecture Adaptations
- Ensure RoPE (Rotary Position Embedding) compatibility
- Handle attention mask properly for variable length sequences
- Optimize pooling operations for Hexagon NPU

### 3. Quantization Strategy

```python
# Apply quantization for mobile deployment
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

qconfig_mapping = get_default_qconfig_mapping("qnnpack")
model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)
model_quantized = convert_fx(model_prepared)
```

## Performance Optimizations

### Expected Performance on SM8650
Based on similar models and the Qualcomm VIT benchmarks:
- **Inference Time**: ~15-25ms per forward pass (quantized)
- **Memory Usage**: ~560MB (INT8), ~2.2GB (FP16)
- **Throughput**: ~40-65 sequences/second
- **Power Efficiency**: Significant improvement over CPU-only execution

### Optimization Techniques
1. **Operator Fusion**: Combine consecutive operations
2. **Memory Layout**: Optimize for Hexagon NPU access patterns
3. **Batch Processing**: Process multiple sequences simultaneously
4. **Layer Pruning**: Remove less critical attention heads if needed

## Deployment Guide

### 1. Prepare Device

```bash
# Create device directory
DEVICE_DIR=/data/local/tmp/executorch_e5_model/
adb shell "mkdir -p ${DEVICE_DIR}"

# Push QNN libraries
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}
```

### 2. Deploy Model

```bash
# Push model and runtime
adb push ./e5_model_qnn.pte ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build_android/examples/qualcomm/qnn_executor_runner ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build_android/lib/libqnn_executorch_backend.so ${DEVICE_DIR}

# Run inference
adb shell "cd ${DEVICE_DIR} \
           && export LD_LIBRARY_PATH=${DEVICE_DIR} \
           && export ADSP_LIBRARY_PATH=${DEVICE_DIR} \
           && ./qnn_executor_runner --model_path ./e5_model_qnn.pte"
```

## Usage Examples

### Basic Text Embedding

```python
# See scripts/inference_example.py for complete implementation
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = get_detailed_instruct(task, 'How to optimize deep learning models for mobile?')
embedding = model.encode(query)
```

### Multilingual Support

```python
# Chinese example
task_cn = 'Given a web search query, retrieve relevant passages that answer the query'
query_cn = get_detailed_instruct(task_cn, '如何优化移动端深度学习模型？')
embedding_cn = model.encode(query_cn)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use quantization
2. **Library Loading**: Ensure correct ADSP_LIBRARY_PATH
3. **Model Compatibility**: Check operator support in QNN backend
4. **Performance Issues**: Verify NPU utilization and operator delegation

### Performance Tuning

1. **Profile Model**: Use ExecutorCH profiling tools
2. **Optimize Operators**: Check for unsupported operations falling back to CPU
3. **Memory Layout**: Ensure optimal tensor layouts for Hexagon NPU
4. **Quantization**: Fine-tune quantization parameters

## Validation

### Accuracy Testing

```python
# Compare outputs between original and quantized models
original_output = original_model.encode(test_inputs)
mobile_output = mobile_model_inference(test_inputs)
cosine_similarity = F.cosine_similarity(original_output, mobile_output)
```

### Performance Benchmarking

```python
# Measure inference time
import time
start_time = time.time()
embeddings = model.encode(batch_inputs)
inference_time = time.time() - start_time
```

## Integration with Android Apps

### Java/Kotlin Integration

```kotlin
// Example Android integration
class EmbeddingModel {
    private external fun initModel(modelPath: String): Long
    private external fun runInference(modelPtr: Long, input: FloatArray): FloatArray
    
    companion object {
        init {
            System.loadLibrary("executorch_jni")
        }
    }
}
```

## Future Enhancements

1. **Dynamic Batching**: Support variable batch sizes
2. **Streaming Inference**: Process long documents in chunks
3. **Multi-Model Support**: Deploy multiple embedding models
4. **Advanced Quantization**: Explore mixed-precision and pruning

## References

- [PyTorch ExecutorCH Documentation](https://docs.pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html)
- [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [Multilingual E5 Technical Report](https://arxiv.org/abs/2402.05672)
- [Qualcomm VIT Model Performance](https://huggingface.co/qualcomm/VIT)

## License

This project follows the MIT license of the original multilingual-e5-large-instruct model.
