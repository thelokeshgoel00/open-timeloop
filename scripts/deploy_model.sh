#!/bin/bash
"""
Deployment script for multilingual-e5-large-instruct model on Qualcomm SM8650 devices.

This script deploys the exported .pte model file to an Android device and runs inference
using the Qualcomm AI Engine Direct backend with Hexagon NPU acceleration.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_DEVICE_DIR="/data/local/tmp/executorch_e5_model"
DEFAULT_MODEL_FILE="e5_model_qnn.pte"
DEFAULT_QNN_SDK_ROOT="/opt/qcom/aistack/qnn"
DEFAULT_EXECUTORCH_ROOT="../executorch"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check device connection
check_device_connection() {
    print_status "Checking device connection..."
    
    if ! command_exists adb; then
        print_error "ADB not found. Please install Android SDK Platform Tools."
        exit 1
    fi
    
    # Check if device is connected
    if ! adb devices | grep -q "device$"; then
        print_error "No Android device found or device is not authorized."
        print_status "Please ensure:"
        echo "  1. USB debugging is enabled"
        echo "  2. Device is connected and authorized"
        echo "  3. Run 'adb devices' to verify connection"
        exit 1
    fi
    
    # Get device info
    local device_model=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
    local device_sdk=$(adb shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')
    local chipset=$(adb shell getprop ro.board.platform 2>/dev/null | tr -d '\r')
    
    print_success "Device connected: $device_model (SDK: $device_sdk, Platform: $chipset)"
    
    # Check for SM8650 (or compatible)
    if [[ $chipset == *"8650"* ]] || [[ $chipset == *"taro"* ]]; then
        print_success "SM8650 (Snapdragon 8 Gen 3) detected - NPU acceleration available"
    elif [[ $chipset == *"8550"* ]]; then
        print_warning "SM8550 detected - Performance may vary from SM8650"
    else
        print_warning "Chipset not explicitly supported, but may still work: $chipset"
    fi
}

# Function to setup device directories
setup_device_directories() {
    local device_dir=$1
    
    print_status "Setting up device directories..."
    
    # Create directory
    adb shell "mkdir -p $device_dir" || {
        print_error "Failed to create directory on device"
        exit 1
    }
    
    # Check write permissions
    adb shell "touch $device_dir/test_write && rm $device_dir/test_write" || {
        print_error "No write permission to $device_dir"
        print_status "Try using: /sdcard/executorch_e5_model/"
        exit 1
    }
    
    print_success "Device directory ready: $device_dir"
}

# Function to push QNN libraries
push_qnn_libraries() {
    local qnn_sdk_root=$1
    local device_dir=$2
    
    print_status "Pushing QNN libraries to device..."
    
    # Check if QNN SDK exists
    if [ ! -d "$qnn_sdk_root" ]; then
        print_error "QNN SDK not found at: $qnn_sdk_root"
        print_status "Please install Qualcomm AI Engine Direct SDK"
        exit 1
    fi
    
    # List of required libraries
    local qnn_libs=(
        "lib/aarch64-android/libQnnHtp.so"
        "lib/aarch64-android/libQnnHtpV69Stub.so"
        "lib/aarch64-android/libQnnHtpV73Stub.so"
        "lib/aarch64-android/libQnnSystem.so"
        "lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so"
        "lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so"
    )
    
    # Push each library
    for lib in "${qnn_libs[@]}"; do
        local lib_path="$qnn_sdk_root/$lib"
        if [ -f "$lib_path" ]; then
            print_status "Pushing $(basename $lib_path)..."
            adb push "$lib_path" "$device_dir/" || {
                print_error "Failed to push $lib_path"
                exit 1
            }
        else
            print_warning "Library not found: $lib_path"
        fi
    done
    
    print_success "QNN libraries pushed successfully"
}

# Function to push ExecutorCH runtime
push_executorch_runtime() {
    local executorch_root=$1
    local device_dir=$2
    
    print_status "Pushing ExecutorCH runtime..."
    
    # Check if ExecutorCH build exists
    local build_dir="$executorch_root/build_android"
    if [ ! -d "$build_dir" ]; then
        print_error "ExecutorCH Android build not found at: $build_dir"
        print_status "Please build ExecutorCH for Android first"
        exit 1
    fi
    
    # Push executor runner
    local executor_runner="$build_dir/examples/qualcomm/qnn_executor_runner"
    if [ -f "$executor_runner" ]; then
        print_status "Pushing executor runner..."
        adb push "$executor_runner" "$device_dir/" || {
            print_error "Failed to push executor runner"
            exit 1
        }
        
        # Make executable
        adb shell "chmod +x $device_dir/qnn_executor_runner"
    else
        print_error "Executor runner not found: $executor_runner"
        exit 1
    fi
    
    # Push backend library
    local backend_lib="$build_dir/lib/libqnn_executorch_backend.so"
    if [ -f "$backend_lib" ]; then
        print_status "Pushing backend library..."
        adb push "$backend_lib" "$device_dir/" || {
            print_error "Failed to push backend library"
            exit 1
        }
    else
        print_error "Backend library not found: $backend_lib"
        exit 1
    fi
    
    print_success "ExecutorCH runtime pushed successfully"
}

# Function to push model file
push_model() {
    local model_file=$1
    local device_dir=$2
    
    print_status "Pushing model file..."
    
    if [ ! -f "$model_file" ]; then
        print_error "Model file not found: $model_file"
        print_status "Please export the model first using export_e5_model.py"
        exit 1
    fi
    
    # Get model size
    local model_size=$(du -h "$model_file" | cut -f1)
    print_status "Model size: $model_size"
    
    # Push model
    adb push "$model_file" "$device_dir/" || {
        print_error "Failed to push model file"
        exit 1
    }
    
    print_success "Model file pushed successfully"
}

# Function to run inference
run_inference() {
    local device_dir=$1
    local model_name=$2
    local iterations=${3:-10}
    
    print_status "Running inference on device..."
    
    # Set up environment and run
    local cmd="cd $device_dir && \
               export LD_LIBRARY_PATH=$device_dir && \
               export ADSP_LIBRARY_PATH=$device_dir && \
               ./qnn_executor_runner --model_path ./$model_name --iterations $iterations"
    
    print_status "Executing: $cmd"
    
    # Run inference and capture output
    local output=$(adb shell "$cmd" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Inference completed successfully!"
        echo "$output"
        
        # Extract performance metrics if available
        local avg_time=$(echo "$output" | grep -o "avg [0-9.]\+ ms" | grep -o "[0-9.]\+")
        if [ ! -z "$avg_time" ]; then
            print_success "Average inference time: ${avg_time}ms"
        fi
        
    else
        print_error "Inference failed!"
        echo "$output"
        return 1
    fi
}

# Function to run benchmark
run_benchmark() {
    local device_dir=$1
    local model_name=$2
    
    print_status "Running benchmark..."
    
    # Warm up
    print_status "Warming up (5 iterations)..."
    run_inference "$device_dir" "$model_name" 5 > /dev/null
    
    # Benchmark runs
    local iterations=(10 50 100)
    
    for iter in "${iterations[@]}"; do
        print_status "Benchmarking with $iter iterations..."
        run_inference "$device_dir" "$model_name" "$iter"
        echo "---"
    done
}

# Function to print device info
print_device_info() {
    print_status "Device Information:"
    echo "Model: $(adb shell getprop ro.product.model | tr -d '\r')"
    echo "Manufacturer: $(adb shell getprop ro.product.manufacturer | tr -d '\r')"
    echo "Android Version: $(adb shell getprop ro.build.version.release | tr -d '\r')"
    echo "SDK Version: $(adb shell getprop ro.build.version.sdk | tr -d '\r')"
    echo "Platform: $(adb shell getprop ro.board.platform | tr -d '\r')"
    echo "CPU ABI: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
    echo "Available RAM: $(adb shell cat /proc/meminfo | grep MemTotal | tr -d '\r')"
}

# Function to cleanup device
cleanup_device() {
    local device_dir=$1
    
    read -p "Remove files from device? (y/N): " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        print_status "Cleaning up device..."
        adb shell "rm -rf $device_dir"
        print_success "Device cleaned up"
    fi
}

# Main function
main() {
    local model_file=""
    local device_dir="$DEFAULT_DEVICE_DIR"
    local qnn_sdk_root="$DEFAULT_QNN_SDK_ROOT"
    local executorch_root="$DEFAULT_EXECUTORCH_ROOT"
    local benchmark=false
    local info_only=false
    local cleanup=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model)
                model_file="$2"
                shift 2
                ;;
            -d|--device-dir)
                device_dir="$2"
                shift 2
                ;;
            --qnn-sdk)
                qnn_sdk_root="$2"
                shift 2
                ;;
            --executorch-root)
                executorch_root="$2"
                shift 2
                ;;
            -b|--benchmark)
                benchmark=true
                shift
                ;;
            --info)
                info_only=true
                shift
                ;;
            --cleanup)
                cleanup=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -m, --model PATH         Model file path (.pte)"
                echo "  -d, --device-dir PATH    Device directory (default: $DEFAULT_DEVICE_DIR)"
                echo "  --qnn-sdk PATH          QNN SDK root path (default: $DEFAULT_QNN_SDK_ROOT)"
                echo "  --executorch-root PATH  ExecutorCH root path (default: $DEFAULT_EXECUTORCH_ROOT)"
                echo "  -b, --benchmark         Run benchmark tests"
                echo "  --info                  Show device info only"
                echo "  --cleanup               Cleanup device files"
                echo "  -h, --help              Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check device connection
    check_device_connection
    
    # Show device info if requested
    if [ "$info_only" = true ]; then
        print_device_info
        exit 0
    fi
    
    # Cleanup if requested
    if [ "$cleanup" = true ]; then
        cleanup_device "$device_dir"
        exit 0
    fi
    
    # Use default model if not specified
    if [ -z "$model_file" ]; then
        model_file="$DEFAULT_MODEL_FILE"
    fi
    
    print_status "Starting deployment..."
    print_status "Model: $model_file"
    print_status "Device directory: $device_dir"
    print_status "QNN SDK: $qnn_sdk_root"
    print_status "ExecutorCH: $executorch_root"
    echo "---"
    
    # Setup device
    setup_device_directories "$device_dir"
    
    # Push libraries and runtime
    push_qnn_libraries "$qnn_sdk_root" "$device_dir"
    push_executorch_runtime "$executorch_root" "$device_dir"
    
    # Push model
    push_model "$model_file" "$device_dir"
    
    # Run inference or benchmark
    local model_name=$(basename "$model_file")
    
    if [ "$benchmark" = true ]; then
        run_benchmark "$device_dir" "$model_name"
    else
        run_inference "$device_dir" "$model_name" 10
    fi
    
    print_success "Deployment and testing completed!"
    print_status "Model is ready for use on the device"
    
    # Offer cleanup
    echo ""
    cleanup_device "$device_dir"
}

# Run main function with all arguments
main "$@" 