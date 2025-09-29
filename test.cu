#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <cstring>

// CUDA 头文件
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA 错误检查宏
#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;\
        exit(1);\
    }\
}

// 简单的矩阵类
class Matrix {
public:
    float* data;
    int rows, cols;
    bool device_allocated;

    Matrix(int r, int c) : rows(r), cols(c), device_allocated(false) {
        data = new float[rows * cols];
    }

    ~Matrix() {
        if (!device_allocated) {
            delete[] data;
        } else {
            cudaError_t err = cudaFree(data);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in free: " << cudaGetErrorString(err) << std::endl;
            }
        }
        data = nullptr;
    }

    // 复制到设备
    void toDevice() {
        if (!device_allocated) {
            float* device_data;
            CUDA_CHECK(cudaMalloc(&device_data, rows * cols * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(device_data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
            // 保存主机数据
            float* host_data = new float[rows * cols];
            std::memcpy(host_data, data, rows * cols * sizeof(float));
            delete[] data;
            data = device_data;
            device_allocated = true;
        }
    }

    // 复制到主机
    void toHost() {
        if (device_allocated) {
            float* host_data = new float[rows * cols];
            CUDA_CHECK(cudaMemcpy(host_data, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(data));
            data = host_data;
            device_allocated = false;
        }
    }

    float& operator()(int i, int j) {
        return data[i * cols + j];
    }
};

// CUDA 核函数：矩阵乘法
__global__ void matrixMultiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// CUDA 核函数：ReLU 激活函数
__global__ void relu(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA 核函数：ReLU 导数
__global__ void reluDerivative(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// CUDA 核函数：Softmax
__global__ void softmax(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = -INFINITY;
        float sum = 0.0f;
        
        // 找最大值（数值稳定性）
        for (int i = 0; i < cols; i++) {
            max_val = fmaxf(max_val, input[row * cols + i]);
        }
        
        // 计算指数和
        for (int i = 0; i < cols; i++) {
            output[row * cols + i] = expf(input[row * cols + i] - max_val);
            sum += output[row * cols + i];
        }
        
        // 归一化
        for (int i = 0; i < cols; i++) {
            output[row * cols + i] /= sum;
        }
    }
}

// CUDA 核函数：元素级加法
__global__ void elementwiseAdd(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA 核函数：元素级减法
__global__ void elementwiseSubtract(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

// CUDA 核函数：元素级乘法
__global__ void elementwiseMultiply(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// CUDA 核函数：标量乘法
__global__ void scalarMultiply(float* A, float scalar, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

// 神经网络层类
class DenseLayer {
private:
    Matrix weights, biases;
    Matrix input, output, z;
    int input_size, output_size;
    
public:
    DenseLayer(int in_size, int out_size) : 
        weights(out_size, in_size), biases(out_size, 1),
        input(1, in_size), output(1, out_size), z(1, out_size),
        input_size(in_size), output_size(out_size) {
        
        // Xavier 初始化权重
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = sqrtf(6.0f / (input_size + output_size));
        std::uniform_real_distribution<float> dis(-limit, limit);
        
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                weights(i, j) = dis(gen);
            }
            biases(i, 0) = 0.1f;
        }
        
        weights.toDevice();
        biases.toDevice();
        // 不在这里预先分配input/output/z到设备，而是在使用时分配
    }
    
    Matrix& forward(Matrix& input_matrix) {
        // 确保输入矩阵在设备上
        if (!input_matrix.device_allocated) {
            input_matrix.toDevice();
        }
        
        dim3 blockDim(16, 16);
        dim3 gridDim((output_size + 15) / 16, (1 + 15) / 16);
        
        // z = input * weights^T + biases
        matrixMultiply<<<gridDim, blockDim>>>(
            input_matrix.data, weights.data, z.data,
            1, input_size, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        elementwiseAdd<<<(output_size + 255) / 256, 256>>>(
            z.data, biases.data, z.data, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU 激活
        relu<<<(output_size + 255) / 256, 256>>>(z.data, output.data, output_size);
        CUDA_CHECK(cudaGetLastError());
        
        return output;
    }
    
    Matrix backward(Matrix& grad_output, float learning_rate) {
        Matrix grad_input(1, input_size);
        Matrix relu_grad(1, output_size);
        grad_input.toDevice();
        relu_grad.toDevice();
        
        // ReLU 导数
        reluDerivative<<<(output_size + 255) / 256, 256>>>(z.data, relu_grad.data, output_size);
        CUDA_CHECK(cudaGetLastError());
        
        // grad_z = grad_output * relu_grad
        Matrix grad_z(1, output_size);
        grad_z.toDevice();
        elementwiseMultiply<<<(output_size + 255) / 256, 256>>>(
            grad_output.data, relu_grad.data, grad_z.data, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 计算权重梯度
        Matrix grad_weights(output_size, input_size);
        grad_weights.toDevice();
        
        dim3 blockDim(16, 16);
        dim3 gridDim((input_size + 15) / 16, (output_size + 15) / 16);
        
        matrixMultiply<<<gridDim, blockDim>>>(
            grad_z.data, input.data, grad_weights.data,
            output_size, 1, input_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 计算输入梯度
        matrixMultiply<<<gridDim, blockDim>>>(
            grad_z.data, weights.data, grad_input.data,
            1, output_size, input_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 更新权重和偏置
        scalarMultiply<<<(output_size * input_size + 255) / 256, 256>>>(
            grad_weights.data, -learning_rate, grad_weights.data, output_size * input_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        elementwiseAdd<<<(output_size * input_size + 255) / 256, 256>>>(
            weights.data, grad_weights.data, weights.data, output_size * input_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        scalarMultiply<<<(output_size + 255) / 256, 256>>>(
            grad_z.data, -learning_rate, grad_z.data, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        elementwiseAdd<<<(output_size + 255) / 256, 256>>>(
            biases.data, grad_z.data, biases.data, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        grad_input.toHost();
        return grad_input;
    }
};

// 简单的神经网络类
class NeuralNetwork {
private:
    std::vector<DenseLayer> layers;
    
public:
    void addLayer(int input_size, int output_size) {
        layers.emplace_back(input_size, output_size);
    }
    
    Matrix forward(Matrix& input) {
        Matrix current = input;
        for (auto& layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
    
    void backward(Matrix& grad_output, float learning_rate) {
        Matrix current_grad = grad_output;
        for (int i = layers.size() - 1; i >= 0; i--) {
            current_grad = layers[i].backward(current_grad, learning_rate);
        }
    }
    
    // 交叉熵损失
    float computeLoss(Matrix& output, Matrix& target) {
        // 确保都在主机上计算损失
        if (output.device_allocated) {
            output.toHost();
        }
        if (target.device_allocated) {
            target.toHost();
        }
        
        float loss = 0.0f;
        for (int i = 0; i < output.cols; i++) {
            loss += -target(0, i) * logf(output(0, i) + 1e-8f);
        }
        return loss;
    }
};

// 训练示例
int main() {
    // 创建神经网络：2个输入 -> 4个隐藏神经元 -> 2个输出
    NeuralNetwork nn;
    nn.addLayer(2, 4);  // 隐藏层
    nn.addLayer(4, 2);  // 输出层
    
    // 简单的 XOR 问题训练数据
    std::vector<std::vector<float>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<float>> targets = {
        {1, 0}, {0, 1}, {0, 1}, {1, 0}  // one-hot 编码
    };
    
    const int epochs = 1000;
    const float learning_rate = 0.1f;
    
    std::cout << "开始训练..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (int i = 0; i < inputs.size(); i++) {
            // 准备输入和目标
            Matrix input(1, 2);
            Matrix target(1, 2);
            
            for (int j = 0; j < 2; j++) {
                input(0, j) = inputs[i][j];
                target(0, j) = targets[i][j];
            }
            
            // 前向传播
            Matrix output = nn.forward(input);
            
            // 计算损失
            total_loss += nn.computeLoss(output, target);
            
            // 计算梯度 (output - target)
            Matrix grad_output(1, 2);
            // 确保output和target都在设备上进行计算
            if (!output.device_allocated) output.toDevice();
            if (!target.device_allocated) target.toDevice();
            if (!grad_output.device_allocated) grad_output.toDevice();
            
            elementwiseSubtract<<<(2 + 255) / 256, 256>>>(
                output.data, target.data, grad_output.data, 2
            );
            CUDA_CHECK(cudaGetLastError());
            
            // 反向传播
            nn.backward(grad_output, learning_rate);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
    
    // 测试训练结果
    std::cout << "\n测试结果:" << std::endl;
    for (int i = 0; i < inputs.size(); i++) {
        Matrix input(1, 2);
        for (int j = 0; j < 2; j++) {
            input(0, j) = inputs[i][j];
        }
        
        Matrix output = nn.forward(input);
        output.toHost();
        
        std::cout << "输入: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> ";
        std::cout << "输出: [" << output(0, 0) << ", " << output(0, 1) << "]" << std::endl;
    }
    
    return 0;
}