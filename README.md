# pyHIPIFY

一个将NVIDIA CUDA源代码自动转换为AMD HIP源代码的Python工具，提取自PyTorch仓库。该工具能够帮助开发人员轻松地将CUDA应用程序移植到AMD GPU平台，支持将CUDA源文件（.cu、.cuh、.cpp）自动转换为等效的HIP源文件。

## 功能特点

- **自动API转换**：将CUDA API调用转换为等效的HIP API
- **多格式支持**：支持.cu、.cuh、.cpp、.h等多种文件格式
- **批量处理**：支持同时转换多个源文件
- **进度可视化**：提供详细的转换进度和统计信息
- **并行处理**：支持多线程并行处理，加快大型项目转换速度
- **多种处理模式**：支持不同的文件处理策略（完整复制、仅转换可转换文件、浅层目录处理）
- **易集成**：提供命令行界面，便于集成到现有构建系统中

## 安装方法

在当前目录下运行以下命令进行安装：

```bash
python setup.py install
```

## 使用方法

安装完成后，可通过命令行使用pyHIPIFY：

```bash
pyHIPIFY [选项]
```

### 命令行选项

- `--project-directory`：指定CUDA项目根目录路径
- `--output-directory`：指定转换后HIP文件的输出目录
- `--include`：指定需要转换的目录列表（可选）
- `--ignores`：指定需要排除的目录列表（可选）
- `--jobs`：设置并行处理的工作线程数（0表示自动检测）
- `--stats`：显示转换过程的详细统计信息
- `--no-progress`：禁用进度显示
- `--out-of-place-only`：仅执行外部位置转换（创建新文件，不修改原文件）
- `--file-processing-mode`：文件处理模式，可选值包括：
  - `all`：复制所有文件并转换可转换的CUDA文件（默认模式）
  - `convertible-only`：仅处理和转换可转换的文件，其他文件会被忽略
  - `shallow`：仅处理当前目录（不递归处理子目录）中的文件

### 使用示例

**转换整个CUDA项目：**
```bash
pyHIPIFY --project-directory /path/to/cuda/project \
         --output-directory /path/to/hip/project \
         --stats
```

**仅转换指定目录：**
```bash
pyHIPIFY --project-directory /path/to/cuda/project \
         --include src/cuda kernels \
         --stats
```

**使用并行处理加速转换：**
```bash
pyHIPIFY --project-directory /path/to/cuda/project \
         --jobs 4 \
         --stats
```

**只转换可转换的文件（不复制其他文件）：**
```bash
pyHIPIFY --project-directory /path/to/cuda/project \
         --output-directory /path/to/hip/project \
         --file-processing-mode convertible-only \
         --stats
```

**仅处理当前目录（不递归子目录）：**
```bash
pyHIPIFY --project-directory /path/to/cuda/project \
         --output-directory /path/to/hip/project \
         --file-processing-mode shallow \
         --stats
```

## 工作原理

pyHIPIFY通过预定义的CUDA到HIP API映射表进行代码转换，核心映射规则定义在[cuda_to_hip_mappings.py](pyHIPIFY/cuda_to_hip_mappings.py)文件中，涵盖：

- CUDA运行时API函数
- CUDA驱动程序API函数  
- CUDA数据类型和常量
- 数学函数
- 内核启动语法

工具会解析源文件，识别CUDA特定语法结构，并根据映射表将其替换为等效的HIP实现。

## 支持的转换类型

- CUDA运行时API → HIP运行时API
- CUDA驱动程序API → HIP驱动程序API  
- CUDA BLAS函数 → HIP BLAS函数
- CUDA数学函数 → HIP数学函数
- CUDA内核启动语法 → HIP内核启动语法
- CUDA头文件引入 → HIP头文件引入
- CUDA类型声明 → HIP类型声明

## 注意事项

- **代码兼容性**：复杂的模板代码可能无法完全自动转换
- **功能覆盖**：部分CUDA特有功能可能缺乏直接的HIP等效实现
- **手动验证**：转换后建议进行人工代码审查和测试验证
- **API覆盖**：目前尚未覆盖所有CUDA API到HIP的映射

## 反馈与支持

目前该项目暂未在GitHub上开放，如有问题或建议请联系相关维护人员。