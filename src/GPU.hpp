#ifndef GPU_HPP
#define GPU_HPP

// 前向声明
struct GPUObjectsImpl;

// 使用Pimpl惯用法的GPU对象结构体
struct GPUObjects {
  GPUObjectsImpl* impl;
};

// 初始化GPU对象并返回
GPUObjects GPU_init(int* _ia, int* _ja, double* _a, double* _B, double* _Geta_X, int _op_plus_mp);

// 使用GPU求解X
void GPU_solveX(int& _Icount, int* ia, int* ja, double* a, double* B, double* geta_X, const GPUObjects& _obj);

// 释放GPU资源
void GPU_release(int* _ia, int* _ja, double* _a, double* _B, double* _Geta_X, GPUObjects& _obj);

#endif