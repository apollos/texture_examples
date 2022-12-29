#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <helper_cuda.h>

typedef float4 typ;

texture<typ, 1> texref;
texture<typ, 2> texref2;
texture<typ, 3> texref3;
int RUNS = 1000;
cudaEvent_t     start, stop;
float   elapsedTime;

/* some utilities */


__device__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x+b.x, a.y+b.y);
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

/* 1D linear memory */

__global__ void touch1Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;

    for(; i < M-2; i += N) {
        outPtr[i] = devPtr[i] + devPtr[i+1] + devPtr[i+2] + devPtr[i+3];
    }
}

__global__ void touch1Dlinear2(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ typ s[1024];

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;
    s[i%1024] = devPtr[i];
    s[(i+1)%1024] = devPtr[i+1];
    s[(i+2)%1024] = devPtr[i+2];
    s[(i+3)%1024] = devPtr[i+3];
    __syncthreads();

    for(; i < M-4; i += N) {
        outPtr[i] = s[i%1024] + s[(i+1)%1024] + s[(i+2)%1024] + s[(i+3)%1024];
    }
}

void time1Dlinear()
{
    void* devPtr;
    void* outPtr;
    long M = 1024*1024*10;
    int blocks = 4096;
    int threads = 64;

    checkCudaErrors( cudaMalloc(&devPtr, M*sizeof(typ)) );
    checkCudaErrors( cudaMalloc(&outPtr, M*sizeof(typ)) );

    checkCudaErrors( cudaEventRecord( start, 0 ) );

    for(int i = 0; i < RUNS; i ++){
        touch1Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
    
    
    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaFree(devPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("linear 1D: %.1f ms\n", elapsedTime);
}

void time1Dlinearshare()
{
    void* devPtr;
    void* outPtr;
    long M = 1024*1024*10;
    int blocks = 4096;
    int threads = 64;

    checkCudaErrors( cudaMalloc(&devPtr, M*sizeof(typ)) );
    checkCudaErrors( cudaMalloc(&outPtr, M*sizeof(typ)) );

    checkCudaErrors( cudaEventRecord( start, 0 ) );

    for(int i = 0; i < RUNS; i ++){
        touch1Dlinear2<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
    
    
    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaFree(devPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("linear share memory 1D: %.1f ms\n", elapsedTime);
}

/* 1D texture memory */

__global__ void touch1Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    typ* outPtr = (typ*) outPtr_;

    for(; i < M-2; i += N) {
        outPtr[i] = (
            tex1Dfetch(texref, i) +
            tex1Dfetch(texref, i+1) +
            tex1Dfetch(texref, i+2) + 
            tex1Dfetch(texref, i+3)
        );
    }
}

void time1Dtexture()
{
    void* refPtr;
    void* outPtr;
    long M = 1024*1024*10;
    int blocks = 4096;
    int threads = 64;

    checkCudaErrors( cudaMalloc(&refPtr, M*sizeof(typ)) );
    checkCudaErrors( cudaMalloc(&outPtr, M*sizeof(typ)) );
    checkCudaErrors( cudaBindTexture(NULL, texref, refPtr, M) );
    
    checkCudaErrors( cudaEventRecord( start, 0 ) );
    for(int i = 0; i < RUNS; i++){
        touch1Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaUnbindTexture(texref) );    
    checkCudaErrors( cudaFree(refPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("texture 1D: %.1f ms\n", elapsedTime);
}

/* 2D linear memory */

__global__ void touch2Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            outPtr[ix*M+iy] = (
                devPtr[ix*M+iy] +
                devPtr[(ix+1)*M+iy] +
                devPtr[ix*M+iy+1] + 
                devPtr[(ix+1)*M+iy+1]
            );
        }
    }
}

void time2Dlinear()
{
    void* devPtr;
    void* outPtr;
    long M = 4096;
    dim3 blocks(32,32);
    dim3 threads(16,16);

    checkCudaErrors( cudaMalloc(&devPtr, M*M*sizeof(typ)) );
    checkCudaErrors( cudaMalloc(&outPtr, M*M*sizeof(typ)) );

    checkCudaErrors( cudaEventRecord( start, 0 ) );

    for(int i = 0; i < RUNS; i ++){
        touch2Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
    
    
    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaFree(devPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("linear 2D: %.1f ms\n", elapsedTime);
}

/* 2D texture memory */

__global__ void touch2Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;

    typ* outPtr = (typ*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            outPtr[ix*M+iy] = (
                tex2D(texref2, ix, iy) +
                tex2D(texref2, ix+1, iy) +
                tex2D(texref2, ix, iy+1) +
                tex2D(texref2, ix+1, iy+1)
            );
        }
    }
}

void time2Dtexture()
{
    long M = 4096;
    dim3 blocks(32, 32);
    dim3 threads(16,16);

    void* outPtr;
    checkCudaErrors( cudaMalloc(&outPtr, M*M*sizeof(typ)) );

    cudaArray *refPtr;
    checkCudaErrors( cudaMallocArray(&refPtr, &texref2.channelDesc, M, M) );
    checkCudaErrors( cudaBindTextureToArray(texref2, refPtr) );
    
    checkCudaErrors( cudaEventRecord( start, 0 ) );
    for(int i = 0; i < RUNS; i++){
        touch2Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaUnbindTexture(texref) );    
    checkCudaErrors( cudaFreeArray(refPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("texture 2D: %.1f ms\n", elapsedTime);
}

/* 3D linear memory */

__global__ void touch3Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            for(; iz < M-1; iz += N) {
                outPtr[ix*M*M+iy*M+iz] = (
                    devPtr[ix*M*M+iy*M+iz] +
                    devPtr[ix*M*M+iy*M+(iz+1)] +
                    devPtr[ix*M*M+(iy+1)*M+iz] +
                    devPtr[(ix+1)*M*M+iy*M+iz]
                );
            }
        }
    }
}

void time3Dlinear()
{
    void* devPtr;
    void* outPtr;
    long M = 128;
    dim3 blocks(32,32,32);
    dim3 threads(4, 4, 4);

    checkCudaErrors( cudaMalloc(&devPtr, M*M*M*sizeof(typ)) );
    checkCudaErrors( cudaMalloc(&outPtr, M*M*M*sizeof(typ)) );

    checkCudaErrors( cudaEventRecord( start, 0 ) );

    for(int i = 0; i < RUNS; i ++){
        touch3Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
    
    
    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaFree(devPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("linear 2D: %.1f ms\n", elapsedTime);
}

/* 3D texture memory */

__global__ void touch3Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;

    typ* outPtr = (typ*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            for(; iz < M-1; iz += N) {
                outPtr[ix*M*M+iy*M+iz] = (
                    tex3D(texref3, ix, iy, iz) +                    
                    tex3D(texref3, ix, iy, iz+1) +
                    tex3D(texref3, ix, iy+1, iz) +
                    tex3D(texref3, ix+1, iy, iz)
                );
            }
        }
    }
}

void time3Dtexture()
{
    unsigned long M = 128;
    dim3 blocks(32,32,32);
    dim3 threads(4,4,4);

    void* outPtr;
    checkCudaErrors( cudaMalloc(&outPtr, M*M*M*sizeof(typ)) );
    
    cudaArray* refPtr;
    checkCudaErrors( cudaMalloc3DArray(&refPtr, &texref2.channelDesc, {M, M, M}) );
    checkCudaErrors( cudaBindTextureToArray(texref3, refPtr) );
    
    checkCudaErrors( cudaEventRecord( start, 0 ) );
    for(int i = 0; i < RUNS; i++){
        touch3Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaEventRecord( stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    checkCudaErrors( cudaUnbindTexture(texref) );    
    checkCudaErrors( cudaFreeArray(refPtr) );
    checkCudaErrors( cudaFree(outPtr) );
    
    printf("texture 3D: %.1f ms\n", elapsedTime);
}


int main(int argc, char *argv[]) {
    checkCudaErrors(cudaEventCreate( &start ));
    checkCudaErrors(cudaEventCreate( &stop ));
    time1Dlinear();
    time1Dlinearshare();
    time1Dtexture();
    time2Dlinear();
    time2Dtexture();
    time3Dlinear();
    time3Dtexture();
    checkCudaErrors( cudaEventDestroy( start ) );
    checkCudaErrors( cudaEventDestroy( stop ) );
    return 0;
}