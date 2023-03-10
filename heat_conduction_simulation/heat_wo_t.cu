#include "cuda.h"
#include "../common/cpu_anim2.h"
#include "../common/book.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f



__global__ void blend_kernel_linear( float *in,
                              float *constSrc) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)   left++;
    if (x == DIM-1) right--; 

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)   top += DIM;
    if (y == DIM-1) bottom -= DIM;

    float   t, l, c, r, b;

    if (constSrc[offset] != 0)
        in[offset] = constSrc[offset];
    __syncthreads();

    t = in[top];
    l = in[left];
    c = in[offset];
    r = in[right];
    b = in[bottom];

    
    in[offset] = c + SPEED * (t + b + r + l - 4 * c);
    __syncthreads();
}


// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    CPUAnimBitmap  *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
    FILE           *fp;
    int             imageSize;
};

void anim_gpu_linear( DataBlock *d, int times ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    for (int i = 0; i < 90*times; i++)
    {
        blend_kernel_linear<<<blocks,threads>>>( d->dev_inSrc, d->dev_constSrc );
        ++d->frames;
    }

    float_to_color<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc );
    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );    
    d->totalTime += elapsedTime;    
    printf( "Average Time per frame:  %3.1f ms\n",
            elapsedTime/times);
    /*
    if (ticks == 100){
        fwrite(bitmap->pixels, d->imageSize , 1, d->fp );
    }
    */
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d, bool texture_flag) {
   
    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_constSrc ) );    
    fclose(d->fp);
}

void test_linear_method(DataBlock *data){
    data->fp = fopen( "linear_result_file.dmp" , "w" );
    HANDLE_ERROR( cudaMalloc( (void**)&data->dev_inSrc,
                              data->imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data->dev_outSrc,
                              data->imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data->dev_constSrc,
                              data->imageSize ) );
    // intialize the constant data
    float *temp = (float*)malloc( data->imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data->dev_constSrc, temp,
                              data->imageSize,
                              cudaMemcpyHostToDevice ) );    
    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data->dev_inSrc, temp,
                              data->imageSize,
                              cudaMemcpyHostToDevice ) );
    free( temp );
    anim_gpu_linear(data, 500);
    fwrite(data->bitmap->pixels, data->imageSize , 1, data->fp); //write calculate result    
    anim_exit(data, false);
}

int main( void ) {
    DataBlock   data;    
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    data.imageSize = bitmap.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               data.imageSize ) );    

    test_linear_method(&data);
    HANDLE_ERROR( cudaEventDestroy( data.start ) );
    HANDLE_ERROR( cudaEventDestroy( data.stop ) );
}
