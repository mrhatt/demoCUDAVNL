// @file       demoKernel.cu
// @author     Charles Hatt <hatt@wisc.edu>
// @date       Apr-25-2015
// Please reference this work if you used it for your research!

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>



//These functions are defined in the demoKernel.cu file
cudaError_t CUSetupTexture(cudaArray* d_I, cudaChannelFormatDesc chDesc);
cudaError_t CUGetCurrentCost(float* h_curxy,float* h_curcost, float* d_curcost);

/////////Textures//////////////////
texture<float, 2, cudaReadModeElementType> texIMG;


/////////Constant data//////////////////
__constant__ float cons_xy[2];


__global__ void kernelComputeCostFunction(float* d_cost)
{

    float x   = cons_xy[0];
    float y   = cons_xy[1];

    //Fetch the value of the image at coordinate x,y
    d_cost[0] = tex2D(texIMG,x-0.5,y-0.5);

}

cudaError_t CUGetCurrentCost(float* h_curxy,float* h_curcost, float* d_curcost)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(cons_xy, h_curxy, 2*sizeof(float));
    if(status != cudaSuccess){
        printf("Error x y params to constant memory\n");
        return status;
    }

    //Call the kernel.  In this simple example, only 1 thread and 1 block are launched
    kernelComputeCostFunction<<<1,1>>>(d_curcost);
    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("Error running the CUDA kernel");
        return status;
    }

    status = cudaMemcpy(h_curcost, d_curcost, 1*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){
        printf("Copying cost back to the host failed\n");
        return status;
    }
    return status;
}


cudaError_t CUSetupTexture(cudaArray* d_I, cudaChannelFormatDesc chDesc)
{
    cudaError_t status;

    texIMG.addressMode[0] = cudaAddressModeClamp;
    texIMG.addressMode[1] = cudaAddressModeClamp;
    texIMG.filterMode     = cudaFilterModeLinear;
    texIMG.normalized     = false;

    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("Error setting texture parameters\n");
        return status;
    }

    status = cudaBindTextureToArray(texIMG, d_I, chDesc);
    if (status != cudaSuccess){
        printf("Binding the texture failed\n");
        return status;
    }

    return status;
}


