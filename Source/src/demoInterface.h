// @file       demoInterface.cxx
// @author     Charles Hatt <hatt@wisc.edu>
// @date       Apr-25-2015
// Please reference this work if you used it for your research!

#ifndef DEMOINTERFACE_H
#define DEMOINTERFACE_H

#define H 512
#define W 512

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

//These functions are defined in the demoKernel.cu file
cudaError_t CUSetupTexture(cudaArray* d_I, cudaChannelFormatDesc chDesc);
cudaError_t CUGetCurrentCost(float* h_curxy, float* h_curcost, float* d_curcost);


class DemoInterface
{

    float       h_xy[2];        //Stores the current point data on the host
    float       h_img[H*W];     //Stores the image on the host
    float       h_cost[1];      //Stores the value of the cost function on the host
    float*      d_cost;         //Stores the value of the cost function on the device (GPU)
    cudaArray*  d_img;          //Stores the image on the device (GPU)

    cudaChannelFormatDesc channelDescImage;     //Used to setup the image as a texture for fast texture reads

    cudaError_t cudaStatus;                     //
    int programStatus;

public:
    DemoInterface();
    ~DemoInterface();
    void StartUpGPU();      //This function clears and resets the GPU
    bool isOK();             //Error checking function, checks GPU status
    void ReadImage();
    void SetParameters(float x, float y); //
    float ComputeCostFunction();

};

#endif
