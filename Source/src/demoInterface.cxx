// @file       demoInterface.cxx
// @author     Charles Hatt <hatt@wisc.edu>
// @date       Apr-25-2015
// Please reference this work if you used it for your research!

#include "demoInterface.h"

DemoInterface::DemoInterface()
{

    //This is the constructor.  We take care of reading all the data
    //in, as well as memory allocation and transfers here, but recognize that
    //your program may have to do this elsewhere

    //Everything is OK for now
    programStatus=0;

    StartUpGPU();

    if(!isOK())
    {
        printf("Problem starting up GPU \n");
        return;
    }

    //This object is used for setting up the image as a texture.
    channelDescImage = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaStatus = cudaMallocArray(&d_img, &channelDescImage,W,H);
    if(cudaStatus!=cudaSuccess)
    {
        printf("Allocating image failed\n");
        return;
    }

    cudaStatus = cudaMalloc((void **) &d_cost, 1*sizeof(float));
    if(cudaStatus!=cudaSuccess){
        printf("Allocating the cost function parameters failed\n");
        return;
    }

    ReadImage();

    if(!isOK())
    {

        printf("Problem reading image \n");
        return;
    }

    cudaStatus = CUSetupTexture(d_img, channelDescImage);
    if(!isOK())
    {
        printf("Problem setting up texture\n");
        return;
    }

    cudaStatus = cudaMemcpyToArray(d_img, 0,0, h_img, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus!=cudaSuccess)
    {
        printf("Copying image to array\n");
        return;
    }
}

DemoInterface::~DemoInterface()
{
    cudaFreeArray(d_img);
    cudaFree(d_cost);
}


void DemoInterface::StartUpGPU()
{
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("InitGPU: Setting the Device to 0 failed!\n");
		return;
    }

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        printf("InitGPU: Resetting the Device to 0 failed!\n");
		return;
    }
    if (cudaStatus == cudaSuccess)
    {
        printf("InitGPU: GPU Started up OK!\n");
    }
}

bool DemoInterface::isOK(){
    if(cudaStatus == cudaSuccess & programStatus==0)
    {
        return true;
    }

    else if(cudaStatus != cudaSuccess)
    {
        printf("CUDA Error code: %d\n",cudaStatus);
        return false;
    }
    else
    {
        printf("Program Error code: %d\n",programStatus);
        return false;
    }
}

void DemoInterface::ReadImage()
{
    std::string filename;
    filename = "img.bin";
	FILE *fp;
    fp = fopen(filename.c_str(), "r");
	if(fp==NULL){
        printf("Error reading: %s\n",filename.c_str());
        programStatus = -1;
		return;
	}
    size_t r = fread(h_img, sizeof(float), W*H, fp);
	fclose(fp);
	return;
}


void DemoInterface::SetParameters(float x, float y)
{
    h_xy[0] = x;
    h_xy[1] = y;
}

float DemoInterface::ComputeCostFunction()
{
    cudaStatus =  CUGetCurrentCost(h_xy,h_cost,d_cost);
    return h_cost[0];
}

