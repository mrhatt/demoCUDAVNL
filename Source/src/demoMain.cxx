// @file       demoMain.cxx
// @author     Charles Hatt <hatt@wisc.edu>
// @date       Apr-25-2015
// Please reference this work if you used it for your research!


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "demoCostFunction.h"
#include <vnl/algo/vnl_amoeba.h> //Nelder-mead
#include <vnl/algo/vnl_powell.h> //Powell

void Powell(DemoCostFunction &cf, vnl_vector<double> XY0, double stepsize);
void Amoeba(DemoCostFunction &cf, vnl_vector<double> XY0);


int main(int argc, char* args[] ){

    //Handle the arguments.  This program takes a single argument, an integer
    //representing the optimizer to use
    if(argc < 4)
    {
        printf("Proper usage: demo Optimizer(1=Powell 2=Amoeba) x0 y0]\n");
        return -1;
    }

    int optimizer = 1;
    float ix=50;
    float iy=50;

    //The stringstream class is really nice for reading in text and converting it
    //to other datatypes
    std::stringstream ss;
    ss << args[1]; ss >> optimizer;
    ss.str(""); ss.clear();
    ss << args[2]; ss >> ix;
    ss.str(""); ss.clear();
    ss << args[3]; ss >> iy;
    ss.str(""); ss.clear();

    //Create the demo object used
    DemoInterface DemoObject;

    //Check and see if the object was initialized without errors
    if(!DemoObject.isOK()){
        return -1;
    }

    //Create the cost function object. The "2" is used because there are two parameters (x-y coords)
    DemoCostFunction CF(2);
    CF.SetDemoObject(&DemoObject);

    //This specifies the initial starting point for optimization
    //The "2" here is because we need a vector of size 2 to store the initial values
    vnl_vector<double> initXY(2);
    initXY(0) = (double)ix;
    initXY(1) = (double)iy;

    //These functions are for timing, which is useful for profiling how fast your code is
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////Timing Start

    if(optimizer == 1)
    {
        printf("Using Powell Optimizer\n");
        Powell(CF,initXY,0.1);  //Stepsize is 0.1 pixels

    }
    else
    {
        printf("Using Amoeba Simplex Optimizer\n");
        Amoeba(CF,initXY);
    }

    //////////////////////////Timing Stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if(!CF.isOK())
    {
       printf("Something went wrong. Exiting\n");
       return 1;
    }

    //Print out useful info
    printf("Time (ms): %f \n", milliseconds);
    printf("Total function evaluations : %d \n", CF.GetFunctionEvaluations());
    printf("Initial X,Y coordinate: %f,%f \n", initXY(0),initXY(1));
    printf("Final   X,Y coordinate: %f,%f \n", CF.GetCurrentX(),CF.GetCurrentY());

    //Write all of the optimization steps to "history.bin"
    std::string outfile = "history.bin";
    CF.WriteOptimizationHistory(outfile);
    printf("Output file: %s\n",outfile.c_str());
    return 0;

}

void Amoeba(DemoCostFunction &cf, vnl_vector<double> XY0)
{
    vnl_amoeba Minimizer(cf);
    //These parameters control when the optimizer will assume it has converged
    Minimizer.set_f_tolerance(0.000001);
    Minimizer.set_x_tolerance(0.000001);
    Minimizer.minimize(XY0);
}

void Powell(DemoCostFunction &cf, vnl_vector<double> XY0, double stepsize)
{
    //Notice that the powell optimizer needs the address of the cost function object.
    vnl_powell Minimizer(&cf);
    //These parameters control when the optimizer will assume it has converged
    Minimizer.set_initial_step(stepsize);
    Minimizer.set_f_tolerance(0.000001);
    Minimizer.set_x_tolerance(0.000001);
    Minimizer.minimize(XY0);
}


