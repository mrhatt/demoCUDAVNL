// @file       demoCostFunction.h
// @author     Charles Hatt <hatt@wisc.edu>
// @date       Apr-25-2015
// Please reference this work if you used it for your research!

#ifndef DEMOCOSTFUNCTION_H
#define DEMOCOSTFUNCTION_H

#include <vector>
#include "demoInterface.h"
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_cost_function.h>

using namespace std;


class DemoCostFunction : public vnl_cost_function
{


    DemoInterface* DemoObject;   //Provides an interface to the GPU
    float curx;         //Stores the current x coordinate
    float cury;         //Stores the current x coordinate
    int numParams;      //Stores the number of optimization parameters
    bool demoObjectOK;  //Stores the status of the DemoObject, to check for CUDA errors
    int fevals;         //Stores the current number of cost function evaluations
    vector< vector<float> > history;    //Stores the x,y coordaitne and cost function value for each cost function evaluation

    public:

        //Constructor
        DemoCostFunction(const int NumVars) : vnl_cost_function(NumVars)
        {
            numParams = (int)NumVars;
            demoObjectOK=true;
            fevals=0;
        }

        //Function declarations
        double f(vnl_vector<double> const &x);
        void gradf(vnl_vector<double> const &x, vnl_vector<double> &dx);
        void SetDemoObject(DemoInterface* inptr);
        bool isOK();
        float GetCurrentX();
        float GetCurrentY();
        int GetFunctionEvaluations();
        void WriteOptimizationHistory(string filename);


};

#endif
