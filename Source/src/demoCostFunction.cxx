// @file       demoCostFunction.cxx
// @author     Charles Hatt <hatt@wisc.edu>
// @date       Apr-25-2015
// Please reference this work if you used it for your research!

#include "demoCostFunction.h"
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_cost_function.h>

double DemoCostFunction::f(vnl_vector<double> const &x)
{

    float cost = 0.0;
    if (DemoObject->isOK())
    {
        curx = (float)x(0);
        cury = (float)x(1);
        demoObjectOK=true;
        DemoObject->SetParameters(curx,cury);
        cost = DemoObject->ComputeCostFunction();

        vector<float> tmp;
        tmp.push_back(curx);
        tmp.push_back(cury);
        tmp.push_back(cost);
        history.push_back(tmp);

        fevals++;
    }
    else
    {
        fevals=-1;
        demoObjectOK=false;
    }
    return (double)cost;

}

void DemoCostFunction::gradf(vnl_vector<double> const &x, vnl_vector<double> &dx)
{
    //Use Finite Difference gradient.  This is needed for LBFGS if you want to use that optimizer.
    fdgradf(x, dx);
}

void DemoCostFunction::SetDemoObject(DemoInterface* inptr)
{
    DemoObject = inptr;
}

bool DemoCostFunction::isOK()
{
    return demoObjectOK;
}

float DemoCostFunction::GetCurrentX()
{
    return curx;
}

float DemoCostFunction::GetCurrentY()
{
    return cury;
}

int DemoCostFunction::GetFunctionEvaluations()
{
    return fevals;
}

void DemoCostFunction::WriteOptimizationHistory(std::string filename)
{

    FILE *fp;
    fp = fopen(filename.c_str(), "w");
    if(fp==NULL){
        printf("Error opening: %s\n",filename.c_str());
        return;
    }

    //The output file is a binary file and its easiest to write this as a
    //1D array of floats.  The length should be the
    float* tmp = (float*)malloc(3*history.size()*sizeof(float));
    int k=0;
    for(int j=0; j<3;j++)
    {
        for(int i=0; i<history.size();i++)
        {
            tmp[k] = history[i][j];
            k++;
        }
    }
    fwrite(tmp, sizeof(float), 3*history.size(), fp);
    fclose(fp);
    free(tmp);
    return;

}





