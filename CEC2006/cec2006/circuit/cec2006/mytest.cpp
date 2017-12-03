#include <dlfcn.h>
#include <pthread.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

typedef void (*PPROC) (double *, double *, double *, double *, int, int, int, int);
#define LIBHANDLE void *
#define GetProcedure dlsym
#define CloseDynalink dlclose

using namespace std;
void g02(); // provided g02 is not correct, re-write if according to the technical report
double* read_config(size_t dim);
int main(int argc, char** argv)
{
    if(argc != 5)
        return EXIT_FAILURE;
    char* prob = argv[1];
    const size_t dim = atoi(argv[2]);
    const size_t ng  = atoi(argv[3]);
    const size_t nh  = atoi(argv[4]);
    cout << setprecision(18);

    if(string(prob) == "g02")
    {
        g02();
        return EXIT_SUCCESS;
    }

    LIBHANDLE hLibrary = dlopen ("./fcnsuite.so", RTLD_NOW);
    PPROC pfcn         = (PPROC) GetProcedure (hLibrary, prob); /* g01 to g24 is valid */

    double* x   = read_config(dim);
    double  f   = numeric_limits<double>::infinity();
    double* g   = new double[ng];
    double* h   = new double[nh];
    pfcn (x, &f, g, h, dim, 1, ng, nh);
    cout << f << ' ';
    for(size_t i = 0; i < ng; ++i)
        cout << g[i] << ' ';
    cout << endl;

    delete[] x;
    delete[] g;
    delete[] h;
    CloseDynalink (hLibrary);
    return EXIT_SUCCESS;
}

double* read_config(size_t dim)
{
    ifstream ifile;
    ifile.open("./param");
    if((!ifile.is_open()) || ifile.fail())
    {
        cerr << "param file can not open" << endl;
        exit(EXIT_FAILURE);
    }

    double param;
    vector<double> tmp_params;
    while(ifile >> param)
    {
        tmp_params.push_back(param);
    }
    if(tmp_params.size() != dim)
    {
        cerr << "Invalid dimension" << endl;
        exit(EXIT_FAILURE);
    }
    
    double* xs = new double[dim];
    for(size_t i = 0; i < dim; ++i)
        xs[i] = tmp_params[i];
    return xs;
}
void g02()
{
    const size_t dim = 20;
    double* x = read_config(dim);

    double part1 = 0;
    double part2 = 2;
    double part3 = 0;
    double prob  = 1;
    double sum   = 0;
    for(size_t i = 0; i < dim; ++i)
    {
        prob  *= x[i];
        sum   += x[i];
        part1 += pow(cos(x[i]), 4);
        part2 *= pow(cos(x[i]), 2);
        part3 += (i+1) * pow(x[i], 2);
    }

    const double f  = -1 * abs((part1 - part2) / sqrt(part3));
    const double g1 = 0.75 - prob;
    const double g2 = sum - 7.5 * dim;
    cout << f << " " << g1 << " " << g2 << endl;

    delete[] x;
}
