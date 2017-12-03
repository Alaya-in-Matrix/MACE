/*
  The MATLAB wrapper for the test function suite fcnsuite.dll or fcnsuite.so
  Thomas Philip Runarsson (email: tpr@hi.is) 
  Time-stamp: "2005-11-16 15:47:32 tpr"
*/  

/* OS specific function calls for shared objects and dynamic linking */ 
#ifdef WINDOWS
#include <windows.h>
#include <process.h>
typedef void (WINAPI * PPROC) (double *, double *, double *, double *, int, int, int, int);

#define LIBHANDLE HANDLE
#define GetProcedure GetProcAddress
#define CloseDynalink FreeLibrary
#else
#include <dlfcn.h>
#include <pthread.h>
typedef void (*PPROC) (double *, double *, double *, double *, int, int, int, int);

#define LIBHANDLE void *
#define GetProcedure dlsym
#define CloseDynalink dlclose
#endif
  
#include "mex.h"
  
#ifdef __STDC__
void mexFunction (int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) 
#else
void  mexFunction (nlhs, plhs, nrhs, prhs) 
int     nlhs, nrhs;
mxArray *plhs[];
const mxArray *prhs[];
#endif
{
  int i, j, m, n, ng, nh, size;
  double *H, *G, *F, *X, *tmpptr;
  char *fcnname;
  PPROC pfcn;
  LIBHANDLE hLibrary;
  if ((nrhs < 4) || (nlhs < 3))
    {
      mexPrintf ("usage: [f, g, h] = mlbsuite(x, ng, nh, 'function_name');\n");
      mexErrMsgTxt ("example: [f, g, h] = mlbsuite([14.095 0.84296]', 2, 0, 'g06');");
    }
  n = mxGetM (prhs[0]);
  m = mxGetN (prhs[0]);
  X = mxGetPr (prhs[0]);
  size = mxGetNumberOfElements (prhs[3]) + 1;
  fcnname = mxCalloc (size, sizeof (char));
  if (mxGetString (prhs[3], fcnname, size) != 0)
    mexErrMsgTxt ("Could not convert string data.");
#ifdef WINDOWS
  hLibrary = LoadLibrary ("fcnsuite.dll");
#else
  hLibrary = dlopen ("./fcnsuite.so", RTLD_NOW);
#endif
  if (NULL == hLibrary)
    mexErrMsgTxt ("could not load fcnsuite.dll (win32) or fcnsuite.so (linux) file");
  pfcn = (PPROC) GetProcedure (hLibrary, fcnname);
  if (NULL == pfcn)
    {
      mexPrintf ("procedure %s not found in library file!", fcnname);
      mexErrMsgTxt ("failed to load procedure");
    }
  plhs[0] = mxCreateDoubleMatrix (1, m, mxREAL);
  F = mxGetPr (plhs[0]);
  tmpptr = mxGetPr (prhs[1]);
  ng = (int) tmpptr[0];
  plhs[1] = mxCreateDoubleMatrix (ng, m, mxREAL);
  G = mxGetPr (plhs[1]);
  tmpptr = mxGetPr (prhs[2]);
  nh = (int) tmpptr[0];
  plhs[2] = mxCreateDoubleMatrix (nh, m, mxREAL);
  H = mxGetPr (plhs[2]);
  for (i = 0; i < m; i++)
    {
      pfcn (&X[i * n], &F[i], &G[i * ng], &H[i * nh], n, 1, ng, nh);
    }
  CloseDynalink (hLibrary);
  mxFree (fcnname);
}
