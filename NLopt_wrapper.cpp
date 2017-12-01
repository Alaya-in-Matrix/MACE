#include "util.h"
#include "NLopt_wrapper.h"
using namespace std;
using namespace Eigen;
NLopt_wrapper::NLopt_wrapper(nlopt::algorithm a, size_t d, double lb, double ub)
    : _opt(nlopt::opt(a, d))
{
    _opt.set_lower_bounds(lb);
    _opt.set_upper_bounds(ub);
}
void NLopt_wrapper::set_min_objective(NLopt_wrapper::func f)
{
    _f = f;
    _opt.set_min_objective([](const vector<double>& x, vector<double>& grad, void* data) -> double {
        NLopt_wrapper* nlopt_ptr = reinterpret_cast<NLopt_wrapper*>(data);
        VectorXd vgrad;
        double val = nlopt_ptr->_f(convert(x), vgrad);
        grad       = convert(vgrad);
        return val;
    }, this);
}
void NLopt_wrapper::set_maxeval(size_t v){_opt.set_maxeval(v);}
void NLopt_wrapper::set_ftol_abs(double v){_opt.set_ftol_abs(v);}
void NLopt_wrapper::set_ftol_rel(double v){_opt.set_ftol_rel(v);}
void NLopt_wrapper::set_xtol_abs(double v){_opt.set_xtol_abs(v);}
void NLopt_wrapper::set_xtol_rel(double v){_opt.set_xtol_rel(v);}
void NLopt_wrapper::optimize(Eigen::VectorXd& sp, double& val)
{
    vector<double> stlsp = convert(sp);
    try
    {
        _opt.optimize(stlsp, val);
        sp = convert(stlsp);
    }
    catch(...)
    {
        VectorXd fake_g;
        sp  = convert(stlsp);
        val = _f(sp, fake_g);
        throw;
    }
}
