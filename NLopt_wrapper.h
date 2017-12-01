#pragma once
#include "Eigen/Dense"
#include "nlopt.hpp"
class NLopt_wrapper
{
public:
    typedef std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)> func;
    NLopt_wrapper(nlopt::algorithm, size_t dim, double lb, double ub);

    void set_min_objective(func);
    void set_maxeval(size_t max_eval);
    void set_ftol_abs(double v);
    void set_ftol_rel(double v);
    void set_xtol_abs(double v);
    void set_xtol_rel(double v);
    void optimize(Eigen::VectorXd& sp, double& val);

protected:
    nlopt::opt _opt;
    func _f;
};
