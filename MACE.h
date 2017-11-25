// MACE means "Multi-objective ACquisition Ensemble"
#pragma once
#include "def.h"
#include "GP.h"
#include <Eigen/Dense>
#include <random>
// TODO: use boost::log
class MACE
{
public:
    typedef std::function<Eigen::VectorXd(const Eigen::VectorXd&)> Obj;
    MACE(Obj f, size_t num_spec, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
         std::string log_name = "weibo.log");
    ~MACE();
    void initialize(const Eigen::MatrixXd& dbx, const Eigen::MatrixXd& dby);
    void initialize(size_t);
    void initialize(std::string xfile, std::string yfile);

    void set_max_eval(size_t);
    void set_force_select_hyp(bool);
    void set_tol_no_improvement(size_t);
    void set_seed(size_t);
    void set_gp_noise_lower_bound(double);

    void optimize_one_step(); // one iteration of BO, so that BO could be used as a plugin of other application
    void optimize();          // bayesian optimization

    Eigen::VectorXd best_x() const;
    Eigen::VectorXd best_y() const;

protected:
    // config
    const size_t _num_spec;
    const size_t _dim;
    const size_t _max_eval;
    const size_t _force_select_hyp;
    const size_t _tol_no_improvement;

    // inner state
    const Eigen::VectorXd _lb;
    const Eigen::VectorXd _ub;
    const double _scaled_lb;
    const double _scaled_ub;
    Eigen::VectorXd _a;
    Eigen::VectorXd _b;

    std::mt19937_64 _engein = std::mt19937_64(std::random_device{}());
    GP* _gp                 = nullptr;
    size_t _eval_counter    = 0;
    bool   _have_feas       = false;
    Eigen::VectorXd _best_x;
    Eigen::VectorXd _best_y;
    Eigen::MatrixXd _dbx;
    Eigen::VectorXd _dby;

    // inner functions
    double _run_func(const Eigen::VectorXd&);

private:
    Obj _func;
};
