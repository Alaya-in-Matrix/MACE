// MACE means "Multi-objective ACquisition Ensemble"
#pragma once
#include "def.h"
#include "GP.h"
#include "MOO.h"
#include "NLopt_wrapper.h"
#include <Eigen/Dense>
#include <random>
// TODO: use boost::log
class MACE
{
public:
    typedef std::function<Eigen::VectorXd(const Eigen::VectorXd&)> Obj;
    MACE(Obj f, size_t num_spec, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
         std::string log_name = "mace.log");
    ~MACE();
    void initialize(const Eigen::MatrixXd& dbx, const Eigen::MatrixXd& dby);
    void initialize(size_t);
    void initialize(std::string xfile, std::string yfile);

    void set_init_num(size_t);
    void set_max_eval(size_t);
    void set_force_select_hyp(bool);
    void set_tol_no_improvement(size_t);
    void set_eval_fixed(size_t);
    void set_seed(size_t);
    void set_gp_noise_lower_bound(double);
    void set_mo_record(bool);
    void set_mo_gen(size_t);
    void set_mo_np(size_t);
    void set_mo_f(double);
    void set_mo_cr(double);
    void set_batch(size_t);
    void set_use_extreme(bool flag){_use_extreme = flag;}
    void set_noise_free(bool flag){_noise_free = flag;}

    Eigen::VectorXd best_x() const;
    Eigen::VectorXd best_y() const;

    void optimize_one_step(); // one iteration of BO, so that BO could be used as a plugin of other application
    void optimize();          // bayesian optimization

private:
    Obj _func;

protected:
    const Eigen::VectorXd _lb;
    const Eigen::VectorXd _ub;
    const double _scaled_lb;
    const double _scaled_ub;
    Eigen::VectorXd _a;
    Eigen::VectorXd _b;
    std::string _log_name;

    // config
    const size_t _num_spec;
    const size_t _dim;
    double _noise_lvl          = 1e-3;
    size_t _num_init           = 2;
    size_t _max_eval           = 100;
    size_t _batch_size         = 1;
    bool _force_select_hyp     = false;
    size_t _tol_no_improvement = 10;
    size_t _eval_fixed         = 100;  // after _eval_fixed evaluations, only train GP when _tol_no_improvement is reached
    bool _mo_record            = false;
    size_t _mo_gen             = 100;
    size_t _mo_np              = 100;
    double _mo_f               = 0.8;
    double _mo_cr              = 0.8;
    double _seed               = std::random_device{}();
    bool _noise_free           = false;
    bool _use_extreme          = true;  // when selecting points on PF, firstly select the point with extreme value, if batch =
                                        // 1, select the point with best EI, if batch = 2, select points with best EI and best
                                        // LCB

    // inner state
    GP* _gp                    = nullptr;
    size_t _eval_counter       = 0;
    size_t _no_improve_counter = 0;
    bool   _have_feas          = false;
    double _delta              = 0.1;
    double _upsilon            = 0.2;
    double _kappa              = 1.0;
    Eigen::MatrixXd _nlz; // negative log likelihood of the GP model on training data
    Eigen::MatrixXd _hyps;
    Eigen::VectorXd _best_x;
    Eigen::VectorXd _best_y;
    Eigen::MatrixXd _eval_x;
    Eigen::MatrixXd _eval_y;
    std::mt19937_64 _engine = std::mt19937_64(_seed);

    // inner functions
    Eigen::MatrixXd _set_random(size_t num); // random sampling in [_scaled_lb, _scaled_lbub]
    Eigen::MatrixXd _doe(size_t num); // design of experiments via sobol quasi-random
    void _train_GP();

    Eigen::MatrixXd _rescale(const Eigen::MatrixXd& xs) const noexcept; // scale x from [scaled_lb, scaled_ub] to [lb, ub]
    Eigen::MatrixXd _unscale(const Eigen::MatrixXd& xs) const noexcept; // scale x from [lb, ub] to [scaled_lb, scaled_ub]

    double _violation(const Eigen::VectorXd& v) const;
    bool   _better(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const;
    bool   _is_feas(const Eigen::VectorXd& v) const;
    
    void _init_boost_log() const;

    size_t _find_best(const Eigen::MatrixXd& dby) const;
    std::vector<size_t> _seq_idx(size_t) const;
    std::vector<size_t> _pick_from_seq(size_t, size_t);
    Eigen::MatrixXd _slice_matrix(const Eigen::MatrixXd&, const std::vector<size_t>&) const;
    void _moo_config(MOO&) const;
    void _print_log();

    Eigen::MatrixXd _run_func(const Eigen::MatrixXd&);

    // acquisition functions
    double _pf(const Eigen::VectorXd&) const;
    double _pf(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;
    double _log_pf(const Eigen::VectorXd&) const;
    double _log_pf(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;
    double _ei(const Eigen::VectorXd&) const;
    double _ei(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;
    double _log_ei(const Eigen::VectorXd&) const;
    double _log_ei(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;
    double _lcb_improv(const Eigen::VectorXd&) const;  // tau - lcb
    double _lcb_improv(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;
    double _lcb_improv_transf(const Eigen::VectorXd&) const; // transform lcb - tau from [-inf, inf] to [-inf, 0]
    double _lcb_improv_transf(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;
    double _log_lcb_improv_transf(const Eigen::VectorXd&) const;
    double _log_lcb_improv_transf(const Eigen::VectorXd&, Eigen::VectorXd& grad) const;

    
    Eigen::VectorXd _msp(NLopt_wrapper::func f, const Eigen::MatrixXd& sp, nlopt::algorithm=nlopt::LD_SLSQP, size_t max_eval = 100);
    Eigen::MatrixXd _set_anchor();
    Eigen::MatrixXd _select_candidate(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
    void _set_kappa();
};
