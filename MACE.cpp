#include "MACE.h"
#include "MOO.h"
#include "util.h"
#include "NLopt_wrapper.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <gsl/gsl_qrng.h>
#include <omp.h>
#include <chrono>
#include <set>
#include <fstream>
using namespace std;
using namespace std::chrono;
using namespace Eigen;
MACE::MACE(Obj f, size_t num_spec, const VectorXd& lb, const VectorXd& ub, string log_name)
    : _func(f),
      _lb(lb),
      _ub(ub),
      _scaled_lb(-25), 
      _scaled_ub(25), 
      _a((_ub - _lb) / (_scaled_ub - _scaled_lb)), 
      _b(0.5 * (_ub + _lb)), 
      _log_name(log_name), 
      _num_spec(num_spec), 
      _dim(lb.size()),
      _max_eval(500),
      _tol_no_improvement(10),
      _gp(nullptr),
      _eval_counter(0), 
      _have_feas(false), 
      _best_x(VectorXd::Constant(_dim, 1, INF)), 
      _best_y(RowVectorXd::Constant(1, _num_spec, INF)), 
      _eval_x(MatrixXd(_dim, 0)), 
      _eval_y(MatrixXd(_num_spec, 0))
{
    assert(_scaled_lb   < _scaled_ub);
    assert((_lb.array() < _ub.array()).all());
    _init_boost_log();
    BOOST_LOG_TRIVIAL(info) << "MACE Created";
}
MatrixXd MACE::_run_func(const MatrixXd& xs)
{
    bool no_improve = true;
    const auto t1            = chrono::high_resolution_clock::now();
    const size_t num_pnts = xs.cols();
    const MatrixXd scaled_xs = _rescale(xs);
    MatrixXd ys(_num_spec, num_pnts);
#pragma omp parallel for
    for(size_t i = 0; i < num_pnts; ++i)
    {
        ys.col(i) = _func(scaled_xs.col(i));
    }

    for(size_t i = 0; i < num_pnts; ++i)
    {
        if(_better(ys.col(i), _best_y))
        {
            _best_x    = scaled_xs.col(i);
            _best_y    = ys.col(i);
            no_improve = false;
        }
        if(_is_feas(ys.col(i)))
            _have_feas = true;
    }
    if(no_improve)
        ++_no_improve_counter;
    else
        _no_improve_counter = 0;
    const auto t2       = chrono::high_resolution_clock::now();
    const double t_eval = static_cast<double>(chrono::duration_cast<milliseconds>(t2 -t1).count()) / 1000.0;
    BOOST_LOG_TRIVIAL(info) << "Time for " << num_pnts << " evaluations: " << t_eval << " sec";
    _eval_counter += num_pnts;
    return ys;
}
MACE::~MACE()
{
    delete _gp;
}
void MACE::_init_boost_log() const
{
    boost::log::add_common_attributes();
#ifndef MYDEBUG
    boost::log::add_file_log(boost::log::keywords::file_name = _log_name, boost::log::keywords::auto_flush = true);
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );
#else
    boost::log::add_file_log(
            boost::log::keywords::file_name  = _log_name,
            boost::log::keywords::auto_flush = true
            // boost::log::keywords::format     = "[%TimeStamp%]:\n%Message%"
    );
#endif
}
bool MACE::_is_feas(const VectorXd& v) const
{
    bool feas = true;
    if(v.size() > 1)
        feas = (v.tail(v.size() - 1).array() <= 0).all();
    return feas;
}
bool MACE::_better(const VectorXd& v1, const VectorXd& v2) const
{
    if(_is_feas(v1) and _is_feas(v2))
        return v1(0) < v2(0);
    else if(_is_feas(v1))
        return true;
    else if(_is_feas(v2))
        return false;
    else
        return _violation(v1) < _violation(v2);
}
double MACE::_violation(const VectorXd& xs)  const
{
    return xs.size() == 1 ? 0 : xs.tail(xs.size() - 1).cwiseMax(0).sum();
}

// convert from [_scaled_lb, _scaled_ub] to [_lb, _ub]
MatrixXd MACE::_rescale(const MatrixXd& xs) const noexcept
{
    return _a.replicate(1, xs.cols()).cwiseProduct(xs).colwise() + _b;
}

// convert from [lb, ub] to [_scaled_lb, _scaled_ub]
MatrixXd MACE::_unscale(const MatrixXd& xs) const noexcept
{
    return (xs.colwise() - _b).cwiseQuotient(_a.replicate(1, xs.cols()));
}
void MACE::initialize(string xfile, string yfile)
{
    const MatrixXd dbx = read_matrix(xfile);
    const MatrixXd dby = read_matrix(yfile);
    initialize(dbx, dby);
}
void MACE::initialize(const MatrixXd& dbx, const MatrixXd& dby)
{
    // User can provide data for initializing
    if(_gp != nullptr)
    {
        BOOST_LOG_TRIVIAL(error) << "GP is already created!" << endl;
        exit(EXIT_FAILURE);
    }
    MYASSERT(static_cast<size_t>(dbx.rows()) == _dim);
    MYASSERT(static_cast<size_t>(dby.rows()) == _num_spec);
    MYASSERT(dbx.cols() == dby.cols());
    MYASSERT((dbx.array() <= _ub.replicate(1, dbx.cols()).array()).all());
    MYASSERT((dbx.array() >= _lb.replicate(1, dbx.cols()).array()).all());

    MatrixXd scaled_dbx = _unscale(dbx); // scaled from [lb, ub] to [_scaled_lb, _scaled_ub]
    if (dbx.cols() < 2)
    {
        BOOST_LOG_TRIVIAL(error) << "Size of initial sampling is less than 2" << endl;
        exit(EXIT_FAILURE);
    }
    if(! dby.allFinite())
    {
        BOOST_LOG_TRIVIAL(error) << "There are INF|NAN values in dby" << endl;
        exit(EXIT_FAILURE);
    }
    const size_t best_id = _find_best(dby);
    _best_x              = dbx.col(best_id);
    _best_y              = dby.col(best_id);
    _have_feas           = _is_feas(_best_y);
    _gp                  = new GP(scaled_dbx, dby.transpose());
    _no_improve_counter  = 0;
    _hyps                = _gp->get_default_hyps();
    _gp->set_noise_lower_bound(_noise_lvl);
    BOOST_LOG_TRIVIAL(info) << "Initial DBX:\n" << dbx << endl;
    BOOST_LOG_TRIVIAL(info) << "Initial DBY:\n" << dby << endl;
}
void MACE::initialize(size_t init_size)
{
    // Initialization by random simulation
    const MatrixXd dbx = _doe(init_size);
    const MatrixXd dby = _run_func(dbx);
    initialize(_rescale(dbx), dby);
}
size_t MACE::_find_best(const MatrixXd& dby) const
{
    vector<size_t> idxs = _seq_idx(dby.cols());
    sort(idxs.begin(), idxs.end(), [&](const size_t i1, size_t i2)->bool{
        return _better(dby.col(i1), dby.col(i2));
    });
    return idxs[0];
}
vector<size_t> MACE::_seq_idx(size_t n) const
{
    vector<size_t> idxs(n);
    for(size_t i = 0; i < n; ++i)
        idxs[i] = i;
    return idxs;
}
Eigen::MatrixXd MACE::_set_random(size_t num) 
{
    return rand_matrix(num, VectorXd::Constant(_dim, 1, _scaled_lb), VectorXd::Constant(_dim, 1, _scaled_ub), _engine);
}
Eigen::MatrixXd MACE::_doe(size_t num)
{
    MatrixXd sampled(_dim, num);
    
    // DoE to generate points in [0, 1]
    if(_dim <= 40)
    {
        // According to the doc of GSL library, the sobol method only works for dimensions less than 40
        gsl_qrng* q = gsl_qrng_alloc(gsl_qrng_sobol, _dim);
        double* vec = new double[_dim];
        for(size_t i = 0; i < num; ++i)
        {
            gsl_qrng_get(q, vec);
            sampled.col(i) = Eigen::Map<VectorXd>(vec, _dim);
        }
        delete[] vec;
        gsl_qrng_free(q);
    }
    else
    {
        sampled.setRandom();
        sampled = 0.5 * (sampled.array() + 1);
    }

    // transform points from [0, 1] to [lb, ub]
    for(size_t i = 0; i < _dim; ++i)
    {
        double a = _scaled_ub - _scaled_lb;
        double b = _scaled_lb;
        sampled.row(i) = a * sampled.row(i).array() + b;
    }
    return sampled;
}
void MACE::set_init_num(size_t n) { _num_init = n; }
void MACE::set_max_eval(size_t n) { _max_eval = n; }
void MACE::set_batch(size_t n) { _batch_size = n; }
void MACE::set_force_select_hyp(bool f) { _force_select_hyp = f; }
void MACE::set_tol_no_improvement(size_t n) { _tol_no_improvement = n; }
void MACE::set_seed(size_t s) 
{ 
    _seed = s;
    _engine.seed(_seed); 
}
void MACE::set_gp_noise_lower_bound(double lvl) { _noise_lvl = lvl; }
void MACE::set_mo_record(bool r) {_mo_record = r;}
void MACE::set_mo_gen(size_t gen){_mo_gen = gen;}
void MACE::set_mo_np(size_t np){_mo_np = np;}
void MACE::set_mo_f(double f){_mo_f = f;}
void MACE::set_mo_cr(double cr){_mo_cr = cr;}
VectorXd MACE::best_x() const { return _best_x; }
VectorXd MACE::best_y() const { return _best_y; }
void MACE::optimize()
{
    if(_gp == nullptr)
        initialize(_num_init);
    while(_eval_counter < _max_eval)
    {
        optimize_one_step();
    }
}
void MACE::optimize_one_step() // one iteration of BO, so that BO could be used as a plugin of other application
{
    // Train GP model
    if(_gp == nullptr)
    {
        BOOST_LOG_TRIVIAL(error) << "GP not initialized";
        exit(EXIT_FAILURE);
    }
    _train_GP();
    
    // XXX: This is a fast-prototype, possible improvements includes:
    // 1. Use gradient-based MSP to optimize the PF
    // 2. More advanced techniques to incorporate constraints
    // 3. More advanced techniques to transform the LCB function
    
    MOO::ObjF neg_log_pf = [&](const VectorXd xs)->VectorXd{
        MatrixXd y, s2;
        _gp->predict(xs, y, s2);
        return -1 * logphi(-1 * y.cwiseQuotient(s2.cwiseSqrt()));
    };
    if(not _have_feas)
    {
        // If no feasible solution is found, optimize PF firstly
        MOO pf_optimizer(neg_log_pf, 1, VectorXd::Constant(_dim, 1, _scaled_lb), VectorXd::Constant(_dim, 1, _scaled_ub));
        _moo_config(pf_optimizer);
        pf_optimizer.moo();
        _eval_x = pf_optimizer.dby().col(pf_optimizer.best());
        _eval_y = _run_func(_eval_x);
        MYASSERT(pf_optimizer.pareto_set().cols() == 1);
    }
    else
    {
        // If there are feasible solutions, perform MOO to (EI, LCB) functions
        // TODO: incorporate PF
        MYASSERT(_num_spec == 1);
        MOO::ObjF mo_acq = [&](const VectorXd xs)->VectorXd{
            VectorXd objs(2);
            objs << -1 * _log_lcb_improv_transf(xs), -1 * _log_ei(xs);
            return objs;
        };
        MOO acq_optimizer(mo_acq, 2, VectorXd::Constant(_dim, 1, _scaled_lb), VectorXd::Constant(_dim, 1, _scaled_ub));
        _moo_config(acq_optimizer);
        acq_optimizer.set_anchor(_set_anchor());
        acq_optimizer.moo();
        MatrixXd ps = acq_optimizer.pareto_set();
        _eval_x = _slice_matrix(ps, _pick_from_seq(ps.cols(), (size_t)ps.cols() > _batch_size ? _batch_size : ps.cols()));
        _eval_y = _run_func(_eval_x);
#ifdef MYDEBUG
        MatrixXd pf = acq_optimizer.pareto_front();
        BOOST_LOG_TRIVIAL(trace) << "Pareto set:\n"   << _rescale(ps).transpose() << endl;
        BOOST_LOG_TRIVIAL(trace) << "Pareto front:\n" << pf.transpose() << endl;

        VectorXd true_global = _unscale(VectorXd::Zero(_dim));
        MatrixXd y_glb, s2_glb;
        _gp->predict(true_global, y_glb, s2_glb);
        VectorXd acq_glb = mo_acq(true_global);
        BOOST_LOG_TRIVIAL(debug) << "Acq for true global: "  << acq_glb.transpose();
        BOOST_LOG_TRIVIAL(debug) << "Anchor acquisitions:\n" << acq_optimizer.anchor_y().transpose();
#endif
    }
    _print_log();
    _gp->add_data(_eval_x, _eval_y.transpose());
}
void MACE::_print_log()
{
    MatrixXd pred_y, pred_s2;
    _gp->predict(_eval_x, pred_y, pred_s2);
    BOOST_LOG_TRIVIAL(info) << "X:\n"        << _rescale(_eval_x).transpose();
    BOOST_LOG_TRIVIAL(info) << "Pred-S-Eval:";
    for(long i = 0; i < _eval_x.cols(); ++i)
    {
        MatrixXd record(3, _num_spec);
        record << pred_y.row(i), pred_s2.row(i).cwiseSqrt(), _eval_y.col(i).transpose();
        BOOST_LOG_TRIVIAL(info) << record;
        BOOST_LOG_TRIVIAL(info) << "-----";
    }
    BOOST_LOG_TRIVIAL(info) << "Best_y: "    << _best_y;
    BOOST_LOG_TRIVIAL(info) << "Evaluated: " << _eval_counter;
    BOOST_LOG_TRIVIAL(info) << "=============================================";
}
MatrixXd MACE::_slice_matrix(const MatrixXd& m, const vector<size_t>& idxs) const
{
    MYASSERT((long)*max_element(idxs.begin(), idxs.end()) < m.cols());
    MatrixXd sm(m.rows(), idxs.size());
    for(size_t i = 0; i < idxs.size(); ++i)
        sm.col(i) = m.col(idxs[i]);
    return sm;
}

void MACE::_moo_config(MOO& moo_optimizer) const
{
    moo_optimizer.set_f(_mo_f);
    moo_optimizer.set_cr(_mo_cr);
    moo_optimizer.set_np(_mo_np);
    moo_optimizer.set_gen(_mo_gen);
    moo_optimizer.set_seed(_seed);
    moo_optimizer.set_record(_mo_record);
}

void MACE::_train_GP()
{
    auto train_start = chrono::high_resolution_clock::now();
    if (_force_select_hyp || (_no_improve_counter > 0 && _no_improve_counter % _tol_no_improvement == 0))
    {
        BOOST_LOG_TRIVIAL(info) << "Re-select initial hyp" << endl;
        _hyps = _gp->select_init_hyp(1000, _gp->get_default_hyps());
        BOOST_LOG_TRIVIAL(info) << _hyps << endl;
    }
    _nlz  = _gp->train(_hyps);
    _hyps = _gp->get_hyp();
    auto train_end          = chrono::high_resolution_clock::now();
    const double time_train = duration_cast<chrono::milliseconds>(train_end - train_start).count();
    BOOST_LOG_TRIVIAL(info) << "Hyps: \n"               << _hyps.transpose();
    BOOST_LOG_TRIVIAL(info) << "nlz for training set: " << _nlz.transpose();
    BOOST_LOG_TRIVIAL(info) << "Time for GP training: " << (time_train/1000.0) << " s";
}

std::vector<size_t> MACE::_pick_from_seq(size_t n, size_t m)
{
    MYASSERT(m <= n);
    set<size_t> picked_set;
    uniform_int_distribution<size_t> i_distr(0, n - 1);
    while(picked_set.size() < m)
        picked_set.insert(i_distr(_engine));
    vector<size_t> picked_vec(m);
    std::copy(picked_set.begin(), picked_set.end(), picked_vec.begin());
    return picked_vec;
}
double MACE::_pf(const VectorXd& xs) const
{
    MYASSERT(_gp->trained());
    if(_num_spec == 1)
        return 1.0;
    MatrixXd gpy, gps2;
    _gp->predict(xs, gpy, gps2);
    MatrixXd normed = -1 * gpy.cwiseQuotient(gps2.cwiseSqrt());
    double prob = 1.0;
    for(long i = 1; i < gpy.cols(); ++i)
        prob *= normcdf(normed(i));
    return prob;
}
double MACE::_pf(const VectorXd& xs, VectorXd& grad) const
{
    MYASSERT(_gp->trained());
    const double pf  = exp(_log_pf(xs, grad));
    grad            *= pf;
    return pf;
}
double MACE::_log_pf(const VectorXd& xs) const
{
    MYASSERT(_gp->trained());
    if(_num_spec == 1)
        return 0.0;
    MatrixXd gpy, gps2;
    _gp->predict(xs, gpy, gps2);
    MatrixXd normed = -1 * gpy.cwiseQuotient(gps2.cwiseSqrt());
    double log_prob = 0.0;
    for(long i = 1; i < gpy.cols(); ++i)
        log_prob += logphi(normed(i));
    return log_prob;
}
double MACE::_log_pf(const VectorXd& xs, VectorXd& grad) const
{
    MYASSERT(_gp->trained());
    if(_num_spec == 1)
        return 0.0;
    double log_prob = 0.0;
    for(size_t i = 1; i < _num_spec; ++i)
    {
        double y, s2, s;
        VectorXd gy, gs2, gs;
        _gp->predict_with_grad(i, xs, y, s2, gy, gs2);
        s      = sqrt(s2);
        gs     = 0.5 * gs2 / sqrt(s2);
        double normed    = -1 * y / s;
        VectorXd gnormed = -1 * (s * gy - y * gs) / s2;
        double lp, dlp;
        logphi(normed, lp, dlp);
        log_prob += lp;
        grad     += dlp * gnormed;
    }
    return log_prob;
}

double MACE::_ei(const Eigen::VectorXd& x) const
{
    MYASSERT(_gp->trained());
    double  y, s2;
    _gp->predict(0, x, y, s2);
    const double s      = sqrt(s2);
    const double tau    = _best_y(0);
    const double normed = (tau - y) / sqrt(s2);
    return s * (normed * normcdf(normed) + normpdf(normed));
}

double MACE::_ei(const Eigen::VectorXd& x, VectorXd& grad) const
{
    MYASSERT(_gp->trained());
    const double tau = _best_y(0);
    double  y, s2, s;
    VectorXd gy, gs2, gs;
    _gp->predict_with_grad(0, x, y, s2, gy, gs2);
    s  = sqrt(s2);
    gs = 0.5 * gs2 / sqrt(s2);
    const double   normed    = (tau - y) / sqrt(s2);
    const double   cdfnormed = normcdf(normed);
    const VectorXd gnormed   = -1 * (s * gy + (tau - y) * gs) / s2;
    const double   lambda    = normed * cdfnormed + normpdf(normed);
    grad = s * cdfnormed * gnormed + lambda * gs;
    return s * lambda;
}

double MACE::_log_ei(const Eigen::VectorXd& x) const
{
    double y, s2;
    _gp->predict(0, x, y, s2);
    const double s      = sqrt(s2);
    const double tau    = _best_y(0);
    const double normed = (tau - y) / sqrt(s2);
    return normed > -6 ? log(s * (normed * normcdf(normed) + normpdf(normed))) 
                       : log(s) - 0.5 * pow(normed, 2) - log(sqrt(2 * M_PI)) - log(pow(normed, 2) - 1);
    // \lim_{z \to -\infty} \log\big(z\Phi(z) + \phi(z)\big) = \log \phi(z) - \log(z^2 - 1) 
}

double MACE::_log_ei(const Eigen::VectorXd& x, VectorXd& grad) const
{
    const double tau = _best_y(0);
    double y, s2, s;
    VectorXd gy, gs2, gs;
    _gp->predict_with_grad(0, x, y, s2, gy, gs2);
    s  = sqrt(s2);
    gs = 0.5 * gs2 / sqrt(s2);
    const double normed = (tau - y) / sqrt(s2);
    const VectorXd gnormed   = -1 * (s * gy + (tau - y) * gs) / s2;
    double log_ei;
    if(normed > -6)
    {
        const double   cdfnormed = normcdf(normed);
        const double   lambda    = normed * cdfnormed + normpdf(normed);
        double ei                = s * lambda;
        grad   = (s * cdfnormed * gnormed + lambda * gs) / ei;
        log_ei = log(ei);
    }
    else
    {
        grad   = gs / s - normed * gnormed - (2 * normed) / (pow(normed, 2) - 1) * gnormed;
        log_ei = log(s) - 0.5 * pow(normed, 2) - log(sqrt(2 * M_PI)) - log(pow(normed, 2) - 1);
    }
    return log_ei;
}
double MACE::_lcb_improv(const Eigen::VectorXd& x) const 
{
    const double tau   = _best_y(0);
    double y, s2;
    _gp->predict(0, x, y, s2);
    const double lcb = y - _kappa * sqrt(s2);
    return tau - lcb;
}
double MACE::_lcb_improv(const Eigen::VectorXd& x, VectorXd& grad) const 
{
    const double tau   = _best_y(0);
    double y, s2, lcb;
    VectorXd gy, gs2, gs;
    _gp->predict_with_grad(0, x, y, s2, gy, gs2);
    gs   = 0.5 * gs2 / sqrt(s2);
    lcb  = y - _kappa * sqrt(s2);
    grad = -1 * (gy - _kappa * gs);
    return tau - lcb;
}
double MACE::_lcb_improv_transf(const Eigen::VectorXd& x) const
{
    return log(1 + exp(_lcb_improv(x)));
}
double MACE::_lcb_improv_transf(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const
{
    const double lcb_improve = _lcb_improv(x, grad);
    const double val         = log(1+exp(lcb_improve));
    grad *= exp(val) / (1 + exp(val));
    return val;
}
double MACE::_log_lcb_improv_transf(const Eigen::VectorXd& x) const
{
    const double lcb_improve = _lcb_improv(x);
    return lcb_improve > -10 ? log(log(1+exp(lcb_improve)))
                             : lcb_improve - 0.5 * exp(lcb_improve);
}
double MACE::_log_lcb_improv_transf(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const
{
    const double lcb_improve = _lcb_improv(x, grad);
    double val;
    if(lcb_improve > -10)
    {
        val   = log(log(1+exp(lcb_improve)));
        grad *= exp(lcb_improve) / (log(1+exp(lcb_improve)) * (1 + exp(lcb_improve)));
    }
    else
    {
        val   = lcb_improve - 0.5 * exp(lcb_improve);
        grad *= (1 - 0.5 * exp(lcb_improve));
    }
    return val;
}
Eigen::VectorXd MACE::_msp(NLopt_wrapper::func f, const Eigen::MatrixXd& sp, nlopt::algorithm algo, size_t max_eval)
{
    double best_y = INF;
    VectorXd best_x;
#pragma omp parallel for
    for (long i = 0; i < sp.cols(); ++i)
    {
        NLopt_wrapper opt(algo, _dim, _scaled_lb, _scaled_ub);
        opt.set_maxeval(max_eval);
        opt.set_ftol_rel(1e-6);
        opt.set_xtol_rel(1e-6);
        opt.set_min_objective(f);
        VectorXd x = sp.col(i);
        double y = INF;
        try
        {
            opt.optimize(x, y);
        }
        catch (runtime_error& e)  // this kind of exception can usually be ignored
        {
#ifdef MYDEBUG
            BOOST_LOG_TRIVIAL(warning) << "Nlopt exception: " << e.what() << " for sp: " << sp.col(i).transpose()
                                       << ", y = " << y;
#endif
        }
        catch (exception& e)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Nlopt exception: " << e.what() << " for sp: " << sp.col(i).transpose()
                                     << ", y = " << y;
            exit(EXIT_FAILURE);
        }
#pragma omp critical
        {
            if (y < best_y)
            {
                best_x = x;
                best_y = y;
            }
        }
    }
    return best_x;
}
MatrixXd MACE::_set_anchor()
{
    const size_t num_weight    = 10;
    const size_t num_rand_samp = 5;
    MatrixXd sp(_dim, 1 + num_rand_samp + _eval_x.cols());
    MatrixXd anchor(_dim, num_weight + 1 + sp.cols());
    sp << _unscale(_best_x), _eval_x, _set_random(num_rand_samp);
    MatrixXd heuristic_anchors(_dim, num_weight + 1);
    for(size_t i = 0; i <= num_weight; ++i)
    {
        const double alpha = (1.0 * i) / num_weight;
        NLopt_wrapper::func f = [&](const VectorXd& x, VectorXd& grad)->double{
            double log_ei, log_lcb_improv_transf;
            VectorXd glog_ei, glog_lcb_improv_transf;
            log_ei                = _log_ei(x, glog_ei);
            log_lcb_improv_transf = _log_lcb_improv_transf(x, glog_lcb_improv_transf);
            grad                  = -1 * (alpha * glog_ei + (1.0 - alpha) * glog_lcb_improv_transf);
            return -1 * (alpha * log_ei + (1.0 - alpha) * log_lcb_improv_transf);
        };
        heuristic_anchors.col(i) = _msp(f, sp, nlopt::LD_SLSQP);
    }
    anchor << sp, heuristic_anchors;
    return anchor;
}
