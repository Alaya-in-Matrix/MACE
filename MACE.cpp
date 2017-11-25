#include "MACE.h"
#include "util.h"
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
      _best_y(RowVectorXd::Constant(1, _num_spec, INF))
{
    assert(_scaled_lb   < _scaled_ub);
    assert((_lb.array() < _ub.array()).all());
    _init_boost_log();
    BOOST_LOG_TRIVIAL(info) << "MACE Created" << endl;
    _run_func = [&](const VectorXd& xs) -> VectorXd {
        ++_eval_counter;
        const auto t1            = chrono::high_resolution_clock::now();
        const VectorXd scaled_xs = _rescale(xs);
        const VectorXd ys        = _func(scaled_xs);
        bool no_improve = true;
        if (_better(ys, _best_y))
        {
            _best_x    = scaled_xs;
            _best_y    = ys;
            no_improve = false;
        }
        if (_is_feas(ys))
            _have_feas = true;
        if (no_improve)
            ++_no_improve_counter;
        else
            _no_improve_counter = 0;
        const auto t2       = chrono::high_resolution_clock::now();
        const double t_eval = static_cast<double>(chrono::duration_cast<milliseconds>(t2 -t1).count()) / 1000.0;
        BOOST_LOG_TRIVIAL(info) << "Time for evaluation " << _eval_counter << ": " << t_eval << " sec";
        return ys;
    };
}
MACE::~MACE()
{
    delete _gp;
}
void MACE::_init_boost_log() const
{
    boost::log::add_common_attributes();
    boost::log::add_file_log(_log_name);
#ifndef MYDEBUG
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
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
    _gp                  = new GP(scaled_dbx, dby);
    _no_improve_counter  = 0;
    _hyps                = _gp->get_default_hyps();
    BOOST_LOG_TRIVIAL(info) << "Initial DBX:\n" << dbx << endl;
    BOOST_LOG_TRIVIAL(info) << "Initial DBY:\n" << dby << endl;
}
void MACE::initialize(size_t init_size)
{
    // Initialization by random simulation
    MatrixXd dbx = _doe(init_size);
    MatrixXd dby = MatrixXd(_num_spec, init_size);
    
    for(size_t i = 0; i < init_size; ++i)
        dby.col(i) = _run_func(dbx.col(i));
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
