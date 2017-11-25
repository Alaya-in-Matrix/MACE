#include "MACE.h"
#include <omp.h>
#include <chrono>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
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
        const double t_eval = chrono::duration_cast<milliseconds>(t2 -t1).count();
        BOOST_LOG_TRIVIAL(info) << "Time for evaluation " << _eval_counter << ", " << t_eval << endl; 
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
    // boost::log::add_file_log(boost::log::keywords::file_name = _log_name,
    //                          boost::log::keywords::format = "[%TimeStamp%]: %Message%"
    //                          );
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
