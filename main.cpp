#include "util.h"
#include "Config.h"
#include "MACE.h"
#include "MACE_util.h"
#include "NLopt_wrapper.h"
#include <iostream>
#include <boost/optional/optional_io.hpp>
#include <omp.h>
using namespace std;
using namespace Eigen;
int main(int arg_num, char** args) 
{
    srand(rand_seed);

    if(arg_num < 2)
    {
        cerr << "Usage: mobo_wapi path/to/conf/file" << endl;
        return EXIT_FAILURE;
    }
#ifdef MYDEBUG
    run_cmd("rm -rvf work");
#endif

    string conf_file(args[1]);

    Config conf(conf_file);
    conf.parse();
    conf.print();

    size_t num_spec   = 0;

    try
    {
        num_spec   = conf.lookup("num_spec").value();
    }
    catch (const boost::bad_optional_access& e)
    {
        cerr << e.what() <<" You need to specify `num_spec` in configure file." << endl;
        return EXIT_FAILURE;
    }

    const size_t dim        = conf.lb().size();
    const size_t num_thread = conf.lookup("num_thread").value_or(1);
    const size_t max_eval   = conf.lookup("max_eval").value_or(dim * 20);
    const size_t num_init   = conf.lookup("num_init").value_or(1 + dim);
    const size_t tol_no_improvement = conf.lookup("tol_no_improvement").value_or(10);
    const double noise_lb   = conf.lookup("noise_lb").value_or(1e-6);
    const bool   mo_record  = conf.lookup("mo_record").value_or(false);
    const double mo_f       = conf.lookup("mo_f").value_or(0.8);
    const double mo_cr      = conf.lookup("mo_cr").value_or(0.8);
    const size_t mo_gen     = conf.lookup("mo_gen").value_or(100);
    const size_t mo_np      = conf.lookup("mo_np").value_or(100);

    omp_set_num_threads(num_thread);

    MACE::Obj obj = conf.gen_obj();

    MACE mace(obj, num_spec, conf.lb(), conf.ub());
    // TODO: more manual configuration
    mace.set_tol_no_improvement(tol_no_improvement);
    mace.set_max_eval(max_eval);
    mace.set_init_num(num_init);
    mace.set_gp_noise_lower_bound(noise_lb);
    mace.set_batch(num_thread);
    mace.set_mo_record(mo_record);
    mace.set_mo_f(mo_f);
    mace.set_mo_cr(mo_cr);
    mace.set_mo_gen(mo_gen);
    mace.set_mo_np(mo_np);
    mace.initialize(num_init);
    mace.optimize();
    cout << "Best x: " << mace.best_x().transpose() << endl;
    cout << "Best y: " << mace.best_y().transpose() << endl;
    return EXIT_SUCCESS;
}
