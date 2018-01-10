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
    // srand(rand_seed);
    srand(random_device{}());

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

    const size_t dim                = conf.lb().size();
    const size_t num_thread         = conf.lookup("num_thread").value_or(1);
    const size_t max_eval           = conf.lookup("max_eval").value_or(dim * 20);
    const size_t num_init           = conf.lookup("num_init").value_or(1 + dim);
    const size_t tol_no_improvement = conf.lookup("tol_no_improvement").value_or(10);
    const double noise_lb           = conf.lookup("noise_lb").value_or(1e-6);
    const double eval_fixed         = conf.lookup("eval_fixed").value_or(max_eval);
    const bool   mo_record          = conf.lookup("mo_record").value_or(false);
    const double mo_f               = conf.lookup("mo_f").value_or(0.5);
    const double mo_cr              = conf.lookup("mo_cr").value_or(0.3);
    const size_t mo_gen             = conf.lookup("mo_gen").value_or(250);
    const size_t mo_np              = conf.lookup("mo_np").value_or(100);
    const size_t selection_strategy = conf.lookup("selection_strategy").value_or(0);
    const bool   use_sobol          = conf.lookup("use_sobol").value_or(false);
    const bool   noise_free         = conf.lookup("noise_free").value_or(false);
    const double upsilon            = conf.lookup("upsilon").value_or(0.2);
    const double delta              = conf.lookup("delta").value_or(0.1);
    const double EI_jitter          = conf.lookup("EI_jitter").value_or(0.0);
    const double eps                = conf.lookup("eps").value_or(1e-3);
    const bool   force_select_hyp   = conf.lookup("force_select_hyp").value_or(false);
    const string algo               = conf.algo();
    MACE::SelectStrategy ss;
    switch(selection_strategy)
    {
        case 0:
            ss = MACE::SelectStrategy::Random;
            break;
        case 1:
            ss = MACE::SelectStrategy::Greedy;
            break;
        case 2:
            ss = MACE::SelectStrategy::Extreme;
            break;
        default:
            cout << "Unknown selection strategy, 0 for random, 1 for greedy, 2 for extreme" << endl;
            exit(EXIT_FAILURE);
    }

    omp_set_num_threads(num_thread);

    MACE::Obj obj = conf.gen_obj();

    MACE mace(obj, num_spec, conf.lb(), conf.ub());
    // Optional algorithm settings
    mace.set_tol_no_improvement(tol_no_improvement);
    mace.set_eval_fixed(eval_fixed);
    mace.set_max_eval(max_eval);
    mace.set_init_num(num_init);
    mace.set_batch(num_thread);
    mace.set_mo_record(mo_record);
    mace.set_force_select_hyp(force_select_hyp);
    mace.set_mo_f(mo_f);
    mace.set_mo_cr(mo_cr);
    mace.set_mo_gen(mo_gen);
    mace.set_mo_np(mo_np);
    mace.set_selection_strategy(ss);
    mace.set_use_sobol(use_sobol);
    mace.set_lcb_upsilon(upsilon);
    mace.set_lcb_delta(delta);
    mace.set_EI_jitter(EI_jitter);
    mace.set_eps(eps);
    if(not noise_free)
        mace.set_gp_noise_lower_bound(noise_lb);
    mace.set_noise_free(noise_free);
    mace.initialize(num_init);
    if(algo == "mace")
        mace.optimize();
    else if(algo == "blcb")
        mace.blcb();
    else 
    {
        cerr << "Unknown algo: " << algo << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Best x: " << mace.best_x().transpose() << endl;
    cout << "Best y: " << mace.best_y().transpose() << endl;
    return EXIT_SUCCESS;
}
