#include "util.h"
#include "Config.h"
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
    const double noise_lb   = conf.lookup("noise_lb").value_or(1e-6);
    const double alpha      = conf.lookup("alpha").value_or(0.5);

    omp_set_num_threads(num_thread);

    VectorXd param = rand_matrix(1, conf.lb(), conf.ub());
    MACE::Obj obj = conf.gen_obj();

    cout << obj(conf.lb()).transpose() << endl;
    cout << obj(conf.ub()).transpose() << endl;

    MACE mace(obj, 1, conf.lb(), conf.ub());
    mace.initialize(num_init);

    return EXIT_SUCCESS;
}
