#include "Config.h"
#include "util.h"
#include "MACE_util.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <omp.h>
using namespace std;
using namespace Eigen;
Config::Config(string file_path) : _file_path(file_path) {}
void Config::parse()
{
    ifstream f;
    f.open(_file_path);
    MYASSERT(f.is_open() && !f.fail());
    string line;
    _des_var_names.clear();
    vector<double> lbs;
    vector<double> ubs;
    while (getline(f, line))
    {
        string tok;
        stringstream ss(line);
        ss >> tok;
        if (tok == "workdir")
        {
            ss >> _work_dir;
        }
        else if (tok == "des_var")
        {
            string name;
            double lb, ub;
            ss >> name >> lb >> ub;
            if (ub < lb) throw std::runtime_error("ub > lb for line " + line);
            _des_var_names.push_back(name);
            lbs.push_back(lb);
            ubs.push_back(ub);
        }
        else if (tok == "option")
        {
            string k;
            double v;
            ss >> k >> v;
            _options[k] = v;
        }
    }
    MYASSERT(_des_var_names.size() == lbs.size());
    MYASSERT(_des_var_names.size() == ubs.size());
    _des_var_lb = convert(lbs);
    _des_var_ub = convert(ubs);
    f.close();
}
string Config::work_dir() const { return _work_dir; }
const map<string, double>& Config::options() const { return _options; }
VectorXd Config::lb() const { return _des_var_lb; }
VectorXd Config::ub() const { return _des_var_ub; }
MACE::Obj Config::gen_obj()
{
    string cir_dir = _work_dir + "/circuit";
    run_cmd("mkdir "  + _work_dir + "/work/");
    const size_t num_threads = omp_get_max_threads();
    for(size_t i = 0; i < num_threads; ++i)
        run_cmd("cp -r " + cir_dir + " " + _work_dir + "/work/" + to_string(i)); 
    MACE::Obj f =  [&](const VectorXd& xs) -> VectorXd {
        // check range
        const size_t dim       = _des_var_names.size();
        const size_t num_spec  = with_default<size_t>(_options, "num_spec", 1);
        const string opt_dir   = _work_dir + "/work/" + to_string(omp_get_thread_num());
        MYASSERT((size_t)xs.rows() == dim);
        VectorXd sim_results(num_spec);

        ofstream param_f;
        string param_file = opt_dir + "/param";
        param_f.exceptions(param_f.badbit | param_f.failbit);
        param_f << setprecision(18);
        param_f.open(param_file);
        for (size_t j = 0; j < dim; ++j) 
            param_f << ".param " << _des_var_names[j] << " = " << xs(j) << endl;
        param_f.close();
        const string cmd = "cd " + opt_dir + " && perl run.pl > output_info.log 2>&1";
        int ret = system(cmd.c_str());
        if (ret != 0)
        {
            cerr << "Fail to run cmd " << cmd << endl;
            exit(EXIT_FAILURE);
        }
        MatrixXd result = read_matrix(opt_dir + "/result.po");
        MYASSERT(result.rows() == 1);
        MYASSERT((size_t)result.size() == num_spec);
        sim_results = result.transpose();

        return sim_results;
    };
    return f;
}
void Config::print()
{
    cout << "Conf path: " << _file_path << endl;
    cout << "work dir: " <<  _work_dir  << endl;
    for(size_t i = 0; i < _des_var_names.size(); ++i)
    {
        cout << _des_var_names[i] << ": " << _des_var_lb[i] << ", " << _des_var_ub[i] << endl;
    }
    for(auto p : _options)
    {
        cout << "Option " << p.first << " = " << p.second << endl;
    }
}
boost::optional<double> Config::lookup(string name) const
{
    auto find_result = _options.find(name);
    if(find_result == _options.end())
        return boost::none;
    else
        return find_result->second;
}
