#pragma once
#include "MACE.h"
#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <boost/optional.hpp>
// read configuration 
class Config
{
    std::string              _file_path;
    std::string              _work_dir;
    Eigen::VectorXd          _des_var_lb;
    Eigen::VectorXd          _des_var_ub;
    std::vector<std::string> _des_var_names;
    std::map<std::string, double> _options;
public:
    explicit Config(std::string);
    void parse();
    void print();
    std::string work_dir() const;
    const decltype(_options)& options() const;
    MACE::Obj gen_obj();
    MACE::Obj gen_obj(std::string);
    Eigen::VectorXd lb() const;
    Eigen::VectorXd ub() const;
    boost::optional<double> lookup(std::string) const;
};
