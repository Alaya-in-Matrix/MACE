#include "MACE_util.h"
#include <iostream>
using namespace std;

void run_cmd(string cmd)
{
    int ret = system(cmd.c_str());
    if(ret != 0)
    {
        cerr << "Fail to execute cmd \"" << cmd << "\", exit code: " << ret;
        exit(ret);
    }
}
int run_cmd(vector<string> cmds)
{
    int ret;
    for(auto c : cmds)
    {
        ret = system(c.c_str());
        if(ret != 0)
            break;
    }
    return ret;
}
