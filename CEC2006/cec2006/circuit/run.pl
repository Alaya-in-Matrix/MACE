#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

my @params;
my $prob = "g01";
my $ng   = 9;
my $nh   = 0;

open my $param_f, "<", "./param" or die "Can't open param:$!\n";
while(my $line = <$param_f>)
{
    chomp($line);
    if($line =~ /^\.param.*\=\s*(.*)/)
    {
        push @params, $1;
    }
    else
    {
        say "Invalid line in param:$line";
    }
}
close $param_f;
my $dim = scalar @params;


open my $cec_param, ">", "./cec2006/param" or die "Can't create cec2006/param:$!\n";
say $cec_param $_ for(@params);
close $cec_param;

run_cmd("cd cec2006 && mytest $prob $dim $ng $nh > result.po");
run_cmd("cp ./cec2006/result.po ./");

sub run_cmd
{
    my $cmd = shift;
    my $ret = system($cmd);
    if($ret != 0)
    {
        die "Fail to run cmd: $cmd\n";
    }
}
