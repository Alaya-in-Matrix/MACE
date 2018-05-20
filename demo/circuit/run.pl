#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

my %params;
open my $ifh, "<", "param" or die "Can't read param:$!\n";
while(my $line = <$ifh>)
{
    chomp($line);
    if($line =~ /\.param\s+(\w+)\s+=\s+(.*)/)
    {
        $params{$1} = $2+0;
    }
}
close $ifh;
die "x does not exist" if(not exists $params{x});
die "y does not exist" if(not exists $params{y});
my $x   = $params{x}-1;
my $y   = $params{y}-1;
my $fom = (1-$x)**2 + 100 * ($y - $x**2)**2;

open my $ofh, ">", "result.po" or die "Can't create result.po: $!\n";
say $ofh $fom;
close $ofh;

open my $rfh, ">>", "record" or die "Can't create record: $!\n";
say $rfh "$x $y $fom";
close $rfh;
