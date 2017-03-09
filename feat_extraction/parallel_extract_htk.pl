#!/usr/bin/perl

#
# parallel feature extraction
# (c) Florian Eyben, 2013
#

#
# WARNING: this script does not overwrite existing files.
# Set the variable $clean=1 to remove exisiting files.
# It also assumes the output extension .htk.
#

use strict;

my $clean : shared = 0;  # set to 1 to overwrite exisiting .htk files
my $wavpath = $ARGV[0];
my $out_dir = $ARGV[1];
my $conf = $ARGV[2];
my $npar = $ARGV[3];
my $smilebin = $ARGV[4];
unless ($smilebin) {
  $smilebin = "SMILExtract";  # use binary in path
}

print "openSMILE parallel extraction script. Author: Florian Eyben.\n";

if ($#ARGV < 3) {
  print "USAGE: extract.pl <wav-glob> <base path for output htk files> <smile config> <nParallel> [smile binary, full path]\n";
  print "  If <wav-glob> begins with --list= , e.g. --list=/filename/of/list then\n";
  print "  a list of wave files is loaded from the given text file,\n";
  print "  instead of the directory glob 'wav-glob' (e.g. '/my/path/*.wav')\n";
  print "  NOTE: the instances in the arff file are NOT in the same order\n";
  print "        as the wave files. Use tum_mmk_tools/perlscripts/arff_sort.pl\n";
  print "        to sort the instances in ascending alphabetical order.\n";
  exit 1;
}

use threads;
use threads::shared;
use File::Basename;
use Cwd;
use FileHandle;

my $n_done : shared = 0;
print "  nParallel = $npar\n";

my @wavs;
if ($wavpath =~ /--list=(.+)$/) {
  my $wavlist = $1;
  print "Loading list of wave files from $wavlist ...\n";
  open(F,"<$wavlist");
  while(<F>) {
    chomp;
    push(@wavs, $_);
  }
  close(F);
} else {
  print "Getting list of wave files by glob: '$wavpath' ...\n";
  @wavs = glob("$wavpath");
}
print "    ".($#wavs+1)." wav files in list.\n";
my $n_total : shared = $#wavs+1;

if ($#wavs < 0) {
  print "ERROR: no wave files found.\n";
  exit 1;
}

if ($#wavs < $npar-1) {
  print "Set too small for parallelisation with $npar folds.\n";
  print "  Running sequentially (npar = 1).\n";
  $npar = 1;
}

print "Splitting in $npar folds...\n";
my @w; 
my $i = 0;
my $j = 0;
my $n = 0;
my @ids = ();
my $curid = 0;
for ($i=0; $i<=$#wavs; ) {
  for ($j=0; $j<$npar; $j++) {
    $w[$j][$n] = $wavs[$i];
    push(@ids, $wavs[$i]);
    $i++;
  }
  $n++;
}


sub start_thread {
  my $threadnr = shift(@_);
  my $smilebin = shift(@_);
  my $out_dir = shift(@_);
  my $conf = shift(@_);
  my @args = @_;
  my $i;
  for ($i=0; $i<=$#args; $i++) {
    my $wav = $args[$i];
    unless ($wav) { next; }
    #if ($wav =~ /\.tmp\.wav$/) { next; }
    my $b = basename($wav);
    $b =~ s/\.wav$/.htk/i;
    unless (-e "$out_dir/$b" && $clean == 0) {
      my $cmd = "$smilebin -l 0 -nologfile -C \"$conf\" -I \"$wav\" -O \"$out_dir/$b\"";
      my $ret = system("nice -19 $cmd 2> /dev/null > /dev/null");
      if ($ret) { 
        print "ERROR running openSMILE.\n";
        print "Failed command is:\n  $cmd\n";
        return 1; 
      }
    }
    {
      lock($n_done);
      $n_done++;
      if (($n_done) % 100 == 0) {
        print "Done ".($n_done)." of $n_total files.\n";
      }
    }
  }
  return 0;
}


mkdir($out_dir);
my @thr;
print "Starting feature extrator threads... ";
for ($i=0; $i<$npar; $i++) {
  $thr[$i] = threads->create('start_thread', $i, $smilebin, $out_dir, $conf, @{$w[$i]});
  print "$i ";
}
print "DONE.\n";
# TODO: for distributed systems: get the *.done file in the arff directory by polling every 10sec.
for ($i=0; $i<$npar; $i++) {
  $thr[$i]->join();
}

print "All threads finished. DONE.\n";

