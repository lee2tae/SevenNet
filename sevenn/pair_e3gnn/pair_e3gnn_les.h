/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
http://lammps.sandia.gov, Sandia National Laboratories
Steve Plimpton, sjplimp@sandia.gov

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(e3gnn/les, PairE3GNNLES)

#else

#ifndef LMP_PAIR_E3GNN_LES
#define LMP_PAIR_E3GNN_LES

#include "pair.h"

#include <torch/torch.h>

namespace LAMMPS_NS {
class PairE3GNNLES : public Pair {
private:
  double cutoff;
  double cutoff_square;
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  bool print_info = false;

  int nedges_bound = -1;

public:
  PairE3GNNLES(class LAMMPS *);
  ~PairE3GNNLES();
  void compute(int, int);

  void settings(int, char **);
  // read Atom type string from input script & related coeff
  void coeff(int, char **);
  void allocate();

  void init_style();
  double init_one(int, int);
};
} // namespace LAMMPS_NS

#endif
#endif
