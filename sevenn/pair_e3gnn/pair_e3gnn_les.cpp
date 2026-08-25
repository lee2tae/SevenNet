/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Yutack Park (SNU), Eui Tae Lee (SNU)

   Pair style for SevenNet models trained with the LES (Latent Ewald
   Summation) long-range module. Unlike pair e3gnn, the deployed LES model
   keeps EdgePreprocess and ForceStressOutput: it takes positions, the
   (periodicity-masked) cell and per-edge integer pbc shifts, and returns
   total energy, forces and the complete stress (SR virial + LR positional
   + LR k-space/cell terms) from a single in-graph strain derivative.

   The Ewald sum needs the entire simulation cell on one rank, so this
   pair style runs on a single MPI rank only (like pair e3gnn, one GPU).
------------------------------------------------------------------------- */

#include <ATen/ops/from_blob.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include "pair_e3gnn_les.h"

using namespace LAMMPS_NS;

// Undefined reference; body in pair_e3gnn_oeq_autograd.cpp to be linked
extern void pair_e3gnn_oeq_register_autograd();

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

PairE3GNNLES::PairE3GNNLES(LAMMPS *lmp) : Pair(lmp) {
  const char *print_flag = std::getenv("SEVENN_PRINT_INFO");
  if (print_flag)
    print_info = true;

  std::string device_name;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "PairE3GNNLES using device : %s\n",
            device_name.c_str());
  }
}

PairE3GNNLES::~PairE3GNNLES() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
  }
}

void PairE3GNNLES::compute(int eflag, int vflag) {
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  if (vflag_atom) {
    error->all(FLERR, "Pair e3gnn/les does not support per-atom virial: the "
                      "long-range stress has no per-atom decomposition");
  }
  if (atom->tag_consecutive() == 0) {
    error->all(FLERR, "Pair e3gnn/les requires consecutive atom IDs");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = list->inum;
  int *ilist = list->ilist;
  const int inum = list->inum;

  tagint *tag = atom->tag;

  std::unordered_map<tagint, int> tag_map;
  std::vector<int> graph_index_to_i(nlocal);

  long num_atoms[1] = {nlocal};

  int *numneigh = list->numneigh;      // j loop cond
  int **firstneigh = list->firstneigh; // j list

  int bound;
  if (this->nedges_bound == -1) {
    bound = std::accumulate(numneigh, numneigh + nlocal, 0);
  } else {
    bound = this->nedges_bound;
  }
  const int nedges_upper_bound = bound;

  // full box cell for the fractional-shift conversion; the model input cell
  // has non-periodic rows zeroed (LES routes bulk/slab/wire/molecule by the
  // zero-row pattern, matching sevenn's _pbc_masked_cell convention)
  float cell_full[3][3];
  cell_full[0][0] = domain->boxhi[0] - domain->boxlo[0];
  cell_full[0][1] = 0.0;
  cell_full[0][2] = 0.0;

  cell_full[1][0] = domain->xy;
  cell_full[1][1] = domain->boxhi[1] - domain->boxlo[1];
  cell_full[1][2] = 0.0;

  cell_full[2][0] = domain->xz;
  cell_full[2][1] = domain->yz;
  cell_full[2][2] = domain->boxhi[2] - domain->boxlo[2];

  const double volume =
      static_cast<double>(cell_full[0][0]) * cell_full[1][1] * cell_full[2][2];

  float cell_masked[3][3];
  for (int d = 0; d < 3; d++) {
    for (int e = 0; e < 3; e++) {
      cell_masked[d][e] = domain->periodicity[d] ? cell_full[d][e] : 0.0;
    }
  }

  torch::Tensor inp_cell_full = torch::from_blob(cell_full, {3, 3}, FLOAT_TYPE);
  torch::Tensor inp_cell =
      torch::from_blob(cell_masked, {3, 3}, FLOAT_TYPE).clone();
  torch::Tensor inp_num_atoms = torch::from_blob(num_atoms, {1}, INTEGER_TYPE);
  torch::Tensor inp_cell_volume = torch::tensor(volume, FLOAT_TYPE);

  torch::Tensor inp_node_type = torch::zeros({nlocal}, INTEGER_TYPE);
  torch::Tensor inp_pos = torch::zeros({nlocal, 3}, FLOAT_TYPE);

  auto node_type = inp_node_type.accessor<long, 1>();
  auto pos = inp_pos.accessor<float, 2>();

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    tag_map[tag[i]] = ii;
    graph_index_to_i[ii] = i;
    node_type[ii] = map[type[i]];
    pos[ii][0] = x[i][0];
    pos[ii][1] = x[i][1];
    pos[ii][2] = x[i][2];
  }

  std::vector<long> edge_idx_src;
  std::vector<long> edge_idx_dst;
  std::vector<float> shift_cart; // x[j_ghost] - x[j_local], flattened (E, 3)
  edge_idx_src.reserve(nedges_upper_bound);
  edge_idx_dst.reserve(nedges_upper_bound);
  shift_cart.reserve(nedges_upper_bound * 3);

  int nedges = 0;
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int i_graph_idx = ii;
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const tagint jtag = tag[j];
      j &= NEIGHMASK;

      const auto found = tag_map.find(jtag);
      if (found == tag_map.end()) continue;
      const int j_graph_idx = found->second;

      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1],
                               x[j][2] - x[i][2]};
      const double Rij =
          delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      if (Rij < cutoff_square) {
        edge_idx_src.push_back(i_graph_idx);
        edge_idx_dst.push_back(j_graph_idx);
        shift_cart.push_back(x[j][0] - pos[j_graph_idx][0]);
        shift_cart.push_back(x[j][1] - pos[j_graph_idx][1]);
        shift_cart.push_back(x[j][2] - pos[j_graph_idx][2]);
        nedges++;
      }
    } // j loop end
  }   // i loop end

  auto edge_idx_src_tensor =
      torch::from_blob(edge_idx_src.data(), {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor =
      torch::from_blob(edge_idx_dst.data(), {nedges}, INTEGER_TYPE);
  auto inp_edge_index =
      torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  // shift_frac = shift_cart @ cell^-1, rounded to exact integers; the model
  // reconstructs edge vectors as pos[dst] - pos[src] + shift @ strained_cell
  auto shift_cart_tensor =
      torch::from_blob(shift_cart.data(), {nedges, 3}, FLOAT_TYPE);
  auto inp_pbc_shift =
      torch::round(torch::matmul(shift_cart_tensor, inp_cell_full.inverse()));

  if (print_info) {
    std::cout << " Nlocal: " << nlocal << std::endl;
    std::cout << " Nedges: " << nedges << "\n" << std::endl;
  }

  torch::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("pos", inp_pos.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("pbc_shift", inp_pbc_shift.to(device));
  input_dict.insert("cell_lattice_vectors", inp_cell.to(device));
  input_dict.insert("cell_volume", inp_cell_volume.to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));

  std::vector<torch::IValue> input(1, input_dict);
  auto output = model.forward(input).toGenericDict();

  torch::Tensor total_energy_tensor =
      output.at("inferred_total_energy").toTensor().to(torch::kCPU).squeeze();
  torch::Tensor force_tensor =
      output.at("inferred_force").toTensor().to(torch::kCPU);
  auto forces = force_tensor.accessor<float, 2>();

  eng_vdwl += total_energy_tensor.item<float>();

  for (int gi = 0; gi < nlocal; gi++) {
    const int i = graph_index_to_i[gi];
    f[i][0] += forces[gi][0];
    f[i][1] += forces[gi][1];
    f[i][2] += forces[gi][2];
  }

  if (vflag) {
    // model stress is -dE/dstrain / V in Voigt order xx yy zz xy yz xz;
    // LAMMPS virial is stress * V in order xx yy zz xy xz yz
    torch::Tensor virial_tensor =
        output.at("inferred_stress").toTensor().to(torch::kCPU) *
        inp_cell_volume;
    auto vs = virial_tensor.accessor<float, 1>();
    virial[0] += vs[0];
    virial[1] += vs[1];
    virial[2] += vs[2];
    virial[3] += vs[3];
    virial[4] += vs[5];
    virial[5] += vs[4];
  }

  if (eflag_atom) {
    // atomic_energy holds the short-range part only; spread the long-range
    // energy uniformly so that sum(eatom) equals the total energy
    torch::Tensor atomic_energy_tensor =
        output.at("atomic_energy").toTensor().to(torch::kCPU).view({nlocal});
    auto atomic_energy = atomic_energy_tensor.accessor<float, 1>();
    const float lr_per_atom =
        output.at("les_lr_energy").toTensor().to(torch::kCPU).item<float>() /
        nlocal;
    for (int gi = 0; gi < nlocal; gi++) {
      const int i = graph_index_to_i[gi];
      eatom[i] += atomic_energy[gi] + lr_per_atom;
    }
  }

  // if it was the first MD step
  if (this->nedges_bound == -1) {
    this->nedges_bound = nedges * 1.2;
  } // else if the nedges is too small, increase the bound
  else if (nedges > this->nedges_bound / 1.2) {
    this->nedges_bound = nedges * 1.2;
  }
}

// allocate arrays (called from coeff)
void PairE3GNNLES::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(map, n + 1, "pair:map");
}

// global settings for pair_style
void PairE3GNNLES::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command");
  }
}

void PairE3GNNLES::coeff(int narg, char **arg) {

  if (allocated) {
    error->all(FLERR, "pair_e3gnn_les coeff called twice");
  }
  allocate();

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR,
               "e3gnn/les: first and second input of pair_coeff should be '*'");
  }
  // expected input : pair_coeff * * pot.pth type_name1 type_name2 ...

  std::unordered_map<std::string, std::string> meta_dict = {
      {"chemical_symbols_to_index", ""},
      {"cutoff", ""},
      {"num_species", ""},
      {"model_type", ""},
      {"version", ""},
      {"dtype", ""},
      {"flashTP", "version mismatch"},
      {"oeq", "version mismatch"},
      {"les", "no"},
      {"time", ""}};

  // model loading from input
  try {
    model = torch::jit::load(std::string(arg[2]), device, meta_dict);
  } catch (const c10::Error &e) {
    error->all(FLERR, "error loading the model, check the path of the model");
  }

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy strategy;
  strategy = {{torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);
  cutoff_square = cutoff * cutoff;

  // to make torch::autograd::grad() works
  if (meta_dict["oeq"] == "yes") {
    pair_e3gnn_oeq_register_autograd();
  }

  if (meta_dict["model_type"].compare("E3_equivariant_model") != 0) {
    error->all(FLERR, "given model type is not E3_equivariant_model");
  }
  if (meta_dict["les"].compare("yes") != 0) {
    error->all(FLERR, "given deployed model is not an LES model; use "
                      "pair_style e3gnn instead");
  }

  std::string chem_str = meta_dict["chemical_symbols_to_index"];
  int ntypes = atom->ntypes;

  auto delim = " ";
  char *tok = std::strtok(const_cast<char *>(chem_str.c_str()), delim);
  std::vector<std::string> chem_vec;
  while (tok != nullptr) {
    chem_vec.push_back(std::string(tok));
    tok = std::strtok(nullptr, delim);
  }

  bool found_flag = false;
  for (int i = 3; i < narg; i++) {
    found_flag = false;
    for (int j = 0; j < chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i]) == 0) {
        map[i - 2] = j;
        found_flag = true;
        fprintf(lmp->logfile, "Chemical specie '%s' is assigned to type %d\n",
                arg[i], i - 2);
        break;
      }
    }
    if (!found_flag) {
      error->all(FLERR, "Unknown chemical specie is given");
    }
  }

  if (ntypes > narg - 3) {
    error->all(FLERR, "Not enough chemical specie is given. Check pair_coeff "
                      "and types in your data/script");
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = 1; j <= ntypes; j++) {
      if ((map[i] >= 0) && (map[j] >= 0)) {
        setflag[i][j] = 1;
        cutsq[i][j] = cutoff * cutoff;
      }
    }
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "from sevenn version '%s' ",
            meta_dict["version"].c_str());
    fprintf(lmp->logfile, "%s precision model, deployed: %s\n",
            meta_dict["dtype"].c_str(), meta_dict["time"].c_str());
    fprintf(lmp->logfile, "FlashTP: %s\n", meta_dict["flashTP"].c_str());
    fprintf(lmp->logfile, "OEQ: %s\n", meta_dict["oeq"].c_str());
    fprintf(lmp->logfile, "LES: %s\n", meta_dict["les"].c_str());
  }
}

// init specific to this pair
void PairE3GNNLES::init_style() {
  // the latent Ewald sum needs every atom of the cell on this rank
  if (comm->nprocs > 1) {
    error->all(FLERR, "Pair e3gnn/les runs on a single MPI rank only "
                      "(the Ewald sum needs the whole cell); use 1 rank or "
                      "pair e3gnn with a short-range model");
  }

  // full neighbor list (this is many-body potential)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNNLES::init_one(int i, int j) { return cutoff; }
