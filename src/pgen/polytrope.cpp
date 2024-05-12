//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file polytrope_simple.cpp
//! \brief Problem generator for n=1 polytrope, written by Joseph Weller
//!
//! REFERENCE: https://www.ucolick.org/~woosley/ay112-14/lectures/lecture7.14.pdf

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

namespace {
Real gconst;
Real m_refine;
LogicalLocation *loc_list;              /* List of logical locations of meshblocks */
int n_mb = 0; /* Number of meshblocks */
Real R_max,xvel,yvel,gam,gm1,gm_,kappa,T_W,da,pa,dc,x1_0,x2_0,x3_0,r_inner_boundary;
}  // namespace


int MassRefine(MeshBlock *pmb);
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
void inner_boundary_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar);

void Mesh::InitUserMeshData(ParameterInput *pin) {

   EnrollUserRadSourceFunction(inner_boundary_function);
  // gconst = pin->GetOrAddReal("problem", "grav_const", 1.0);
  // SetGravitationalConstant(gconst);
  // if (adaptive) {
  //   m_refine = pin->GetReal("problem","m_refine");
  //   EnrollUserRefinementCondition(MassRefine);
  // }
  return;
}

/*
Returns approximate cell sizes if the grid was uniform
*/

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ){

  if (COORDINATE_SYSTEM == "cartesian"){
    *DX = (box_size.x1max-box_size.x1min)/(1. * box_size.nx1);
    *DY = (box_size.x2max-box_size.x2min)/(1. * box_size.nx2);
    *DZ = (box_size.x3max-box_size.x3min)/(1. * box_size.nx3);
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    *DX = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DY = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DZ = (box_size.x3max-box_size.x3min)/(1. * box_size.nx3);

  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    *DX = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DY = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DZ = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
  }
}


void get_minimum_cell_lengths(const Coordinates * pcoord, Real *dx_min, Real *dy_min, Real *dz_min){
    
    //loc_list = pcoord->pmy_block->pmy_mesh->loclist; 
    RegionSize block_size;
    enum BoundaryFlag block_bcs[6];
    //int n_mb = pcoord->pmy_block->pmy_mesh->nbtotal;

    *dx_min = 1e15;
    *dy_min = 1e15;
    *dz_min = 1e15;

    Real DX,DY,DZ; 

    
    block_size = pcoord->pmy_block->block_size;

    
    /* Loop over all mesh blocks by reconstructing the block sizes to find the block that the
     star is located in */
    for (int j=0; j<n_mb; j++) {
        pcoord->pmy_block->pmy_mesh->SetBlockSizeAndBoundaries(loc_list[j], block_size, block_bcs);
        
        get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
        
        if (DX < *dx_min) *dx_min = DX;
        if (DY < *dy_min) *dy_min = DY;
        if (DZ < *dz_min) *dz_min = DZ;
        
    }
}
        
    
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    
    
    r_inner_boundary = 0.;
    loc_list = pmy_mesh->loclist;
    n_mb = pmy_mesh->nbtotal;

    int N_cells_per_boundary_radius = pin->GetOrAddInteger("problem", "boundary_radius", 2);

    Real dx_min,dy_min,dz_min;
    get_minimum_cell_lengths(pcoord, &dx_min, &dy_min, &dz_min);
    
    if (block_size.nx3>1)       r_inner_boundary = N_cells_per_boundary_radius * std::max(std::max(dx_min,dy_min),dz_min); // r_inner_boundary = 2*sqrt( SQR(dx_min) + SQR(dy_min) + SQR(dz_min) );
    else if (block_size.nx2>1)  r_inner_boundary = N_cells_per_boundary_radius * std::max(dx_min,dy_min); //2*sqrt( SQR(dx_min) + SQR(dy_min)               );
    else                        r_inner_boundary = N_cells_per_boundary_radius * dx_min;


    R_max = pin->GetReal("problem", "r_max"); // Polytrope max radius
    // Real dc = pin->GetOrAddReal("problem", "dcent", 1.0); //central density
    xvel = pin->GetOrAddReal("problem", "vx", 0.0); //x velocity of polytrope
    yvel = pin->GetOrAddReal("problem", "vy", 0.0); //y velocity of polytrope

    gam = pin->GetReal("hydro","gamma");
    gm1 = gam - 1.0;

    gm_ = pin->GetOrAddReal("problem","GM",0.0);

    kappa = pin->GetOrAddReal("problem","kappa",1.0);

    dc = pin->GetOrAddReal("problem", "dcent", 1.0);


    T_W = pin->GetOrAddReal("problem", "T_W", 0.08); //ratio of spin kinetic over grav potential energy (T/W)

    pa   = pin->GetOrAddReal("problem", "pamb", 1.0); //ambient pressure
    da   = pin->GetOrAddReal("problem", "damb", 1.0); //ambient density

    x1_0   = pin->GetOrAddReal("problem", "x1_0", 0.0);
    x2_0   = pin->GetOrAddReal("problem", "x2_0", 0.0);
    x3_0   = pin->GetOrAddReal("problem", "x3_0", 0.0);

    return;
    
    
}


void inner_boundary_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar)
{

  apply_inner_boundary_condition(pmb,prim);

  return;
}

/* Apply inner "inflow" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){


   // Prepare index bounds
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  int jl = pmb->js;
  int ju = pmb->je;
  if (pmb->block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmb->block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  Real r,x,y,z;
   for (int k=kl; k<=ku; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {

          Real x = pmb->pcoord->x1v(i);
          Real y = pmb->pcoord->x2v(j);
          Real z = pmb->pcoord->x3v(k);

          r = sqrt( SQR(x) + SQR(y) + SQR(z));

          if (r < r_inner_boundary){
              
            // if (r < R_max) {
              // den_pol = dc*R_max/(PI*rad)*std::sin((PI*rad)/R_max);

              Real den_pol = std::pow( gm1/gam * gm_/(kappa) * (1.0 /r - 1.0/R_max) , 1.0/gm1 );

              // pres_pol = pcent*std::pow(R_max/(PI*rad)*std::sin((PI*rad)/R_max),2);

              Real pres_pol = kappa * std::pow(den_pol,gam);
              
              //calculate x, y velocity using total body motion and spin term from T/W
              //spin intialized as solid body rotation around z axis

              Real spin = std::sqrt(3.0*T_W);  // used to initialize spin in terms of orb_vel
              Real orb_vel = std::sqrt(gm_/R_max);
              Real xvel_new = xvel - spin*orb_vel*(y)/R_max;
              Real yvel_new = yvel + spin*orb_vel*(x)/R_max;

              Real pcent = (2*gconst/PI)*dc*dc*R_max*R_max;
              Real M = (4.0/PI)*dc*R_max*R_max*R_max;

              Real den_pol = dc*R_max/(PI*rad)*std::sin((PI*rad)/R_max);

              Real pres_pol = pcent*std::pow(R_max/(PI*rad)*std::sin((PI*rad)/R_max),2);

            // }

              // fprintf(stderr,"den_pol: %g pre_pol: %g xvel_new: %g yvel_new: %g \n r: %g R_max: %g gm1: %g gam: %g kappa: %g gm_: %g \n",den_pol,pres_pol,xvel_new,yvel_new,r,R_max,gm1,gam,kappa,gm_);
              prim(IDN,k,j,i) = den_pol;
              prim(IVX,k,j,i) = xvel_new;
              prim(IVY,k,j,i) = yvel_new;
              prim(IVZ,k,j,i) = 0.;
              prim(IPR,k,j,i) = pres_pol;


              // /* Prevent outflow from inner boundary */ 

              
              
          }



}}}



}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief n=1 Polytrope body
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  
  Real pa   = 1e-8; //pin->GetOrAddReal("problem", "pamb", 1.0); //ambient pressure
  Real da   = 1e-5; //pin->GetOrAddReal("problem", "damb", 1.0); //ambient density

  
  // calculate central pressure, Mass, and orbital velocity at r=R_max

  // Real pcent = (2*gconst/PI)*dc*dc*R_max*R_max;
  // Real M = (4.0/PI)*dc*R_max*R_max*R_max;
  Real orb_vel = std::sqrt(gm_/R_max);

  // initial spin terms

  Real spin = std::sqrt(3.0*T_W);  // used to initialize spin in terms of orb_vel


  // get coordinates of center of polytrope

  Real x0, y0, z0;
 
  x0 = x1_0;
  y0 = x2_0;
  z0 = x3_0;


  Real pcent = (2*gconst/PI)*dc*dc*R_max*R_max;
  Real M = (4.0/PI)*dc*R_max*R_max*R_max;

  // setup polyrope density with pressure balancing gravity, initiate spin if T/W !=0
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad;
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else {
          std::stringstream msg;
          msg << "problem requires cartesian coordinates, initialization will be incorrect" << std::endl;
            throw std::runtime_error(msg.str().c_str());
        }
        
     //initialize variable at ambient values
        Real den = da;
        Real den_pol = 0.0;
        Real pres = pa;
        Real pres_pol = 0.0;
        Real momx = 0.0;
        Real momy = 0.0;
        Real kin  = 0.0;
          Real x = pcoord->x1v(i);
        Real y = pcoord->x2v(j);

          // when inside polytrope set density, pressure, momentum, and energy
        if (rad < R_max) {
          den_pol = dc*R_max/(PI*rad)*std::sin((PI*rad)/R_max);
          den = den_pol;

          pres_pol = pcent*std::pow(R_max/(PI*rad)*std::sin((PI*rad)/R_max),2);
          pres = pres_pol;
          
            //calculate x, y velocity using total body motion and spin term from T/W
            //spin intialized as solid body rotation around z axis
            Real xvel_new = xvel - spin*orb_vel*(y-y0)/R_max;
          Real yvel_new = yvel + spin*orb_vel*(x-x0)/R_max;

          momx = den*xvel_new;
          momy = den*yvel_new;
          kin = 0.5*den*xvel_new*xvel_new+0.5*den*yvel_new*yvel_new;
        }

        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = momx;
        phydro->u(IM2,k,j,i) = momy;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = (pres/gm1) + kin;
        
      }
    }
  }                              
             
}     


// AMR refinement condition
int MassRefine(MeshBlock *pmb) {
  Real mass  = 0.0;
  const Real dx = pmb->pcoord->dx1f(0);  // assuming uniform cubic cells
  const Real vol = dx*dx*dx;
  for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
    for (int j=pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
      for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
          // find mass of cell
        Real m_amount = vol*pmb->phydro->u(IDN,k,j,i);
        mass = std::max(mass, m_amount);
      }
    }
  }
  if (mass > m_refine)
    return 1;
  if (mass < m_refine * 0.1)
    return -1;
  return 0;
}