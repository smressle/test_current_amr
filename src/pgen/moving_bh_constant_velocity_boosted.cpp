//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_torus.cpp
//  \brief Problem generator for Fishbone-Moncrief torus.

// C++ headers
#include <algorithm>  // max(), max_element(), min(), min_element()
#include <cmath>      // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str(), string
#include <cfloat>
#include <stdio.h>
#include <random>



// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../globals.hpp"
// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
enum b_configs {vertical, normal, renorm, MAD};
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ghost);
void CustomInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar);
void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar);

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real ax, Real ay, Real az,Real *pr,
                                         Real *ptheta, Real *pphi);
// static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     // Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
// static void TransformAphi(Real a3_bl, Real x1,
                     // Real x2, Real x3, Real *pa1, Real *pa2, Real *pa3);
static Real CalculateLFromRPeak(Real r);
static Real CalculateRPeakFromL(Real l_target);
static Real LogHAux(Real r, Real sin_theta);
// static void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3);
// static void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
//                                            Real *pu1, Real *pu2, Real *pu3);
// static Real CalculateBetaMin();
// static bool CalculateBeta(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
//                           Real theta_p, Real phi_m, Real phi_c, Real phi_p, Real *pbeta);
// static bool CalculateBetaFromA(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
//               Real theta_p, Real a_cm, Real a_cp, Real a_mc, Real a_pc, Real *pbeta);
static Real CalculateMagneticPressure(Real bb1, Real bb2, Real bb3, Real r, Real theta,
                                      Real phi);

int RefinementCondition(MeshBlock *pmb);
void  Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);
void  Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);
bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv);


void get_prime_coords(Real x, Real y, Real z, AthenaArray<Real> &orbit_quantities,Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);
void get_bh_position(Real t, Real *xbh, Real *ybh, Real *zbh);
void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ);

void single_bh_metric(Real x1, Real x2, Real x3, ParameterInput *pin,AthenaArray<Real> &g);

Real DivergenceB(MeshBlock *pmb, int iout);
void BoostVector(Real t, Real a0, Real a1, Real a2, Real a3,AthenaArray<Real>&orbit_quantities, Real *pa0, Real *pa1, Real *pa2, Real *pa3);

void UpdateMetricFunction(Real metric_t, MeshBlock *pmb);

void convert_spherical_to_cartesian_ks(Real r, Real th, Real phi, Real ax, Real ay, Real az,
    Real *x, Real *y, Real *z);

void get_orbit_quantities(Real t, AthenaArray<Real>&orbit_quantities);
void interp_orbits(Real t, int iorbit, AthenaArray<Real> &arr, Real *result);



// Global variables
static Real m;                                  // black hole parameters
static Real gamma_adi, k_adi;                      // hydro parameters
static Real r_edge, r_peak, l, rho_max;            // fixed torus parameters
static Real psi, sin_psi, cos_psi;                 // tilt parameters
static Real log_h_edge, log_h_peak;                // calculated torus parameters
static Real pgas_over_rho_peak, rho_peak;          // more calculated torus parameters
static Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
static b_configs field_config;                     // type of magnetic field
static Real potential_cutoff;                      // sets region of torus to magnetize
static Real potential_r_pow, potential_rho_pow;    // set how vector potential scales
static Real beta_min;                              // min ratio of gas to mag pressure
static int sample_n_r, sample_n_theta;             // number of cells in 2D sample grid
static int sample_n_phi;                           // number of cells in 3D sample grid
static Real sample_r_rat;                          // sample grid geometric spacing ratio
static Real sample_cutoff;                         // density cutoff for sample grid
static Real x1_min, x1_max, x2_min, x2_max;        // 2D limits in chosen coordinates
static Real x3_min, x3_max;                        // 3D limits in chosen coordinates
static Real r_min, r_max, theta_min, theta_max;    // limits in r,theta for 2D samples
static Real phi_min, phi_max;                      // limits in phi for 3D samples
static Real pert_amp, pert_kr, pert_kz;            // parameters for initial perturbations
static Real dfloor,pfloor;                         // density and pressure floors

static Real q;          // black hole mass and spin
static Real v_bh2;
static Real Omega_bh2;
static Real eccentricity, tau, mean_angular_motion;
static Real rho0,press0;
static Real field_norm, r_cut;  


// Real rotation_matrix[3][3];


int IX1 = 0;
int IY1 = 1;
int IZ1 = 2;

int IX2 = 3;
int IY2 = 4;
int IZ2 = 5;


int IA1X = 6;
int IA1Y = 7;
int IA1Z = 8;

int IA2X = 9;
int IA2Y = 10;
int IA2Z = 11;


int IV1X = 12;
int IV1Y = 13;
int IV1Z = 14;

int IV2X = 15;
int IV2Y = 16;
int IV2Z = 17;

int Norbit = IV2Z - IX1+1;



int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
int max_second_bh_refinement_level = 0;  /*Maximum allowed level of refinement for AMR on secondary BH */
int max_smr_refinement_level = 0; /*Maximum allowed level of refinement for SMR on primary BH */

static Real SMALL = 1e-5;

//This function performs L * A = A_new 
void matrix_multiply_vector_lefthandside(const AthenaArray<Real> &L , const Real A[4], Real A_new[4]){

  A_new[0] = L(I00) * A[0] + L(I01)*A[1] + L(I02) * A[2] + L(I03) * A[3]; 
  A_new[1] = L(I01) * A[0] + L(I11)*A[1] + L(I12) * A[2] + L(I13) * A[3]; 
  A_new[2] = L(I02) * A[0] + L(I12)*A[1] + L(I22) * A[2] + L(I23) * A[3]; 
  A_new[3] = L(I03) * A[0] + L(I13)*A[1] + L(I23) * A[2] + L(I33) * A[3]; 

}

//----------------------------------------------------------------------------------------
// Functions for calculating determinant
// Inputs:
//   g: array of covariant metric coefficients
//   a11,a12,a13,a21,a22,a23,a31,a32,a33: elements of matrix
//   a11,a12,a21,a22: elements of matrix
// Outputs:
//   returned value: determinant

static Real Determinant(const AthenaArray<Real> &g) {
  const Real &a11 = g(I00);
  const Real &a12 = g(I01);
  const Real &a13 = g(I02);
  const Real &a14 = g(I03);
  const Real &a21 = g(I01);
  const Real &a22 = g(I11);
  const Real &a23 = g(I12);
  const Real &a24 = g(I13);
  const Real &a31 = g(I02);
  const Real &a32 = g(I12);
  const Real &a33 = g(I22);
  const Real &a34 = g(I23);
  const Real &a41 = g(I03);
  const Real &a42 = g(I13);
  const Real &a43 = g(I23);
  const Real &a44 = g(I33);
  Real det = a11 * Determinant(a22, a23, a24, a32, a33, a34, a42, a43, a44)
           - a12 * Determinant(a21, a23, a24, a31, a33, a34, a41, a43, a44)
           + a13 * Determinant(a21, a22, a24, a31, a32, a34, a41, a42, a44)
           - a14 * Determinant(a21, a22, a23, a31, a32, a33, a41, a42, a43);
  return det;
}

static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33) {
  Real det = a11 * Determinant(a22, a23, a32, a33)
           - a12 * Determinant(a21, a23, a31, a33)
           + a13 * Determinant(a21, a22, a31, a32);
  return det;
}

static Real Determinant(Real a11, Real a12, Real a21, Real a22) {
  return a11 * a22 - a12 * a21;
}


void Mesh::InitUserMeshData(ParameterInput *pin) {


  rho0 = 1.0;
  press0 = 1e-3;
  r_cut = 5.0;
  if (MAGNETIC_FIELDS_ENABLED) field_norm =  std::sqrt(1.0/5000.0); //pin->GetReal("problem", "field_norm");

  // Read problem-specific parameters from input file
  rho_min = pin->GetReal("hydro", "rho_min");
  rho_pow = pin->GetReal("hydro", "rho_pow");
  pgas_min = pin->GetReal("hydro", "pgas_min");
  pgas_pow = pin->GetReal("hydro", "pgas_pow");
  k_adi = pin->GetReal("problem", "k_adi");
  r_edge = pin->GetReal("problem", "r_edge");
  r_peak = pin->GetReal("problem", "r_peak");
  l = pin->GetReal("problem", "l");
  rho_max = pin->GetReal("problem", "rho_max");
  psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * PI/180.0;
  sin_psi = std::sin(psi);
  cos_psi = std::cos(psi);
  if (MAGNETIC_FIELDS_ENABLED) {
    std::string field_config_str = pin->GetString("problem",
                                                  "field_config");
    if (field_config_str == "vertical") {
      field_config = vertical;
    } else if (field_config_str == "normal") {
      field_config = normal;
    } else if (field_config_str == "renorm") {
      field_config = renorm;
    } else if (field_config_str == "MAD"){
      field_config = MAD;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in Problem Generator\n"
          << "unrecognized field_config="
          << field_config_str << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    if (field_config != vertical) {
      potential_cutoff = pin->GetReal("problem", "potential_cutoff");
      potential_r_pow = pin->GetReal("problem", "potential_r_pow");
      potential_rho_pow = pin->GetReal("problem", "potential_rho_pow");
    }
    beta_min = pin->GetReal("problem", "beta_min");
    sample_n_r = pin->GetInteger("problem", "sample_n_r");
    sample_n_theta = pin->GetInteger("problem", "sample_n_theta");
    if (psi != 0.0) {
      sample_n_phi = pin->GetInteger("problem", "sample_n_phi");
    } else {
      sample_n_phi = 1;
    }
    sample_r_rat = pin->GetReal("problem", "sample_r_rat");
    sample_cutoff = pin->GetReal("problem", "sample_cutoff");
    x1_min = pin->GetReal("mesh", "x1min");
    x1_max = pin->GetReal("mesh", "x1max");
    x2_min = pin->GetReal("mesh", "x2min");
    x2_max = pin->GetReal("mesh", "x2max");
    x3_min = pin->GetReal("mesh", "x3min");
    x3_max = pin->GetReal("mesh", "x3max");
  }
  pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  pert_kr = pin->GetOrAddReal("problem", "pert_kr", 0.0);
  pert_kz = pin->GetOrAddReal("problem", "pert_kz", 0.0);


  max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);

  max_second_bh_refinement_level = pin->GetOrAddReal("problem","max_bh2_refinement",0);
  max_smr_refinement_level = pin->GetOrAddReal("problem","max_smr_refinement",0);

  if (max_second_bh_refinement_level>max_refinement_level) max_second_bh_refinement_level = max_refinement_level;
  if (max_smr_refinement_level>max_refinement_level) max_smr_refinement_level = max_refinement_level;



  if (max_refinement_level>0) max_refinement_level = max_refinement_level -1;
  if (max_second_bh_refinement_level>0) max_second_bh_refinement_level = max_second_bh_refinement_level -1;
  if (max_smr_refinement_level>0) max_smr_refinement_level = max_smr_refinement_level - 1;

  // // Enroll boundary functions
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, CustomInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, CustomOuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, CustomOuterX2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, CustomInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, CustomOuterX3);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, CustomInnerX3);

  // EnrollUserBoundaryFunction(BoundaryFace::inner_x1, FixedBoundary);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x2, FixedBoundary);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x2, FixedBoundary);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x3, FixedBoundary);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x3, FixedBoundary);


  v_bh2 = pin->GetOrAddReal("problem", "vbh", 0.05);
    //Enroll metric
  EnrollUserMetric(Cartesian_GR);

  EnrollUserRadSourceFunction(inner_boundary_source_function);

  EnrollUserMetricWithoutPin(Binary_BH_Metric);

  if (MAGNETIC_FIELDS_ENABLED) AllocateUserHistoryOutput(1);

  if (MAGNETIC_FIELDS_ENABLED) EnrollUserHistoryOutput(0, DivergenceB, "divB");


  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);
  return;
}



    static Real exp_cut_off(Real r){

      Real rh2 = 2.0*q;

      if (r<=rh2) return 0.0;
      else if (r<= r_cut) return std::exp(5 * (r-r_cut)/r);
      else return 1.0;
    }

    static Real Ax_func(Real x,Real y, Real z){

      return (z  ) * field_norm;  //x 
    }
    static Real Ay_func(Real x, Real y, Real z){
      return 0.0 * field_norm;  //x 
    }
    static Real Az_func(Real x, Real y, Real z){
      return 0.0 * field_norm;
    }


//----------------------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)
// Notes:
//   user arrays are metric and its inverse

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));

  // Get mass and spin of black hole
  m = pcoord->GetMass();
  // a = pcoord->GetSpin();
  q = pin->GetOrAddReal("problem", "q", 0.1);
  // aprime = q * pin->GetOrAddReal("problem", "a_bh2", 0.0);

  v_bh2 = pin->GetOrAddReal("problem", "vbh", 0.05);



  // rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  // r_inner_boundary = rh/2.0;


    // Get mass of black hole
  Real m2 = q;

  // rh2 =  ( m2 + std::sqrt( SQR(m2) - SQR(aprime)) );
  // r_inner_boundary_2 = rh2/2.0;

  int N_user_vars = 7;
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1+NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1+NGHOST);


  int ncells1 = block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
  if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);


  // ruser_meshblock_data[2].NewAthenaArray(ncells3,ncells2,ncells1);
  // ruser_meshblock_data[3].NewAthenaArray(ncells3,ncells2,ncells1);
  // ruser_meshblock_data[4].NewAthenaArray(ncells3,ncells2,ncells1);







  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));


  return;
}



int RefinementCondition(MeshBlock *pmb)
{
  int refine = 0;

    Real DX,DY,DZ;
    Real dx,dy,dz;
  get_uniform_box_spacing(pmb->pmy_mesh->mesh_size,&DX,&DY,&DZ);
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);


  Real total_box_radius = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2.0;
  Real bh2_focus_radius = 3.125*q;
  //Real bh2_focus_radius = 3.125*0.1;

  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  // if (current_level >=max_refinement_level) return 0;

  int any_in_refinement_region = 0;
  int any_at_current_level=0;


  int max_level_required = 0;


  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->time,orbit_quantities);


  // fprintf(stderr,"current level: %d max_refinement_level: %d max_smr_refinement: %d max_bh2_refinement: %d \n",current_level,max_refinement_level,max_smr_refinement_level,max_second_bh_refinement_level);
  //first loop: check if any part of block is within refinement levels for secondary black hole

  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {


          for (int n_level = 1; n_level<=max_second_bh_refinement_level; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(x,y,z, orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Real box_radius = bh2_focus_radius * std::pow(2.,max_second_bh_refinement_level - n_level)*0.9999;

        
            //           if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
            // fprintf(stderr,"current level (AMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
            // }
            if (xprime < box_radius && xprime > -box_radius && yprime < box_radius
              && yprime > -box_radius && zprime < box_radius && zprime > -box_radius ){
              if (n_level>max_level_required) max_level_required=n_level;
              any_in_refinement_region=1;

              if (current_level < n_level){
                // if (current_level==max_refinement_level){
                // Real xbh, ybh, zbh;
                // get_bh_position(pmb->pmy_mesh->time,&xbh,&ybh,&zbh);
                // fprintf(stderr,"x1 min max: %g %g x2 min max: %g %g x3 min max: %g %g \n bh position: %g %g %g \n current_level: %d n_level: %d \n box radius: %g \n", pmb->block_size.x1min,pmb->block_size.x1max,
                // pmb->block_size.x2min,pmb->block_size.x2max,pmb->block_size.x3min,pmb->block_size.x3max,xbh,ybh,zbh,current_level, n_level,box_radius);
                // }
                orbit_quantities.DeleteAthenaArray();
                  return  1;
              }
              if (current_level==n_level) any_at_current_level=1;
            }


          
          }

        }
      }
    }
      

  //second loop: check if any part of block is within refinement levels for primary black hole

  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
          
          for (int n_level = 1; n_level<=max_smr_refinement_level; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(x,y,z,orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

          

             // if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
             //   fprintf(stderr,"current level (SMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
             //    }
            if (x<box_radius && x > -box_radius && y<box_radius
              && y > -box_radius && z<box_radius && z > -box_radius ){


              if (n_level>max_level_required) max_level_required=n_level;
              any_in_refinement_region = 1;
              if (current_level < n_level){
                // if (current_level==max_refinement_level){
                // Real xbh, ybh, zbh;
                // get_bh_position(pmb->pmy_mesh->time,&xbh,&ybh,&zbh);
                // fprintf(stderr,"x1 min max: %g %g x2 min max: %g %g x3 min max: %g %g \n bh position: %g %g %g \n current_level: %d n_level: %d \n box radius: %g \n", pmb->block_size.x1min,pmb->block_size.x1max,
                // pmb->block_size.x2min,pmb->block_size.x2max,pmb->block_size.x3min,pmb->block_size.x3max,xbh,ybh,zbh,current_level, n_level,box_radius);
                // }

                  //fprintf(stderr,"current level: %d n_level: %d box_radius: %g \n xmin: %g ymin: %g zmin: %g xmax: %g ymax: %g zmax: %g\n",current_level,
                    //n_level,box_radius,pmb->block_size.x1min,pmb->block_size.x2min,pmb->block_size.x3min,pmb->block_size.x1max,pmb->block_size.x2max,pmb->block_size.x3max);
                  orbit_quantities.DeleteAthenaArray();
                  return  1;
              }
              if (current_level==n_level) any_at_current_level=1;
            }



          
          }

  }
 }
}

// if (current_level==max_refinement_level){
//     Real xbh, ybh, zbh;
//     get_bh_position(pmb->pmy_mesh->time,&xbh,&ybh,&zbh);
//     fprintf(stderr,"x1 min max: %g %g x2 min max: %g %g x3 min max: %g %g \n bh position: %g %g %g \n current_leve: %d max_level_required: %d \n", pmb->block_size.x1min,pmb->block_size.x1max,
//     pmb->block_size.x2min,pmb->block_size.x2max,pmb->block_size.x3min,pmb->block_size.x3max,xbh,ybh,zbh,current_level, max_level_required);
// }

orbit_quantities.DeleteAthenaArray();
if (current_level>max_level_required) return -1;
else if (current_level==max_level_required) return 0;
else return 1;
//if (any_in_refinement_region==0) return -1;
// if (any_at_current_level==1) return 0;
  // return -1;
}


//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   initializes Fishbone-Moncrief torus
//     sets both primitive and conserved variables
//   defines and enrolls fixed r- and theta-direction boundary conditions
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//              Fishbone 1977, ApJ 215 323 (F)
//   assumes x3 is axisymmetric direction

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }


  // Get ratio of specific heats
  gamma_adi = peos->GetGamma();



  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmy_mesh->metric_time,orbit_quantities);

  // Prepare scratch arrays
  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {

        Real rho = rho0;
        Real pgas = press0;

        Real denom = g(I00,i);


        Real xprime,yprime,zprime,rprime,Rprime;

        get_prime_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);


        Real ut,ux,uy,uz,uu1,uu2,uu3;

        if (rprime>r_cut){
            ut = std::sqrt(-1.0/denom);

            ux = 0.0;
            uy = 0.0;
            uz = 0.0;

            uu1 = ux - gi(I01,i) / gi(I00,i) * ut;
            uu2 = uy - gi(I02,i) / gi(I00,i) * ut;
            uu3 = uz - gi(I03,i) / gi(I00,i) * ut;
         }
         else{
          uu1 = 0.0;
          uu2 = 0.0;
          uu3 = 0.0;
        }




        if (rprime<=r_cut){
          rho = 0.0;
          pgas = 0.0;
        }

       //    Real beta_init = 5.0;
       //    Real B_const = 0.0;

       //  if (MAGNETIC_FIELDS_ENABLED)
       //      Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
       //               + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
       //               + g(I33,i)*uu3*uu3;
       //      Real gamma = std::sqrt(1.0 + tmp);
       //      // user_out_var(0,k,j,i) = gamma;

       //      // Calculate 4-velocity
       //      Real alpha = std::sqrt(-1.0/gi(I00,i));
       //      Real u0 = gamma/alpha;
       //      Real u1 = uu1 - alpha * gamma * gi(I01,i);
       //      Real u2 = uu2 - alpha * gamma * gi(I02,i);
       //      Real u3 = uu3 - alpha * gamma * gi(I03,i);
       //      Real u_0, u_1, u_2, u_3;

       //      pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

       //      // Calculate 4-magnetic field
       //      Real bb1 = 0.0;
       //      Real bb2 = 0.1;
       //      Real bb3 = 0.0;
       //      Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
       //              + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
       //              + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
       //              + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
       //      Real b1 = (bb1 + b0 * u1) / u0;
       //      Real b2 = (bb2 + b0 * u2) / u0;
       //      Real b3 = (bb3 + b0 * u3) / u0;
       //      Real b_0, b_1, b_2, b_3;
       //      pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

       //      // Calculate magnetic pressure
       //      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

       //      Real beta_act = pgas / b_sq * 2.0;

       //      B_const = std::sqrt(beta_act/beta_init*b_sq);
       // }


        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IVX,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
        phydro->w(IVY,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
        phydro->w(IVZ,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }
    }
  }

  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();

  AthenaArray<Real> &g_ = ruser_meshblock_data[0];
  AthenaArray<Real> &gi_ = ruser_meshblock_data[1];


    // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {




    int ncells1 = block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
    if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);

    AthenaArray<Real> A3,A1,A2;

    A1.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );
    A2.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );
    A3.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );

      // Set B^1
      for (int k = kl; k <= ku+1; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          for (int i = il; i <= iu+1; ++i) {

            //A1 defined at cell center in x1 but face in x2 x3, 
            //A2 defined at cell center in x2 but face in x1 x3,
            //A3 defined at cell center in x3 but face in x1 x2

            Real rprime,xprime,zprime,yprime,Rprime;
            Real x_coord;
            if (i<= iu) x_coord = pcoord->x1v(i);
            else x_coord = pcoord->x1v(iu) + pcoord->dx1v(iu);

            get_prime_coords(x_coord,pcoord->x2f(j),pcoord->x3f(k), orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);

            Real Ax = Ax_func(xprime,yprime,zprime);
            Real Ay = Ay_func(xprime,yprime,zprime);
            Real Az = Az_func(xprime,yprime,zprime);

            // Real Ar,Ath,Aphi,A0;;


            A1(k,j,i) = Ax * exp_cut_off(rprime);

            Real y_coord;
            if (j<= ju) y_coord = pcoord->x2v(j);
            else y_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);

            get_prime_coords(pcoord->x1f(i),y_coord,pcoord->x3f(k), orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Ax = Ax_func(xprime,yprime,zprime) ;
            Ay = Ay_func(xprime,yprime,zprime) ;
            Az = Az_func(xprime,yprime,zprime) ;

            A2(k,j,i) = Ay * exp_cut_off(rprime);

            Real z_coord;
            if (k<= ku) z_coord = pcoord->x3v(k);
            else z_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
            get_prime_coords(pcoord->x1f(i),pcoord->x2f(j),z_coord, orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Ax = Ax_func(xprime,yprime,zprime) ;
            Ay = Ay_func(xprime,yprime,zprime) ;
            Az = Az_func(xprime,yprime,zprime) ;
            //TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A3(k,j,i) = Az * exp_cut_off(rprime);



            }
          }
        }


      // Initialize interface fields
    AthenaArray<Real> area;
    area.NewAthenaArray(ncells1+1);

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face2Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          pfield->b.x2f(k,j,i) = -1.0*(pcoord->dx3f(k)*A3(k,j,i+1) - pcoord->dx3f(k)*A3(k,j,i))/area(i);
          if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
          //if (j==ju) fprintf(stderr,"B: %g area: %g theta: %g j: %d A3: %g %g \n",pfield->b.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
           // A3(k,j,i+1), A3(k,j,i));

          if (std::isnan((pfield->b.x2f(k,j,i)))) fprintf(stderr,"isnan in bx2!\n");
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face3Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          pfield->b.x3f(k,j,i) = (pcoord->dx2f(j)*A2(k,j,i+1) - pcoord->dx2f(j)*A2(k,j,i))/area(i);
          //if (area(i)==0) pfield->b.x3f(k,j,i) = 0.0;

          if (std::isnan((pfield->b.x3f(k,j,i)))){

           fprintf(stderr,"isnan in bx3!\n A2: %g %g \n area: %g dx2f: %g \n", A2(k,j,i+1),A2(k,j,i),area(i),pcoord->dx2f(j));
           exit(0);
         }
        }
      }
    }

    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) = (pcoord->dx3f(k)*A3(k,j+1,i) - pcoord->dx3f(k)*A3(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x1f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x1f(k,j,i)))) fprintf(stderr,"isnan in bx1!\n");
          }
        }
      }
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face3Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            pfield->b.x3f(k,j,i) -= (pcoord->dx1f(i)*A1(k,j+1,i) - pcoord->dx1f(i)*A1(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x3f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x3f(k,j,i)))) {
              fprintf(stderr,"isnan in bx3!\n A1: %g %g \n area: %g dx1f: %g \n", A1(k,j+1,i),A1(k,j,i),area(i),pcoord->dx1f(i));
              exit(0);
            }
          }
        }
      }
    }
    // for 3D only
    if (block_size.nx3 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) -= (pcoord->dx2f(j)*A2(k+1,j,i) - pcoord->dx2f(j)*A2(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x1f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x1f(k,j,i)))) fprintf(stderr,"isnan in bx1!\n");
          }
        }
      }
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            pfield->b.x2f(k,j,i) += (pcoord->dx1f(i)*A1(k+1,j,i) - pcoord->dx1f(i)*A1(k,j,i))/area(i);
            if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
            if (std::isnan((pfield->b.x2f(k,j,i)))) fprintf(stderr,"isnan in bx2!\n");
            //if ( ju==je && j==je) fprintf(stderr,"B: %g area: %g theta: %g j: %d A1: %g %g \n",pfield->b.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
            //A1_bound(k+1,j,i), A1_bound(k,j,i));
          }
        }
      }
    }

    area.DeleteAthenaArray();
    A1.DeleteAthenaArray();
    A2.DeleteAthenaArray();
    A3.DeleteAthenaArray();

      // for (int k=kl; k<=ku+1; ++k) {
      //   for (int j=jl; j<=ju+1; ++j) {
      //     for (int i=il; i<=iu+1; ++i) {

      //       // fprintf(stderr,"ijk in loop: %d %d %d \n",i,j,k);

      //       if (j<ju+1 && k<ku+1) ruser_meshblock_data[2](k,j,i) = pfield->b.x1f(k,j,i) ;
      //       if (i<iu+1 && k<ku+1) ruser_meshblock_data[3](k,j,i) = pfield->b.x2f(k,j,i) ;
      //       if (i<iu+1 && j<ju+1) ruser_meshblock_data[4](k,j,i) = pfield->b.x3f(k,j,i) ;

      //     }
      //   }
      // }


  }


  // // Initialize magnetic fields
  // if (MAGNETIC_FIELDS_ENABLED) {

  //     // Set B^1
  //     for (int k = kl; k <= ku; ++k) {
  //       for (int j = jl; j <= ju; ++j) {
  //         for (int i = il; i <= iu+1; ++i) {
  //             pfield->b.x1f(k,j,i) = B_const;
  //           }
  //         }
  //       }
      
  //     // Set B^2
  //     for (int k = kl; k <= ku; ++k) {
  //       for (int j = jl; j <= ju+1; ++j) {
  //         for (int i = il; i <= iu; ++i) {
  //             pfield->b.x2f(k,j,i) = 0.0;
  //           }
  //         }
  //       }
      

  //     // Set B^3
  //     for (int k = kl; k <= ku+1; ++k) {
  //       for (int j = jl; j <= ju; ++j) {
  //         for (int i = il; i <= iu; ++i) {
  //             pfield->b.x3f(k,j,i) = 0.0;
  //           }
  //         }
  //       }
      
  //   }

  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  } else {
    bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  }

  // Initialize conserved values
  if (MAGNETIC_FIELDS_ENABLED) {
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
        kl, ku);
  } else {
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
    bb.DeleteAthenaArray();
  }

  // Call user work function to set output variables
  UserWorkInLoop();

  orbit_quantities.DeleteAthenaArray();
  return;
}


void  MeshBlock::PreserveDivbNewMetric(ParameterInput *pin){
  int SCALE_DIVERGENCE = false; //pin->GetOrAddBoolean("problem","scale_divergence",false);

  if (!SCALE_DIVERGENCE) return;
  fprintf(stderr,"Scaling divergence \n");


  AthenaArray<Real> &g = ruser_meshblock_data[0];
  AthenaArray<Real> &gi = ruser_meshblock_data[1];

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }


  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  AthenaArray<Real> b_old;

  // b_old.NewAthenaArray(3, ncells3, ncells2, ncells1);


  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);


  AthenaArray<Real> divb_old, face1rat,face2rat,face3rat; 
  face1rat.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);
  face2rat.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);
  face3rat.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);

  AthenaArray<Real> face1rat_used,face2rat_used,face3rat_used; 
  face1rat_used.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);
  face2rat_used.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);
  face3rat_used.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);
  divb_old.NewAthenaArray((ke-ks)+1+2*NGHOST,(je-js)+1+2*NGHOST,(ie-is)+1+2*NGHOST);
  Real divbmax_old = 0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {


        AthenaArray<Real> g_old1p;
        AthenaArray<Real> g_old1m;
        AthenaArray<Real> g_old2p;
        AthenaArray<Real> g_old2m;
        AthenaArray<Real> g_old3p;
        AthenaArray<Real> g_old3m;

        g_old1p.NewAthenaArray(NMETRIC);
        g_old2p.NewAthenaArray(NMETRIC);
        g_old3p.NewAthenaArray(NMETRIC);
        g_old1m.NewAthenaArray(NMETRIC);
        g_old2m.NewAthenaArray(NMETRIC);
        g_old3m.NewAthenaArray(NMETRIC);
        

        single_bh_metric(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old1m);
        single_bh_metric(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pin,g_old2m);
        single_bh_metric(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pin,g_old3m);

        single_bh_metric(pcoord->x1f(i+1), pcoord->x2v(j), pcoord->x3v(k), pin,g_old1p);
        single_bh_metric(pcoord->x1v(i), pcoord->x2f(j+1), pcoord->x3v(k), pin,g_old2p);
        single_bh_metric(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k+1), pin,g_old3p);

        Real det_old1m = Determinant(g_old1m);
        Real det_old2m = Determinant(g_old2m);
        Real det_old3m = Determinant(g_old3m);
        Real det_old1p = Determinant(g_old1p);
        Real det_old2p = Determinant(g_old2p);
        Real det_old3p = Determinant(g_old3p);


        Real face1m_ = std::sqrt(-det_old1m) * pcoord->dx2f(j) * pcoord->dx3f(k);
        Real face1p_ = std::sqrt(-det_old1p) * pcoord->dx2f(j) * pcoord->dx3f(k);
        Real face2m_ = std::sqrt(-det_old2m) * pcoord->dx1f(i) * pcoord->dx3f(k);
        Real face2p_ = std::sqrt(-det_old2p) * pcoord->dx1f(i) * pcoord->dx3f(k);
        Real face3m_ = std::sqrt(-det_old3m) * pcoord->dx1f(i) * pcoord->dx2f(j);
        Real face3p_ = std::sqrt(-det_old3p) * pcoord->dx1f(i) * pcoord->dx2f(j);

        face1rat(k,j,i) = face1m_/face1(i);
        face2rat(k,j,i) = face2m_/face2m(i);
        face3rat(k,j,i) = face3m_/face3m(i);





        divb_old(k,j,i)=(face1p_*pfield->b.x1f(k,j,i+1)-face1m_*pfield->b.x1f(k,j,i)
                        +face2p_*pfield->b.x2f(k,j+1,i)-face2m_*pfield->b.x2f(k,j,i)
                        +face3p_*pfield->b.x3f(k+1,j,i)-face3m_*pfield->b.x3f(k,j,i));
        if (divbmax_old<std::abs(divb_old(k,j,i))) divbmax_old = std::abs(divb_old(k,j,i));

        g_old1m.DeleteAthenaArray();
        g_old1p.DeleteAthenaArray();
        g_old2m.DeleteAthenaArray();
        g_old2p.DeleteAthenaArray();
        g_old3m.DeleteAthenaArray();
        g_old3p.DeleteAthenaArray();

        }
      }
    }



for (int dir=0; dir<=2; ++dir){
  int dk = 0;
  int dj = 0;
  int di = 0;

  if (dir==0) di = 1;
  if (dir==1) dj = 1;
  if (dir==2) dk = 1;

   for (int k=kl; k<=ku+dk; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=jl; j<=ju+dj; ++j) {
      if (dir==0) pcoord->Face1Metric(k, j, il, iu+di,g, gi);
      if (dir==1) pcoord->Face2Metric(k, j, il, iu+di,g, gi);
      if (dir==2) pcoord->Face3Metric(k, j, il, iu+di,g, gi);

      if (dir==0) pcoord->Face1Area(k,   j,   il, iu, face1);
      if (dir==1) pcoord->Face2Area(k,   j,   il, iu+di,   face2m);
      if (dir==2) pcoord->Face3Area(k,   j,   il, iu+di,   face3m);
// #pragma simd
      for (int i=il; i<=iu+di; ++i) {

        // Prepare scratch arrays
        AthenaArray<Real> g_tmp,g_old;
        g_tmp.NewAthenaArray(NMETRIC);
        g_old.NewAthenaArray(NMETRIC);
        g_tmp(I00) = g(I00,i);
        g_tmp(I01) = g(I01,i);
        g_tmp(I02) = g(I02,i);
        g_tmp(I03) = g(I03,i);
        g_tmp(I11) = g(I11,i);
        g_tmp(I12) = g(I12,i);
        g_tmp(I13) = g(I13,i);
        g_tmp(I22) = g(I22,i);
        g_tmp(I23) = g(I23,i);
        g_tmp(I33) = g(I33,i);

        Real det_new = Determinant(g_tmp);

        if (dir==0) single_bh_metric(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old);
        if (dir==1) single_bh_metric(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pin,g_old);
        if (dir==2) single_bh_metric(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pin,g_old);


        Real det_old = Determinant(g_old);

        // if (dir==0 and i<=iu){
        //  if (std::sqrt(-det_new) != face1(i)/(pcoord->dx2f(j)*pcoord->dx3f(k))){
        //     fprintf(stderr,"determinants don't match DIR 0!! %g %g \n",std::sqrt(-det_new),face1(i)/(pcoord->dx2f(j)*pcoord->dx3f(k)));
        //   }
        // }
        // if (dir==1){
        //   if (std::sqrt(-det_new) != face2m(i)/(pcoord->dx1f(i)*pcoord->dx3f(k))){
        //     fprintf(stderr,"determinants don't match DIR 1!! %g %g \n",std::sqrt(-det_new),face2m(i)/(pcoord->dx1f(i)*pcoord->dx3f(k)));
        //   }
        // }
        // if (dir==2){
        //   if (std::sqrt(-det_new) != face3m(i)/(pcoord->dx1f(i)*pcoord->dx2f(j))){
        //     fprintf(stderr,"determinants don't match DIR 2!! %g %g \n",std::sqrt(-det_new),face3m(i)/(pcoord->dx2f(j)*pcoord->dx3f(k)));
        //   }
        // }

        if (dir==0) pfield->b.x1f(k,j,i) *= std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==1) pfield->b.x2f(k,j,i) *= std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==2) pfield->b.x3f(k,j,i) *= std::sqrt(-det_old)/std::sqrt(-det_new);


        if (dir==0 && i>=is && i<=ie  && j<=je && j>=js && k<=ke && k>=ks) face1rat_used(k,j,i) = std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==1 && i>=is && i<=ie  && j<=je && j>=js && k<=ke && k>=ks) face2rat_used(k,j,i) = std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==2 && i>=is && i<=ie  && j<=je && j>=js && k<=ke && k>=ks) face3rat_used(k,j,i) = std::sqrt(-det_old)/std::sqrt(-det_new);

        // if (dir==0) pfield->b.x1f(k,j,i) *= 1.0/std::sqrt(-det_new);
        // if (dir==1) pfield->b.x2f(k,j,i) *= 1.0/std::sqrt(-det_new);
        // if (dir==2) pfield->b.x3f(k,j,i) *= 1.0/std::sqrt(-det_new);


        g_tmp.DeleteAthenaArray();
        g_old.DeleteAthenaArray();

      }
    }
  }
}

//   AthenaArray<Real> face1, face2p, face2m, face3p, face3m;

//   face1.NewAthenaArray((ie-is)+2*NGHOST+2);
//   face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
//   face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
//   face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
//   face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

//    for (int k=kl; k<=ku; ++k) {
// #pragma omp parallel for schedule(static)
//     for (int j=jl; j<=ju; ++j) {
//       pcoord->Face1Area(k,   j,   il, iu+1, face1);
//       for (int i=il; i<=iu+1; ++i) {

//         // Prepare scratch arrays
//         AthenaArray<Real> g_old;
//         g_old.NewAthenaArray(NMETRIC);


//         Real sqrt_minus_det_new = face1(i)/(pcoord->dx2f(j)*pcoord->dx3f(k));

//         single_bh_metric(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old);

//         Real det_old = -1.0; //Determinant(g_old);

//         pfield->b.x1f(k,j,i) *= std::sqrt(-det_old)/sqrt_minus_det_new;


//         g_old.DeleteAthenaArray();

//       }
//     }
//   }

//     for (int k=kl; k<=ku; ++k) {
// #pragma omp parallel for schedule(static)
//     for (int j=jl; j<=ju+1; ++j) {
//       pcoord->Face2Area(k,   j,   il, iu, face2m);
//       for (int i=il; i<=iu; ++i) {

//         // Prepare scratch arrays
//         AthenaArray<Real> g_old;
//         g_old.NewAthenaArray(NMETRIC);


//         Real sqrt_minus_det_new = face2m(i)/(pcoord->dx1f(i)*pcoord->dx3f(k));

//         single_bh_metric(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pin,g_old);

//         Real det_old = -1.0; //Determinant(g_old);

//         pfield->b.x2f(k,j,i) *= std::sqrt(-det_old)/sqrt_minus_det_new;


//         g_old.DeleteAthenaArray();

//       }
//     }
//   }

//     for (int k=kl; k<=ku+1; ++k) {
// #pragma omp parallel for schedule(static)
//     for (int j=jl; j<=ju; ++j) {
//       pcoord->Face3Area(k,   j,   il, iu, face3m);
//       for (int i=il; i<=iu; ++i) {

//         // Prepare scratch arrays
//         AthenaArray<Real> g_old;
//         g_old.NewAthenaArray(NMETRIC);


//         Real sqrt_minus_det_new = face3m(i)/(pcoord->dx1f(i)*pcoord->dx2f(j));

//         single_bh_metric(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pin,g_old);

//         Real det_old = -1.0; //Determinant(g_old);

//         pfield->b.x3f(k,j,i) *= std::sqrt(-det_old)/sqrt_minus_det_new;


//         g_old.DeleteAthenaArray();

//       }
//     }
//   }

   for (int k=kl; k<=ku; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=jl; j<=ju; ++j) {
      pcoord->CellMetric(k, j, il, iu,g, gi);
#pragma simd
      for (int i=il; i<=iu; ++i) {

                // Prepare scratch arrays
        AthenaArray<Real> g_tmp,g_old;
        g_tmp.NewAthenaArray(NMETRIC);
        g_old.NewAthenaArray(NMETRIC);
        g_tmp(I00) = g(I00,i);
        g_tmp(I01) = g(I01,i);
        g_tmp(I02) = g(I02,i);
        g_tmp(I03) = g(I03,i);
        g_tmp(I11) = g(I11,i);
        g_tmp(I12) = g(I12,i);
        g_tmp(I13) = g(I13,i);
        g_tmp(I22) = g(I22,i);
        g_tmp(I23) = g(I23,i);
        g_tmp(I33) = g(I33,i);

        Real det_new = Determinant(g_tmp);

        single_bh_metric(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old);

        Real det_old = Determinant(g_old);

         Real fac = std::sqrt(-det_old)/std::sqrt(-det_new);
          for (int n_cons=IDN; n_cons<= IEN; ++n_cons){
            phydro->u(n_cons,k,j,i) *=fac;
          }

        g_tmp.DeleteAthenaArray();
        g_old.DeleteAthenaArray();

      }
    }
  }


  Real divb,divbmax;
  divbmax=0;
  // AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pfield->b;

  // face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  // face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  // face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  // face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  // face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {
        divb=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
        if (divbmax<std::abs(divb)) divbmax = std::abs(divb);

        // if (i<=ie-1 && j<=je-1 && k<=ke-1)fprintf(stderr,"PreserveDivbNewMetric ijk: %d %d %d \n divb divb_old: %g %g \n face1rat: %g face1rat_used: %g \n face2: %g %g \n face3: %g %g \n face1p: %g %g\n face2p: %g %g \n face3p: %g %g \n",
        //   i,j,k,divb,divb_old(k,j,i),face1rat(k,j,i), face1rat_used(k,j,i), face2rat(k,j,i), face2rat_used(k,j,i),
        //   face3rat(k,j,i),face3rat_used(k,j,i),face1rat(k,j,i+1),face1rat_used(k,j,i+1),
        //   face2rat(k,j+1,i), face2rat_used(k,j+1,i), face3rat(k+1,j,i), face3rat_used(k+1,j,i));

        }
      }
    }

    //if (divbmax>1e-14) 
    //fprintf(stderr,"divbmax in PreserveDivbNewMetric vs. old:  %g %g \n",divbmax,divbmax_old);
  

  face1.DeleteAthenaArray();
  face2p.DeleteAthenaArray();
  face2m.DeleteAthenaArray();
  face3p.DeleteAthenaArray();
  face3m.DeleteAthenaArray();

  face1rat.DeleteAthenaArray();
  face2rat.DeleteAthenaArray();
  face3rat.DeleteAthenaArray();
  face1rat_used.DeleteAthenaArray();
  face2rat_used.DeleteAthenaArray();
  face3rat_used.DeleteAthenaArray();

  // b_old.DeleteAthenaArray();

  divb_old.DeleteAthenaArray();

  // // Calculate cell-centered magnetic field
  // AthenaArray<Real> bb;
  // if (MAGNETIC_FIELDS_ENABLED) {
  //   pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
  //       ku);
  // } else {
  //   bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  // }

  // // Initialize conserved values
  // if (MAGNETIC_FIELDS_ENABLED) {
  //   peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
  //       kl, ku);
  // } else {
  //   peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  //   bb.DeleteAthenaArray();
  // }


return;
}


void get_orbit_quantities(Real t, AthenaArray<Real>&orbit_quantities){



      orbit_quantities(IX1) = 0.0;
      orbit_quantities(IY1) = 0.0;
      orbit_quantities(IZ1) = 0.0;

      orbit_quantities(IX2) = 0.0;
      orbit_quantities(IY2) = 0.0;
      orbit_quantities(IZ2) = v_bh2 * (t) - 80.0;

      orbit_quantities(IA1X) = 0.0;
      orbit_quantities(IA1Y) = 0.0;
      orbit_quantities(IA1Z) = 0.0;

      orbit_quantities(IA2X) = 0.0;
      orbit_quantities(IA2Y) = 0.0;
      orbit_quantities(IA2Z) = 0.0;

      orbit_quantities(IV1X) = 0.0;
      orbit_quantities(IV1Y) = 0.0;
      orbit_quantities(IV1Z) = 0.0;

      orbit_quantities(IV2X) = 0.0;
      orbit_quantities(IV2Y) = 0.0;
      orbit_quantities(IV2Z) = v_bh2;

      return;


}




/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar){


  Real r,th,ph;
  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];



  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);

  Real a1x = orbit_quantities(IA1X);
  Real a1y = orbit_quantities(IA1Y);
  Real a1z = orbit_quantities(IA1Z);

  Real a2x = orbit_quantities(IA2X);
  Real a2y = orbit_quantities(IA2Y);
  Real a2z = orbit_quantities(IA2Z);

  Real a1 = std::sqrt( SQR(a1x) + SQR(a1y) + SQR(a1z) );
  Real a2 = std::sqrt( SQR(a2x) + SQR(a2y) + SQR(a2z) );

  Real rh =  ( m + std::sqrt( SQR(m) -SQR(a1)) );
  Real r_inner_boundary = rh*0.95;

  Real rh2 = ( q + std::sqrt( SQR(q) - SQR(a2)) );

   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {


         GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k),a1x,a1y,a1z, &r, &th, &ph);

          if (r < r_inner_boundary){
              

              //set uu assuming u is zero
              Real gamma = 1.0;
              Real alpha = std::sqrt(-1.0/gi(I00,i));
              Real u0 = gamma/alpha;
              Real uu1 = - gi(I01,i)/gi(I00,i) * u0;
              Real uu2 = - gi(I02,i)/gi(I00,i) * u0;
              Real uu3 = - gi(I03,i)/gi(I00,i) * u0;
              
              prim(IDN,k,j,i) = dfloor;
              prim(IVX,k,j,i) = 0.;
              prim(IVY,k,j,i) = 0.;
              prim(IVZ,k,j,i) = 0.;
              prim(IPR,k,j,i) = pfloor;
            
              
              
          }

          Real x = pmb->pcoord->x1v(i);
          Real y = pmb->pcoord->x2v(j);
          Real z = pmb->pcoord->x3v(k);
          Real t = pmb->pmy_mesh->time;

          Real xprime,yprime,zprime,rprime,Rprime;

          get_prime_coords(x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);


          if (rprime < rh2){

              Real bsq_over_rho_max = 1.0;
              Real beta_floor = 0.2;
              


              // Calculate normal frame Lorentz factor
              Real uu1 = 0.0;
              Real uu2 = 0.0;
              Real uu3 = 0.0;
              Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                       + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                       + g(I33,i)*uu3*uu3;
              Real gamma = std::sqrt(1.0 + tmp);

              // Calculate 4-velocity
              Real alpha = std::sqrt(-1.0/gi(I00,i));
              Real u0 = gamma/alpha;
              Real u1 = uu1 - alpha * gamma * gi(I01,i);
              Real u2 = uu2 - alpha * gamma * gi(I02,i);
              Real u3 = uu3 - alpha * gamma * gi(I03,i);



              Real v2x = orbit_quantities(IV2X);
              Real v2y = orbit_quantities(IV2Y);
              Real v2z = orbit_quantities(IV2Z);



              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(t,u0,u1,u2,u3, orbit_quantities,&u0prime,&u1prime,&u2prime,&u3prime);
              // Real u0prime = (u0 + v2x * u1 + v2y * u2 + v2z * u3);
              // Real u1prime = (u1 + v2x * u0);
              // Real u2prime = (u2 + v2y * u0);
              // Real u3prime = (u3 + v2z * u0);



              uu1 = u1prime - gi(I01,i) / gi(I00,i) * u0prime;
              uu2 = u2prime - gi(I02,i) / gi(I00,i) * u0prime;
              uu3 = u3prime - gi(I03,i) / gi(I00,i) * u0prime;

              
              prim(IDN,k,j,i) = dfloor;
              prim(IVX,k,j,i) = uu1;
              prim(IVY,k,j,i) = uu2;
              prim(IVZ,k,j,i) = uu3;
              prim(IPR,k,j,i) = pfloor;


              uu1 = prim(IVX,k,j,i);
              uu2 = prim(IVY,k,j,i);
              uu3 = prim(IVZ,k,j,i);
              tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                       + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                       + g(I33,i)*uu3*uu3;
              gamma = std::sqrt(1.0 + tmp);
              // user_out_var(0,k,j,i) = gamma;

              // Calculate 4-velocity
              alpha = std::sqrt(-1.0/gi(I00,i));
              u0 = gamma/alpha;
              u1 = uu1 - alpha * gamma * gi(I01,i);
              u2 = uu2 - alpha * gamma * gi(I02,i);
              u3 = uu3 - alpha * gamma * gi(I03,i);
              Real u_0, u_1, u_2, u_3;

              // user_out_var(1,k,j,i) = u0;
              // user_out_var(2,k,j,i) = u1;
              // user_out_var(3,k,j,i) = u2;
              // user_out_var(4,k,j,i) = u3;
              if (MAGNETIC_FIELDS_ENABLED) {
    

                pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

                // Calculate 4-magnetic field
                Real bb1 = pmb->pfield->bcc(IB1,k,j,i);
                Real bb2 = pmb->pfield->bcc(IB2,k,j,i);
                Real bb3 = pmb->pfield->bcc(IB3,k,j,i);
                Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
                        + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
                        + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
                        + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
                Real b1 = (bb1 + b0 * u1) / u0;
                Real b2 = (bb2 + b0 * u2) / u0;
                Real b3 = (bb3 + b0 * u3) / u0;
                Real b_0, b_1, b_2, b_3;
                pmb->pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

                // Calculate bsq
                Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

                if (b_sq/prim(IDN,k,j,i) > bsq_over_rho_max) prim(IDN,k,j,i) = b_sq/bsq_over_rho_max;
                if (prim(IPR,k,j,i)*2.0 < beta_floor*b_sq) prim(IPR,k,j,i) = beta_floor*b_sq/2.0;
            
              }
              
          }




}}}


orbit_quantities.DeleteAthenaArray();



}
void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half,AthenaArray<Real> &prim_scalar){

  int i, j, k, kprime;
  int is, ie, js, je, ks, ke;

     for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        pmb->user_out_var(0,k,j,i) = flux[X3DIR](IDN,k,j,i);
        pmb->user_out_var(1,k,j,i) = flux[X3DIR](IDN,k+1,j,i);

        pmb->user_out_var(2,k,j,i) = flux[X3DIR](IPR,k,j,i);
        pmb->user_out_var(3,k,j,i) = flux[X3DIR](IPR,k+1,j,i);


      }
    }
  }


  apply_inner_boundary_condition(pmb,prim,prim_scalar);

  return;
}



/* Store some useful variables like mdot and vr */

Real DivergenceB(MeshBlock *pmb, int iout)
{
  Real divb=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pmb->pfield->b;

  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pmb->pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pmb->pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pmb->pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pmb->pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {
        divb+=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
      }
    }
  }

  face1.DeleteAthenaArray();
  face2p.DeleteAthenaArray();
  face2m.DeleteAthenaArray();
  face3p.DeleteAthenaArray();
  face3m.DeleteAthenaArray();

  return divb;
}
//----------------------------------------------------------------------------------------
// Function responsible for storing useful quantities for output
// Inputs: (none)
// Outputs: (none)
// Notes:
//   writes to user_out_var array the following quantities:
//     0: gamma (normal-frame Lorentz factor)
//     1: p_mag (magnetic pressure)

void MeshBlock::UserWorkInLoop(void)
{
  // Create aliases for metric
  AthenaArray<Real> &g = ruser_meshblock_data[0];
  AthenaArray<Real> &gi = ruser_meshblock_data[1];

  // Real divb=0;
  // Real divbmax=0;
  // AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  // FaceField &b = pfield->b;

  // face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  // face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  // face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  // face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  // face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  // for(int k=ks; k<=ke; k++) {
  //   for(int j=js; j<=je; j++) {
  //     pcoord->Face1Area(k,   j,   is, ie+1, face1);
  //     pcoord->Face2Area(k,   j+1, is, ie,   face2p);
  //     pcoord->Face2Area(k,   j,   is, ie,   face2m);
  //     pcoord->Face3Area(k+1, j,   is, ie,   face3p);
  //     pcoord->Face3Area(k,   j,   is, ie,   face3m);
  //     for(int i=is; i<=ie; i++) {
  //       user_out_var(6,k,j,i)=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
  //             +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
  //             +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));

  //       if  (std::abs(user_out_var(6,k,j,i))>divbmax) divbmax = std::abs(user_out_var(6,k,j,i));
  //     }
  //   }
  // }


  // // if (divbmax>1e-14) fprintf(stderr,"Divbmax in Userwork: %g \n", divbmax);

  // face1.DeleteAthenaArray();
  // face2p.DeleteAthenaArray();
  // face2m.DeleteAthenaArray();
  // face3p.DeleteAthenaArray();
  // face3m.DeleteAthenaArray();

  return;
}

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ){

  if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "gr_user"){
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

//----------------------------------------------------------------------------------------
// Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {
  return;
}

//----------------------------------------------------------------------------------------
// Inflow boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones

void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // Set hydro variables
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        prim(IDN,k,j,i) = prim(IDN,k,j,is);
        prim(IEN,k,j,i) = prim(IEN,k,j,is);
        prim(IM1,k,j,i) = std::min(prim(IM1,k,j,is), static_cast<Real>(0.0));
        prim(IM2,k,j,i) = prim(IM2,k,j,is);
        prim(IM3,k,j,i) = prim(IM3,k,j,is);
      }
    }
  }
  if (not MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // Set radial magnetic field
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x1f(k,j,i) = bb.x1f(k,j,is);
      }
    }
  }

  // Set polar magnetic field
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je+1; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x2f(k,j,i) = bb.x2f(k,j,is);
      }
    }
  }

  // Set azimuthal magnetic field
  for (int k = ks; k <= ke+1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x3f(k,j,i) = bb.x3f(k,j,is);
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates
// Notes:
//   conversion is trivial in all currently implemented coordinate systems
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real ax, Real ay, Real az, Real *pr,
                                         Real *ptheta, Real *pphi) {

    Real x = x1;
    Real y = x2;
    Real z = x3;

    Real a = std::sqrt( SQR(ax) + SQR(ay) + SQR(az) );

    Real a_dot_x = ax * x + ay * y + az * z;

    Real a_cross_x[3];

    a_cross_x[0] = ay * z - az * y;
    a_cross_x[1] = az * x - ax * z;
    a_cross_x[2] = ax * y - ay * x;


    if ((std::fabs(a_dot_x)<SMALL) && (a_dot_x>=0)){

      Real diff = SMALL - a_dot_x/(a+SMALL);
      a_dot_x =  SMALL;

      x = 
      x = x + diff*ax/(a+SMALL); 
      y = y + diff*ay/(a+SMALL);
      z = z + diff*az/(a+SMALL);
    }
    if ((std::fabs(a_dot_x)<SMALL) && (a_dot_x <0)){

      Real diff = -SMALL - a_dot_x/(a+SMALL);;
      a_dot_x =  -SMALL;

      x = x + diff*ax/(a+SMALL);
      y = y + diff*ay/(a+SMALL);
      z = z + diff*az/(a+SMALL);
    } 



    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a_dot_x) )  )/std::sqrt(2.0);

    Real rsq_p_asq = SQR(r) + SQR(a);

    Real lx = (r * x - a_cross_x[0] + a_dot_x * ax/r)/(rsq_p_asq);
    Real ly = (r * y - a_cross_x[1] + a_dot_x * ay/r)/(rsq_p_asq);
    Real lz = (r * z - a_cross_x[2] + a_dot_x * az/r)/(rsq_p_asq);

    if (lz>1.0) lz = 1.0;
    if (lz<-1.0) lz = -1.0;
    *pr = r;
    *ptheta = std::acos(lz); //   std::acos(z/r);
    *pphi = std::atan2(ly,lx); //std::atan2( (r*y-a*x)/(SQR(r)+SQR(a) ), (a*y+r*x)/(SQR(r) + SQR(a) )  );

    // if (std::isnan(*pr) or std::isnan(*ptheta) or std::isnan(*pphi)){
    //   fprintf(stderr,"ISNAN in Get_prime_coords!!! \n xyz: %g %g %g \n ax ay az a: %g %g %g %g \n lx ly lz: %g %g %g \n adotx: %g a_cross_x: %g %g %g \n ",
    //     x,y,z,ax,ay,az,a,lx,ly,lz, a_dot_x,a_cross_x[0],a_cross_x[1],a_cross_x[2] );
    //   exit(0);
    // }
  return;
}
void convert_spherical_to_cartesian_ks(Real r, Real th, Real phi, Real ax, Real ay, Real az,
    Real *x, Real *y, Real *z){

  *x = r * std::sin(th) * std::cos(phi) + ay * std::cos(th)                 - az*std::sin(th) * std::sin(phi);
  *y = r * std::sin(th) * std::sin(phi) + az * std::sin(th) * std::cos(phi) - ax*std::cos(th)                ;
  *z = r * std::cos(th)                 + ax * std::sin(th) * std::sin(phi) - ay*std::sin(th) * std::cos(phi);

}


//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

// static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real x1,
//                      Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

//   if (COORDINATE_SYSTEM == "schwarzschild") {
//     *pa0 = a0_bl;
//     *pa1 = a1_bl;
//     *pa2 = a2_bl;
//     *pa3 = a3_bl;
//   } else if (COORDINATE_SYSTEM == "kerr-schild") {
//     Real r = x1;
//     Real delta = SQR(r) - 2.0*m*r + SQR(a);
//     *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
//     *pa1 = a1_bl;
//     *pa2 = a2_bl;
//     *pa3 = a3_bl + a/delta * a1_bl;
//   }
//     else if (COORDINATE_SYSTEM == "gr_user"){
//     Real x = x1;
//     Real y = x2;
//     Real z = x3;

//     Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
//     Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
//     Real delta = SQR(r) - 2.0*m*r + SQR(a);
//     *pa0 = a0_bl + 2.0*r/delta * a1_bl;
//     *pa1 = a1_bl * ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
//            a2_bl * x*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
//            a3_bl * y; 
//     *pa2 = a1_bl * ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
//            a2_bl * y*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
//            a3_bl * x;
//     *pa3 = a1_bl * z/r - 
//            a2_bl * r * std::sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
//   }
//   return;
// }

//Transform vector potential, A_\mu, from KS to CKS coordinates assuming A_r = A_theta = 0
// A_\mu (cks) = A_nu (ks)  dx^nu (ks)/dx^\mu (cks) = A_phi (ks) dphi (ks)/dx^\mu
// phi_ks = arctan((r*y + a*x)/(r*x - a*y) ) 
//
// static void TransformAphi(Real a3_ks, Real x1,
//                      Real x2, Real x3, Real *pa1, Real *pa2, Real *pa3) {

//   if (COORDINATE_SYSTEM == "gr_user"){
//     Real x = x1;
//     Real y = x2;
//     Real z = x3;

//     Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
//     Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
//     Real delta = SQR(r) - 2.0*m*r + SQR(a);
//     Real sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a);

//     //dphi/dx =  partial phi/partial x + partial phi/partial r partial r/partial x 
//     *pa1 = a3_ks * ( -y/(SQR(x)+SQR(y))  + a*x*r/( (SQR(a)+SQR(r))*sqrt_term ) ); 
//     //dphi/dx =  partial phi/partial y + partial phi/partial r partial r/partial y 
//     *pa2 = a3_ks * (  x/(SQR(x)+SQR(y))  + a*y*r/( (SQR(a)+SQR(r))*sqrt_term ) ); 
//     //dphi/dx =   partial phi/partial r partial r/partial z 
//     *pa3 = a3_ks * ( a*z/(r*sqrt_term) );
//   }
//   else{
//           std::stringstream msg;
//       msg << "### FATAL ERROR in TransformAphi\n"
//           << "this function only works for CKS coordinates"
//           <<  std::endl;
//     throw std::runtime_error(msg.str().c_str());
//   }
//   return;
// }


void get_prime_coords(Real x, Real y, Real z, AthenaArray<Real> &orbit_quantities, Real *xprime, Real *yprime, Real *zprime, Real *rprime, Real *Rprime){


  Real xbh = orbit_quantities(IX2);
  Real ybh = orbit_quantities(IY2);
  Real zbh = orbit_quantities(IZ2);


  Real ax = orbit_quantities(IA2X);
  Real ay = orbit_quantities(IA2Y);
  Real az = orbit_quantities(IA2Z);

  Real a_mag = std::sqrt( SQR(ax) + SQR(ay) + SQR(az) );

  Real vxbh = orbit_quantities(IV2X);
  Real vybh = orbit_quantities(IV2Y);
  Real vzbh = orbit_quantities(IV2Z);


  Real vsq = SQR(vxbh) + SQR(vybh) + SQR(vzbh);
  Real beta_mag = std::sqrt(vsq);
  Real Lorentz = std::sqrt(1.0/(1.0 - vsq));

  Real nx = vxbh/beta_mag;
  Real ny = vybh/beta_mag;
  Real nz = vzbh/beta_mag;

  *xprime = (1.0 + (Lorentz - 1.0) * nx * nx) * ( x - xbh ) + 
            (      (Lorentz - 1.0) * nx * ny) * ( y - ybh ) +
            (      (Lorentz - 1.0) * nx * nz) * ( z - zbh );
  
  *yprime = (      (Lorentz - 1.0) * ny * nx) * ( x - xbh ) + 
            (1.0 + (Lorentz - 1.0) * ny * ny) * ( y - ybh ) +
            (      (Lorentz - 1.0) * ny * nz) * ( z - zbh );  
 
  *zprime = (      (Lorentz - 1.0) * nz * nx) * ( x - xbh ) + 
            (      (Lorentz - 1.0) * nz * ny) * ( y - ybh ) +
            (1.0 + (Lorentz - 1.0) * nz * nz) * ( z - zbh );  


  Real a_dot_x_prime = ax * (*xprime) + ay * (*yprime) + az * (*zprime);

  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime>=0)){

    Real diff = SMALL - a_dot_x_prime/(a_mag+SMALL);
    a_dot_x_prime =  SMALL;

    *xprime = *xprime + diff*ax/(a_mag+SMALL);
    *yprime = *yprime + diff*ay/(a_mag+SMALL);
    *zprime = *zprime + diff*az/(a_mag+SMALL);;
  }
  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime <0)){

    Real diff = -SMALL - a_dot_x_prime/(a_mag+SMALL);;
    a_dot_x_prime =  -SMALL;

    *xprime = *xprime + diff*ax/(a_mag+SMALL);
    *yprime = *yprime + diff*ay/(a_mag+SMALL);
    *zprime = *zprime + diff*az/(a_mag+SMALL);
  } 

  // if (std::fabs(*zprime)<SMALL) *zprime= SMALL;
  *Rprime = std::sqrt(SQR(*xprime) + SQR(*yprime) + SQR(*zprime));
  *rprime = SQR(*Rprime) - SQR(a_mag) + std::sqrt( SQR( SQR(*Rprime) - SQR(a_mag) ) + 4.0*SQR(a_dot_x_prime) );
  *rprime = std::sqrt(*rprime/2.0);


  if (std::isnan(*rprime) or std::isnan(*xprime) or std::isnan(*yprime) or std::isnan(*zprime) ){
      fprintf(stderr,"ISNAN in GetBoyer!!! \n xyz: %g %g %g \n xbh ybh zbh: %g %g %g \n ax ay az a: %g %g %g %g \n adotx: %g \n xyzprime: %g %g %g \n vbh: %g %g %g \n ",
        x,y,z,xbh, ybh, zbh, ax,ay,az,a_mag, a_dot_x_prime,*xprime,*yprime,*zprime, vxbh,vybh,vzbh );
      exit(0);
    }

  return;

}



// void cks_metric(Real x1, Real x2, Real x3,AthenaArray<Real> &g){
//     // Extract inputs
//   Real x = x1;
//   Real y = x2;
//   Real z = x3;

//   Real a_spin = a; //-a;

//   if (std::fabs(z)<SMALL) z= SMALL;

//   if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
//     x=  0.1;
//     y = 0.1;
//     z = 0.1;
//   }
//   Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
//   Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
//   r = std::sqrt(r/2.0);


//   //if (r<0.01) r = 0.01;


//   Real eta[4],l_lower[4],l_upper[4];

//   Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
//   l_upper[0] = -1.0;
//   l_upper[1] = (r*x + a_spin*y)/( SQR(r) + SQR(a) );
//   l_upper[2] = (r*y - a_spin*x)/( SQR(r) + SQR(a) );
//   l_upper[3] = z/r;

//   l_lower[0] = 1.0;
//   l_lower[1] = l_upper[1];
//   l_lower[2] = l_upper[2];
//   l_lower[3] = l_upper[3];

//   eta[0] = -1.0;
//   eta[1] = 1.0;
//   eta[2] = 1.0;
//   eta[3] = 1.0;

//   // Set covariant components
//   g(I00) = eta[0] + f * l_lower[0]*l_lower[0];
//   g(I01) = f * l_lower[0]*l_lower[1];
//   g(I02) = f * l_lower[0]*l_lower[2];
//   g(I03) = f * l_lower[0]*l_lower[3];
//   g(I11) = eta[1] + f * l_lower[1]*l_lower[1];
//   g(I12) = f * l_lower[1]*l_lower[2];
//   g(I13) = f * l_lower[1]*l_lower[3];
//   g(I22) = eta[2] + f * l_lower[2]*l_lower[2];
//   g(I23) = f * l_lower[2]*l_lower[3];
//   g(I33) = eta[3] + f * l_lower[3]*l_lower[3];


// }

// void cks_inverse_metric(Real x1, Real x2, Real x3,AthenaArray<Real> &g_inv){
//     // Extract inputs
//   Real x = x1;
//   Real y = x2;
//   Real z = x3;

//   Real a_spin = a; //-a;

//   if (std::fabs(z)<SMALL) z= SMALL;

//   if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
//     x=  0.1;
//     y = 0.1;
//     z = 0.1;
//   }
//   Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
//   Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
//   r = std::sqrt(r/2.0);


//   //if (r<0.01) r = 0.01;


//   Real eta[4],l_lower[4],l_upper[4];

//   Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
//   l_upper[0] = -1.0;
//   l_upper[1] = (r*x + a_spin*y)/( SQR(r) + SQR(a) );
//   l_upper[2] = (r*y - a_spin*x)/( SQR(r) + SQR(a) );
//   l_upper[3] = z/r;

//   l_lower[0] = 1.0;
//   l_lower[1] = l_upper[1];
//   l_lower[2] = l_upper[2];
//   l_lower[3] = l_upper[3];

//   eta[0] = -1.0;
//   eta[1] = 1.0;
//   eta[2] = 1.0;
//   eta[3] = 1.0;
//     // // Set contravariant components
//   g_inv(I00) = eta[0] - f * l_upper[0]*l_upper[0] ;
//   g_inv(I01) =        - f * l_upper[0]*l_upper[1] ;
//   g_inv(I02) =        - f * l_upper[0]*l_upper[2] ;
//   g_inv(I03) =        - f * l_upper[0]*l_upper[3] ;
//   g_inv(I11) = eta[1] - f * l_upper[1]*l_upper[1] ;
//   g_inv(I12) =        - f * l_upper[1]*l_upper[2] ;
//   g_inv(I13) =        - f * l_upper[1]*l_upper[3] ;
//   g_inv(I22) = eta[2] - f * l_upper[2]*l_upper[2] ;
//   g_inv(I23) =        - f * l_upper[2]*l_upper[3] ;
//   g_inv(I33) = eta[3] - f * l_upper[3]*l_upper[3] ;


// }
// void delta_cks_metric(ParameterInput *pin,Real t, Real x1, Real x2, Real x3,AthenaArray<Real> &delta_g){
//   q = pin->GetOrAddReal("problem", "q", 0.1);
//   aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check

//   Real x = x1;
//   Real y = x2;
//   Real z = x3;


//   Real xprime,yprime,zprime,rprime,Rprime;
//   get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);


// /// prevent metric from getting nan sqrt(-gdet)
//   Real thprime  = std::acos(zprime/rprime);
//   Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
//                               (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

//   Real rhprime = ( q + std::sqrt(q*q-SQR(aprime)) );
//   if (rprime<rhprime/2.0) {
//     rprime = rhprime/2.0;
//     xprime = rprime * std::cos(phiprime)*std::sin(thprime) - aprime * std::sin(phiprime)*std::sin(thprime);
//     yprime = rprime * std::sin(phiprime)*std::sin(thprime) + aprime * std::cos(phiprime)*std::sin(thprime);
//     zprime = rprime * std::cos(thprime);
//   }



//   //if (r<0.01) r = 0.01;


//   Real l_lowerprime[4],l_upperprime[4];

//   Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));
//   l_upperprime[0] = -1.0;
//   l_upperprime[1] = (rprime*xprime + aprime*yprime)/( SQR(rprime) + SQR(aprime) );
//   l_upperprime[2] = (rprime*yprime - aprime*xprime)/( SQR(rprime) + SQR(aprime) );
//   l_upperprime[3] = zprime/rprime;

//   l_lowerprime[0] = 1.0;
//   l_lowerprime[1] = l_upperprime[1];
//   l_lowerprime[2] = l_upperprime[2];
//   l_lowerprime[3] = l_upperprime[3];






//   // Set covariant components
//   delta_g(I00) = fprime * l_lowerprime[0]*l_lowerprime[0];
//   delta_g(I01) = fprime * l_lowerprime[0]*l_lowerprime[1];
//   delta_g(I02) = fprime * l_lowerprime[0]*l_lowerprime[2];
//   delta_g(I03) = fprime * l_lowerprime[0]*l_lowerprime[3];
//   delta_g(I11) = fprime * l_lowerprime[1]*l_lowerprime[1];
//   delta_g(I12) = fprime * l_lowerprime[1]*l_lowerprime[2];
//   delta_g(I13) = fprime * l_lowerprime[1]*l_lowerprime[3];
//   delta_g(I22) = fprime * l_lowerprime[2]*l_lowerprime[2];
//   delta_g(I23) = fprime * l_lowerprime[2]*l_lowerprime[3];
//   delta_g(I33) = fprime * l_lowerprime[3]*l_lowerprime[3];

// }
// void delta_cks_metric_inverse(ParameterInput *pin,Real t, Real x1, Real x2, Real x3,AthenaArray<Real> &delta_g_inv){
//   Real q = pin->GetOrAddReal("problem", "q", 0.1);
//   Real aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check

//   Real x = x1;
//   Real y = x2;
//   Real z = x3;

//   if (std::fabs(z)<SMALL) z= SMALL;

//   if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
//     x=  0.1;
//     y = 0.1;
//     z = 0.1;
//   }

//  // Real t = 10000;
//     // Position of black hole

//   Real r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);
//   Real v_bh2 = 1.0/std::sqrt(r_bh2);
//   Real Omega_bh2 = v_bh2/r_bh2;
//   Real x_bh2 = 0.0;
//   Real y_bh2 = r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
//   Real z_bh2 = r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);

//   Real xprime = x - x_bh2;
//   Real yprime = y - y_bh2;
//   Real zprime = z - z_bh2;


//   Real dx_bh2_dt = 0.0;
//   Real dy_bh2_dt =  2.0*PI*Omega_bh2 * r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);
//   Real dz_bh2_dt = -2.0*PI*Omega_bh2 * r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
//   if (std::fabs(zprime)<SMALL) zprime= SMALL;
//   Real Rprime = std::sqrt(SQR(xprime) + SQR(yprime) + SQR(zprime));
//   Real rprime = SQR(Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(zprime) );
//   rprime = std::sqrt(rprime/2.0);



// /// prevent metric from gettin nan sqrt(-gdet)
//   Real thprime  = std::acos(zprime/rprime);
//   Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
//                               (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

//   Real rhprime = q * ( 1.0 + std::sqrt(1.0-SQR(aprime)) );
//   if (rprime<rhprime/2.0) {
//     rprime = rhprime/2.0;
//     xprime = rprime * std::cos(phiprime)*std::sin(thprime) - aprime * std::sin(phiprime)*std::sin(thprime);
//     yprime = rprime * std::sin(phiprime)*std::sin(thprime) + aprime * std::cos(phiprime)*std::sin(thprime);
//     zprime = rprime * std::cos(thprime);
//   }



//   //if (r<0.01) r = 0.01;


//   Real l_lowerprime[4],l_upperprime[4];

//   Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));
//   l_upperprime[0] = -1.0;
//   l_upperprime[1] = (rprime*xprime + aprime*yprime)/( SQR(rprime) + SQR(aprime) );
//   l_upperprime[2] = (rprime*yprime - aprime*xprime)/( SQR(rprime) + SQR(aprime) );
//   l_upperprime[3] = zprime/rprime;

//   l_lowerprime[0] = 1.0;
//   l_lowerprime[1] = l_upperprime[1];
//   l_lowerprime[2] = l_upperprime[2];
//   l_lowerprime[3] = l_upperprime[3];






//   // Set covariant components
//   delta_g_inv(I00) = -fprime * l_upperprime[0]*l_upperprime[0];
//   delta_g_inv(I01) = -fprime * l_upperprime[0]*l_upperprime[1];
//   delta_g_inv(I02) = -fprime * l_upperprime[0]*l_upperprime[2];
//   delta_g_inv(I03) = -fprime * l_upperprime[0]*l_upperprime[3];
//   delta_g_inv(I11) = -fprime * l_upperprime[1]*l_upperprime[1];
//   delta_g_inv(I12) = -fprime * l_upperprime[1]*l_upperprime[2];
//   delta_g_inv(I13) = -fprime * l_upperprime[1]*l_upperprime[3];
//   delta_g_inv(I22) = -fprime * l_upperprime[2]*l_upperprime[2];
//   delta_g_inv(I23) = -fprime * l_upperprime[2]*l_upperprime[3];
//   delta_g_inv(I33) = -fprime * l_upperprime[3]*l_upperprime[3];

// }

//From BHframe to lab frame

void BoostVector(Real t,Real a0, Real a1, Real a2, Real a3, AthenaArray<Real> &orbit_quantities, Real *pa0, Real *pa1, Real *pa2, Real *pa3){


  Real vxbh = orbit_quantities(IV2X);
  Real vybh = orbit_quantities(IV2Y);
  Real vzbh = orbit_quantities(IV2Z);



  Real vsq = SQR(vxbh) + SQR(vybh) + SQR(vzbh);
  Real beta_mag = std::sqrt(vsq);
  Real Lorentz = std::sqrt(1.0/(1.0 - vsq));

  Real nx = vxbh/beta_mag;
  Real ny = vybh/beta_mag;
  Real nz = vzbh/beta_mag;

  *pa0 =    Lorentz * (a0 + vxbh * a1 + vybh * a2 + vzbh * a3);

  *pa1 =                       Lorentz * vxbh * ( a0 ) +
            (1.0 + (Lorentz - 1.0) * nx * nx) * ( a1 ) + 
            (      (Lorentz - 1.0) * nx * ny) * ( a2 ) +
            (      (Lorentz - 1.0) * nx * nz) * ( a3 ) ;
  
  *pa2 =                       Lorentz * vybh * ( a0 ) +
            (      (Lorentz - 1.0) * ny * nx) * ( a1 ) + 
            (1.0 + (Lorentz - 1.0) * ny * ny) * ( a2 ) +
            (      (Lorentz - 1.0) * ny * nz) * ( a3 );  
 
  *pa3 =                       Lorentz * vzbh * ( a0 ) +
            (      (Lorentz - 1.0) * nz * nx) * ( a1 ) + 
            (      (Lorentz - 1.0) * nz * ny) * ( a2 ) +
            (1.0 + (Lorentz - 1.0) * nz * nz) * ( a3 );  

  return;

}


#define DEL 1e-7
void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{


  v_bh2 = pin->GetOrAddReal("problem", "vbh", 0.05);
  m = pin->GetReal("coord", "m");

  //////////////Perturber Black Hole//////////////////

  Binary_BH_Metric(t,x1,x2,x3,g,g_inv,dg_dx1,dg_dx2,dg_dx3,dg_dt);

  return;

}

void metric_for_derivatives(Real t, Real x1, Real x2, Real x3, AthenaArray<Real> &orbit_quantities,
    AthenaArray<Real> &g)
{

  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a1x = orbit_quantities(IA1X);
  Real a1y = orbit_quantities(IA1Y);
  Real a1z = orbit_quantities(IA1Z);

  Real a2x = orbit_quantities(IA2X);
  Real a2y = orbit_quantities(IA2Y);
  Real a2z = orbit_quantities(IA2Z);

  Real a1 = std::sqrt( SQR(a1x) + SQR(a1y) + SQR(a1z) );
  Real a2 = std::sqrt( SQR(a2x) + SQR(a2y) + SQR(a2z) );

  Real v1x = orbit_quantities(IV1X);
  Real v1y = orbit_quantities(IV1Y);
  Real v1z = orbit_quantities(IV1Z);

  Real v2x = orbit_quantities(IV2X);
  Real v2y = orbit_quantities(IV2Y);
  Real v2z = orbit_quantities(IV2Z);


  Real v1 = std::sqrt( SQR(v1x) + SQR(v1y) + SQR(v1z) );
  Real v2 = std::sqrt( SQR(v2x) + SQR(v2y) + SQR(v2z) );




  Real a_dot_x = a1x * x + a1y * y + a1z * z;

  if ((std::fabs(a_dot_x)<SMALL) && (a_dot_x>=0)){

    Real diff = SMALL - a_dot_x/(a1+SMALL);
    a_dot_x =  SMALL;

    x = x + diff*a1x/(a1+SMALL);
    y = y + diff*a1y/(a1+SMALL);
    z = z + diff*a1z/(a1+SMALL);
  }
  if ((std::fabs(a_dot_x)<SMALL) && (a_dot_x <0)){

    Real diff = -SMALL - a_dot_x/(a1+SMALL);
    a_dot_x =  -SMALL;

    x = x + diff*a1x/(a1+SMALL);
    y = y + diff*a1y/(a1+SMALL);
    z = z + diff*a1z/(a1+SMALL);
  } 


  // if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
  //   x = 0.1;
  //   y = 0.1;
  //   z = 0.1;
  // }

  Real r, th, phi;
  GetBoyerLindquistCoordinates(x,y,z,a1x,a1y,a1z, &r, &th, &phi);


/// prevent metric from getting nan sqrt(-gdet)

  Real rh =  ( m + std::sqrt(SQR(m)-SQR(a1)) );
  if (r<rh*0.5) {
    r = rh*0.5;
    convert_spherical_to_cartesian_ks(r,th,phi, a1x,a1y,a1z,&x,&y,&z);
  }

  //recompute after changes to coordinates
  a_dot_x = a1x * x + a1y * y + a1z * z;


  Real a_cross_x[3];

  a_cross_x[0] = a1y * z - a1z * y;
  a_cross_x[1] = a1z * x - a1x * z;
  a_cross_x[2] = a1x * y - a1y * x;


  Real rsq_p_asq = SQR(r) + SQR(a1);




  Real eta[4],l_lower[4],l_upper[4];

  Real f = 2.0 * m *  SQR(r)*r / (SQR(SQR(r)) + SQR(a_dot_x));
  l_upper[0] = -1.0;
  l_upper[1] = (r * x - a_cross_x[0] + a_dot_x * a1x/r)/(rsq_p_asq);
  l_upper[2] = (r * y - a_cross_x[1] + a_dot_x * a1y/r)/(rsq_p_asq);
  l_upper[3] = (r * z - a_cross_x[2] + a_dot_x * a1z/r)/(rsq_p_asq);

  l_lower[0] = 1.0;
  l_lower[1] = l_upper[1];
  l_lower[2] = l_upper[2];
  l_lower[3] = l_upper[3];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;


  //////////////Perturber Black Hole//////////////////


  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);

  Real a_dot_x_prime = a2x * xprime + a2y * yprime + a2z * zprime;

  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime>=0)){

    Real diff = SMALL - a_dot_x_prime/(a2+SMALL);
    a_dot_x_prime =  SMALL;

    xprime = xprime + diff*a2x/(a2+SMALL);
    yprime = yprime + diff*a2y/(a2+SMALL);
    zprime = zprime + diff*a2z/(a2+SMALL);
  }
  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime <0)){

    Real diff = -SMALL - a_dot_x_prime/(a2+SMALL);
    a_dot_x_prime =  -SMALL;

    xprime = xprime + diff*a2x/(a2+SMALL);
    yprime = yprime + diff*a2y/(a2+SMALL);
    zprime = zprime + diff*a2z/(a2+SMALL);
  } 
  
  Real thprime,phiprime;
  GetBoyerLindquistCoordinates(xprime,yprime,zprime,a2x,a2y,a2z, &rprime, &thprime, &phiprime);


/// prevent metric from getting nan sqrt(-gdet)

  Real rhprime = ( q + std::sqrt(SQR(q)-SQR(a2)) );
  if (rprime < rhprime*0.8) {
    rprime = rhprime*0.8;
    convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a2x,a2y,a2z,&xprime,&yprime,&zprime);
  }

  a_dot_x_prime = a2x * xprime + a2y * yprime + a2z * zprime;

  Real a_cross_x_prime[3];


  a_cross_x_prime[0] = a2y * zprime - a2z * yprime;
  a_cross_x_prime[1] = a2z * xprime - a2x * zprime;
  a_cross_x_prime[2] = a2x * yprime - a2y * xprime;


  Real rsq_p_asq_prime = SQR(rprime) + SQR(a2);

  //First calculated all quantities in BH rest (primed) frame

  Real l_lowerprime[4],l_upperprime[4];
  Real l_lowerprime_transformed[4];
  AthenaArray<Real> Lambda;

  Lambda.NewAthenaArray(NMETRIC);

  Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(a_dot_x_prime));
  l_upperprime[0] = -1.0;
  l_upperprime[1] = (rprime * xprime - a_cross_x_prime[0] + a_dot_x_prime * a2x/rprime)/(rsq_p_asq_prime);
  l_upperprime[2] = (rprime * yprime - a_cross_x_prime[1] + a_dot_x_prime * a2y/rprime)/(rsq_p_asq_prime);
  l_upperprime[3] = (rprime * zprime - a_cross_x_prime[2] + a_dot_x_prime * a2z/rprime)/(rsq_p_asq_prime);

  l_lowerprime[0] = 1.0;
  l_lowerprime[1] = l_upperprime[1];
  l_lowerprime[2] = l_upperprime[2];
  l_lowerprime[3] = l_upperprime[3];

  //Terms for the boost //

  Real vsq = SQR(v2x) + SQR(v2y) + SQR(v2z);
  Real beta_mag = std::sqrt(vsq);
  Real Lorentz = std::sqrt(1.0/(1.0 - vsq));
  ///Real Lorentz = 1.0;
  Real nx = v2x/beta_mag;
  Real ny = v2y/beta_mag;
  Real nz = v2z/beta_mag;


  // This is the inverse transformation since l_mu is lowered.  This 
  // takes a lowered vector from BH frame to lab frame.   
  Lambda(I00) =  Lorentz;
  Lambda(I01) = -Lorentz * v2x;
  Lambda(I02) = -Lorentz * v2y;
  Lambda(I03) = -Lorentz * v2z;
  Lambda(I11) = ( 1.0 + (Lorentz - 1.0) * nx * nx );
  Lambda(I12) = (       (Lorentz - 1.0) * nx * ny ); 
  Lambda(I13) = (       (Lorentz - 1.0) * nx * nz );
  Lambda(I22) = ( 1.0 + (Lorentz - 1.0) * ny * ny ); 
  Lambda(I23) = (       (Lorentz - 1.0) * ny * nz );
  Lambda(I33) = ( 1.0 + (Lorentz - 1.0) * nz * nz );




  // Boost l_mu
  matrix_multiply_vector_lefthandside(Lambda,l_lowerprime,l_lowerprime_transformed);


  // Set covariant components
  g(I00) = eta[0] + f * l_lower[0]*l_lower[0] + fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[0];
  g(I01) =          f * l_lower[0]*l_lower[1] + fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[1];
  g(I02) =          f * l_lower[0]*l_lower[2] + fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[2];
  g(I03) =          f * l_lower[0]*l_lower[3] + fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[3];
  g(I11) = eta[1] + f * l_lower[1]*l_lower[1] + fprime * l_lowerprime_transformed[1]*l_lowerprime_transformed[1];
  g(I12) =          f * l_lower[1]*l_lower[2] + fprime * l_lowerprime_transformed[1]*l_lowerprime_transformed[2];
  g(I13) =          f * l_lower[1]*l_lower[3] + fprime * l_lowerprime_transformed[1]*l_lowerprime_transformed[3];
  g(I22) = eta[2] + f * l_lower[2]*l_lower[2] + fprime * l_lowerprime_transformed[2]*l_lowerprime_transformed[2];
  g(I23) =          f * l_lower[2]*l_lower[3] + fprime * l_lowerprime_transformed[2]*l_lowerprime_transformed[3];
  g(I33) = eta[3] + f * l_lower[3]*l_lower[3] + fprime * l_lowerprime_transformed[3]*l_lowerprime_transformed[3];


  Lambda.DeleteAthenaArray();



  Real det = Determinant(g);
  if (det>=0 or std::isnan(det)){
    fprintf(stderr, "sqrt -g is nan!! xyz: %g %g %g xyzbh: %g %g %g \n xyzprime: %g %g %g \n r th phi: %g %g %g \n r th phi prime: %g %g %g \n",
      x,y,z,orbit_quantities(IX2),orbit_quantities(IY2),orbit_quantities(IZ2),
      xprime,yprime,zprime,r,th,phi,rprime,thprime,phiprime);
    exit(0);
  }



  // fprintf(stderr,"t: %g a1xyz: %g %g %g a1: %g \n a2xyz: %g %g %g a2: %g \n v1xyz: %g %g %g \n v2xyz: %g %g %g\n xx2 y2 z2: %g %g %g \n r th ph: %g %g %g \n rprime thprime phiprime: %g %g %g \n xprime yprime zprime: %g %g %g \n nt: %d q: %g t0: %g t0_orbits: %g dt_orbits: %g\n", 
  //   t, a1x,a1y,a1z,a1,a2x,a2y,a2z,a2,v1x,v1y,v1z,v2x,v2y,v2z, 
  //   orbit_quantities(IX2),orbit_quantities(IY2),orbit_quantities(IZ2),r,th,phi,rprime,thprime,phiprime,xprime,yprime,zprime,
  //   nt,q,t0,t0_orbits, dt_orbits);

  // for (int imetric=0; imetric<NMETRIC; imetric++){
  //   if (std::isnan(g(imetric))) {
  //     fprintf(stderr,"ISNAN in metric!!\n imetric: %d \n",imetric);
  //       fprintf(stderr,"t: %g a1xyz: %g %g %g a1: %g \n a2xyz: %g %g %g a2: %g \n v1xyz: %g %g %g \n v2xyz: %g %g %g\n xx2 y2 z2: %g %g %g \n r th ph: %g %g %g \n rprime thprime phiprime: %g %g %g \n xprime yprime zprime: %g %g %g \n q: %g \n xyz: %g %g %g \n", 
  //         t, a1x,a1y,a1z,a1,a2x,a2y,a2z,a2,v1x,v1y,v1z,v2x,v2y,v2z, 
  //   orbit_quantities(IX2),orbit_quantities(IY2),orbit_quantities(IZ2),r,th,phi,rprime,thprime,phiprime,xprime,yprime,zprime,
  //   q, x, y, z );
  //     exit(0);
  //   }
  // }

  return;
}



#define DEL 1e-7
void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{

  Real x = x1;
  Real y = x2;
  Real z = x3;

  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(t,orbit_quantities);

  metric_for_derivatives(t,x1,x2,x3,orbit_quantities,g);

  bool invertible = gluInvertMatrix(g,g_inv);

  if (invertible==false) {
    fprintf(stderr,"Non-invertible matrix at xyz: %g %g %g\n", x,y,z);
  }



  AthenaArray<Real> gp,gm;


  // Real det = Determinant(g);
  // if (det>=0){
  //   fprintf(stderr, "sqrt -g is nan!! xyz: %g %g %g xyzbh: %g %g %g \n",x,y,z,orbit_quantities(IX2),orbit_quantities(IY2),orbit_quantities(IZ2));
  //   exit(0);
  // }


  Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
  Real a1 = std::sqrt( SQR(orbit_quantities(IA1X)) + SQR(orbit_quantities(IA1Y)) + SQR(orbit_quantities(IA1Z)));
  Real a2 = std::sqrt( SQR(orbit_quantities(IA2X)) + SQR(orbit_quantities(IA2Y)) + SQR(orbit_quantities(IA2Z)));

  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(x,y,z,orbit_quantities,&xprime,&yprime,&zprime,&rprime,&Rprime);

  if (Rprime<=a2 or R<=a1){

    for (int n = 0; n < NMETRIC; ++n) {
         dg_dx1(n) = 0.0;
         dg_dx2(n) = 0.0;
         dg_dx3(n) = 0.0;
         dg_dt(n) = 0.0;
      }
    return;
  }

  gp.NewAthenaArray(NMETRIC);
  // gm.NewAthenaArray(NMETRIC);

  Real x1p = x1 + DEL; // * rprime;
  // Real x1m = x1 - DEL; // * rprime;
  Real x1m = x1;

  metric_for_derivatives(t,x1p,x2,x3,orbit_quantities,gp);
  // metric_for_derivatives(t,x1m,x2,x3,orbit_quantities,gm);

    // // Set x-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx1(n) = (gp(n)-gm(n))/(x1p-x1m);
  // }
    for (int n = 0; n < NMETRIC; ++n) {
     dg_dx1(n) = (gp(n)-g(n))/(x1p-x1m);
  }

  Real x2p = x2 + DEL; // * rprime;
  // Real x2m = x2 - DEL; // * rprime;
  Real x2m = x2;

  metric_for_derivatives(t,x1,x2p,x3,orbit_quantities,gp);
  // metric_for_derivatives(t,x1,x2m,x3,orbit_quantities,gm);
    // // Set y-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx2(n) = (gp(n)-gm(n))/(x2p-x2m);
  // }
  for (int n = 0; n < NMETRIC; ++n) {
     dg_dx2(n) = (gp(n)-g(n))/(x2p-x2m);
  }
  
  Real x3p = x3 + DEL; // * rprime;
  // Real x3m = x3 - DEL; // * rprime;
  Real x3m = x3;

  metric_for_derivatives(t,x1,x2,x3p,orbit_quantities,gp);
  // metric_for_derivatives(t,x1,x2,x3m,orbit_quantities,gm);

    // // Set z-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx3(n) = (gp(n)-gm(n))/(x3p-x3m);
  // }
    for (int n = 0; n < NMETRIC; ++n) {
     dg_dx3(n) = (gp(n)-g(n))/(x3p-x3m);
  }

  Real tp = t + DEL ;
  Real tm = t;
  // Real tm = t - DEL ;

  get_orbit_quantities(tp,orbit_quantities);
  metric_for_derivatives(tp,x1,x2,x3,orbit_quantities,gp);

  // get_orbit_quantities(tm,orbit_quantities);
  // metric_for_derivatives(tm,x1,x2,x3,orbit_quantities,gm);
    // // Set t-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dt(n) = (gp(n)-gm(n))/(tp-tm);
  // }
  for (int n = 0; n < NMETRIC; ++n) {
     dg_dt(n) = (gp(n)-g(n))/(tp-tm);
  }

  gp.DeleteAthenaArray();
  // gm.DeleteAthenaArray();

  orbit_quantities.DeleteAthenaArray();
  return;
}



//----------------------------------------------------------------------------------------
//! \fn void CustomInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void CustomInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = rho0;
        prim(IPR,k,j,is-i) = press0;
        prim(IVX,k,j,is-i) = 0.0;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;

      }
    }}
  

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = 0.0; //pmb->ruser_meshblock_data[2](k,j,(is-i));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = field_norm; //pmb->ruser_meshblock_data[3](k,j,(is-i));
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) = 0.0; //pmb->ruser_meshblock_data[4](k,j,(is-i));
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CustomOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void CustomOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i) = rho0;
        prim(IPR,k,j,ie+i) = press0;
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
      }
    }}


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = 0.0; //pmb->ruser_meshblock_data[2](k,j,(ie+i+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = field_norm; //pmb->ruser_meshblock_data[3](k,j,(ie+i));
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i)) = 0.0; //pmb->ruser_meshblock_data[4](k,j,(ie+i));
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CustomInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void CustomInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,k,js-j,i) = rho0;
        prim(IPR,k,js-j,i) = press0;
        prim(IVX,k,js-j,i) = 0.0;
        prim(IVY,k,js-j,i) = 0.0;
        prim(IVZ,k,js-j,i) = 0.0;
      }
    }}


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) = 0.0; //pmb->ruser_meshblock_data[2](k,(js-j),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = field_norm; // pmb->ruser_meshblock_data[3](k,(js-j),i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) = 0.0; //pmb->ruser_meshblock_data[4](k,(js-j),i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CustomOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void CustomOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,k,je+j,i) = rho0;
        prim(IPR,k,je+j,i) = press0;
        prim(IVX,k,je+j,i) = 0.0;
        prim(IVY,k,je+j,i) = 0.0;
        prim(IVZ,k,je+j,i) = 0.0;
      }
    }}


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) = 0.0; //pmb->ruser_meshblock_data[2](k,(je+j  ),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = field_norm; // pmb->ruser_meshblock_data[3](k,(je+j+1),i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) = 0.0; //pmb->ruser_meshblock_data[4](k,(je+j  ),i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CustomInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void CustomInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ks-k,j,i) = rho0;
        prim(IPR,ks-k,j,i) = press0;
        prim(IVX,ks-k,j,i) = 0.0;
        prim(IVY,ks-k,j,i) = 0.0;
        prim(IVZ,ks-k,j,i) = 0.0;
      }
    }}


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) = 0.0; //pmb->ruser_meshblock_data[2]((ks-k),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) = field_norm; // pmb->ruser_meshblock_data[3]((ks-k),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = 0.0; //pmb->ruser_meshblock_data[4]((ks-k),j,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CustomOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void CustomOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ke+k,j,i) = rho0;
        prim(IPR,ke+k,j,i) = press0;
        prim(IVX,ke+k,j,i) = 0.0;
        prim(IVY,ke+k,j,i) = 0.0;
        prim(IVZ,ke+k,j,i) = 0.0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) = 0.0; //pmb->ruser_meshblock_data[2]((ke+k  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) = field_norm; //pmb->ruser_meshblock_data[3]((ke+k  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = 0.0; //pmb->ruser_meshblock_data[4]((ke+k+1),j,i);
      }
    }}
  }

  return;
}

bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv)
{
    Real det;
    int i;

    inv(I00) = m(I11)  * m(I22) * m(I33) - 
             m(I11)  * m(I23) * m(I23) - 
             m(I12)  * m(I12)  * m(I33) + 
             m(I12)  * m(I13)  * m(I23) +
             m(I13) * m(I12)  * m(I23) - 
             m(I13) * m(I13)  * m(I22);

    inv(I01) = -m(I01)  * m(I22) * m(I33) + 
              m(I01)  * m(I23) * m(I23) + 
              m(I02)  * m(I12)  * m(I33) - 
              m(I02)  * m(I13)  * m(I23) - 
              m(I03) * m(I12)  * m(I23) + 
              m(I03) * m(I13)  * m(I22);


    inv(I02) = m(I01)  * m(I12) * m(I33) - 
             m(I01)  * m(I23) * m(I13) - 
             m(I02)  * m(I11) * m(I33) + 
             m(I02)  * m(I13) * m(I13) + 
             m(I03) * m(I11) * m(I23) - 
             m(I03) * m(I13) * m(I12);


    inv(I03) = -m(I01)  * m(I12) * m(I23) + 
               m(I01)  * m(I22) * m(I13) +
               m(I02)  * m(I11) * m(I23) - 
               m(I02)  * m(I12) * m(I13) - 
               m(I03) * m(I11) * m(I22) + 
               m(I03) * m(I12) * m(I12);


    inv(I11) = m(I00)  * m(I22) * m(I33) - 
             m(I00)  * m(I23) * m(I23) - 
             m(I02)  * m(I02) * m(I33) + 
             m(I02)  * m(I03) * m(I23) + 
             m(I03) * m(I02) * m(I23) - 
             m(I03) * m(I03) * m(I22);

    inv(I12) = -m(I00)  * m(I12) * m(I33) + 
              m(I00)  * m(I23) * m(I13) + 
              m(I02)  * m(I01) * m(I33) - 
              m(I02)  * m(I03) * m(I13) - 
              m(I03) * m(I01) * m(I23) + 
              m(I03) * m(I03) * m(I12);

    inv(I13) = m(I00)  * m(I12) * m(I23) - 
              m(I00)  * m(I22) * m(I13) - 
              m(I02)  * m(I01) * m(I23) + 
              m(I02)  * m(I02) * m(I13) + 
              m(I03) * m(I01) * m(I22) - 
              m(I03) * m(I02) * m(I12);

    inv(I22) = m(I00)  * m(I11) * m(I33) - 
              m(I00)  * m(I13) * m(I13) - 
              m(I01)  * m(I01) * m(I33) + 
              m(I01)  * m(I03) * m(I13) + 
              m(I03) * m(I01) * m(I13) - 
              m(I03) * m(I03) * m(I11);

    inv(I23) = -m(I00)  * m(I11) * m(I23) + 
               m(I00)  * m(I12) * m(I13) + 
               m(I01)  * m(I01) * m(I23) - 
               m(I01)  * m(I02) * m(I13) - 
               m(I03) * m(I01) * m(I12) + 
               m(I03) * m(I02) * m(I11);


    inv(I33) = m(I00) * m(I11) * m(I22) - 
              m(I00) * m(I12) * m(I12) - 
              m(I01) * m(I01) * m(I22) + 
              m(I01) * m(I02) * m(I12) + 
              m(I02) * m(I01) * m(I12) - 
              m(I02) * m(I02) * m(I11);

    det = m(I00) * inv(I00) + m(I01) * inv(I01) + m(I02) * inv(I02) + m(I03) * inv(I03);
    

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (int n = 0; n < NMETRIC; ++n) {        
      inv(n) = inv(n) * det;
    }

    return true;
}

// void UpdateMetricFunction(Real metric_t, MeshBlock *pmb)
// {
//   // Set object names
//   Mesh *pm = pmb>pmy_mesh;
//   RegionSize& block_size = pmb->block_size;
//   Coordinates *pco = pmb->pcoord; 

//   // Set indices
//   int il, iu, jl, ju, kl, ku, ng;
//   if (pco->coarse_flag) {
//     il = pmb->cis;
//     iu = pmb->cie;
//     jl = pmb->cjs;
//     ju = pmb->cje;
//     kl = pmb->cks;
//     ku = pmb->cke;
//     ng = pmb->cnghost;
//   } else {
//     il = pmb->is;
//     iu = pmb->ie;
//     jl = pmb->js;
//     ju = pmb->je;
//     kl = pmb->ks;
//     ku = pmb->ke;
//     ng = NGHOST;
//   }
//   int ill = il - ng;
//   int iuu = iu + ng;
//   int jll, juu;
//   if (block_size.nx2 > 1) {
//     jll = jl - ng;
//     juu = ju + ng;
//   } else {
//     jll = jl;
//     juu = ju;
//   }
//   int kll, kuu;
//   if (block_size.nx3 > 1) {
//     kll = kl - ng;
//     kuu = ku + ng;
//   } else {
//     kll = kl;
//     kuu = ku;
//   }

//   // Allocate arrays for volume-centered coordinates and positions of cells
//   int ncells1 = (iu-il+1) + 2*ng;
//   int ncells2 = 1, ncells3 = 1;
//   if (block_size.nx2 > 1) ncells2 = (ju-jl+1) + 2*ng;
//   if (block_size.nx3 > 1) ncells3 = (ku-kl+1) + 2*ng;


//   // Allocate scratch arrays
//   AthenaArray<Real> g, g_inv, dg_dx1, dg_dx2, dg_dx3, dg_dt,transformation;
//   g.NewAthenaArray(NMETRIC);
//   g_inv.NewAthenaArray(NMETRIC);
//   dg_dx1.NewAthenaArray(NMETRIC);
//   dg_dx2.NewAthenaArray(NMETRIC);
//   dg_dx3.NewAthenaArray(NMETRIC);
//   dg_dt.NewAthenaArray(NMETRIC);
//   if (not coarse_flag) {
//     transformation.NewAthenaArray(2, NTRIANGULAR);
//   }


//   // AthenaArray<Real> divb; 
//   int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
//   // AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
//   FaceField &b = pmb->pfield->b;


//   // Calculate cell-centered geometric quantities
//   for (int k = kll; k <= kuu; ++k) {
//     for (int j = jll; j <= juu; ++j) {
//       for (int i = ill; i <= iuu; ++i) {

//         // Get position and separations
//         Real x1 = pco->x1v(i);
//         Real x2 = pco->x2v(j);
//         Real x3 = pco->x3v(k);
//         Real dx1 = pco->dx1f(i);
//         Real dx2 = pco->dx2f(j);
//         Real dx3 = pco->dx3f(k);

//         // Calculate metric coefficients
//         Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//         // Calculate volumes
//         if (not coarse_flag or METRIC_EVOLUTION) {
//           Real det = Determinant(g);
//           pco->coord_vol_kji_(k,j,i) = std::sqrt(-det) * dx1 * dx2 * dx3;

//         }

//         // Calculate widths
//         if (not coarse_flag) {
//           pco->coord_width1_kji_(k,j,i) = std::sqrt(g(I11)) * dx1;
//           pco->coord_width2_kji_(k,j,i) = std::sqrt(g(I22)) * dx2;
//           pco->coord_width3_kji_(k,j,i) = std::sqrt(g(I33)) * dx3;
//         }

//         // Store metric derivatives
//         if (not coarse_flag) {
//           for (int m = 0; m < NMETRIC; ++m) {
//             pco->coord_src_kji_(0,m,k,j,i) = dg_dx1(m);
//             pco->coord_src_kji_(1,m,k,j,i) = dg_dx2(m);
//             pco->coord_src_kji_(2,m,k,j,i) = dg_dx3(m);
//             if (METRIC_EVOLUTION) pco->coord_src_kji_(3,m,k,j,i) = dg_dt(m);
//             else pco->coord_src_kji_(3,m,k,j,i) = 0.0;
//           }
//         }

//         // Set metric coefficients
//         for (int n = 0; n < NMETRIC; ++n) {
//           pco->metric_cell_kji_(0,n,k,j,i) = g(n);
//           pco->metric_cell_kji_(1,n,k,j,i) = g_inv(n);
//         }

//       }
//     }
//   }

//   // Calculate x1-face-centered geometric quantities
//   if (not coarse_flag ) {
//     for (int k = kll; k <= kuu; ++k) {
//       for (int j = jll; j <= juu; ++j) {
//         for (int i = ill; i <= iuu+1; ++i) {

//           // Get position and separations
//           Real x1 = pco->x1f(i);
//           Real x2 = pco->x2v(j);
//           Real x3 = pco->x3v(k);
//           Real dx2 = pco->dx2f(j);
//           Real dx3 = pco->dx3f(k);

//           // Calculate metric coefficients
//           pco->Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//           // Calculate areas
//           Real det = Determinant(g);
//           pco->coord_area1_kji_(k,j,i) = std::sqrt(-det) * dx2 * dx3;

//           // Set metric coefficients
//           for (int n = 0; n < NMETRIC; ++n) {
//             pco->metric_face1_kji_(0,n,k,j,i) = g(n);
//             pco->metric_face1_kji_(1,n,k,j,i) = g_inv(n);
//           }

//           // Calculate frame transformation
//           pco->CalculateTransformation(g, g_inv, 1, transformation);
//           for (int n = 0; n < 2; ++n) {
//             for (int m = 0; m < NTRIANGULAR; ++m) {
//               pco->trans_face1_kji_(n,m,k,j,i) = transformation(n,m);
//             }
//           }
//         }
//       }
//     }
//   }

//   // Calculate x2-face-centered geometric quantities
//   if (not coarse_flag) {
//     for (int k = kll; k <= kuu; ++k) {
//       for (int j = jll; j <= juu+1; ++j) {
//         for (int i = ill; i <= iuu; ++i) {

//           // Get position and separations
//           Real x1 = pco->x1v(i);
//           Real x2 = pco->x2f(j);
//           Real x3 = pco->x3v(k);
//           Real dx1 = pco->dx1f(i);
//           Real dx3 = pco->dx3f(k);

//           // Calculate metric coefficients
//           Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//           // Calculate areas
//           Real det = Determinant(g);
//           pco->coord_area2_kji_(k,j,i) = std::sqrt(-det) * dx1 * dx3;

//           // Set metric coefficients
//           for (int n = 0; n < NMETRIC; ++n) {
//             pco->metric_face2_kji_(0,n,k,j,i) = g(n);
//             pco->metric_face2_kji_(1,n,k,j,i) = g_inv(n);
//           }

//           // Calculate frame transformation
//           pco->CalculateTransformation(g, g_inv, 2, transformation);
//           for (int n = 0; n < 2; ++n) {
//             for (int m = 0; m < NTRIANGULAR; ++m) {
//               pco->trans_face2_kji_(n,m,k,j,i) = transformation(n,m);
//             }
//           }

//         }
//       }
//     }
//   }

//   // Calculate x3-face-centered geometric quantities
//   if (not coarse_flag) {
//     for (int k = kll; k <= kuu+1; ++k) {
//       for (int j = jll; j <= juu; ++j) {
//         for (int i = ill; i <= iuu; ++i) {

//           // Get position and separations
//           Real x1 = pco->x1v(i);
//           Real x2 = pco->x2v(j);
//           Real x3 = pco->x3f(k);
//           Real dx1 = pco->dx1f(i);
//           Real dx2 = pco->dx2f(j);

//           // Calculate metric coefficients
//           Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//           // Calculate areas
//           Real det = Determinant(g);
//           pco->coord_area3_kji_(k,j,i) = std::sqrt(-det) * dx1 * dx2;

//           // Set metric coefficients
//           for (int n = 0; n < NMETRIC; ++n) {
//             pco->metric_face3_kji_(0,n,k,j,i) = g(n);
//             pco->metric_face3_kji_(1,n,k,j,i) = g_inv(n);
//           }

//           // Calculate frame transformation
//           pco->CalculateTransformation(g, g_inv, 3, transformation);
//           for (int n = 0; n < 2; ++n) {
//             for (int m = 0; m < NTRIANGULAR; ++m) {
//               pco->trans_face3_kji_(n,m,k,j,i) = transformation(n,m);
//             }
//           }

//         }
//       }
//     }
//   }




//   // Calculate x1-edge-centered geometric quantities
//   if (not coarse_flag) {
//     for (int k = kll; k <= kuu+1; ++k) {
//       for (int j = jll; j <= juu+1; ++j) {
//         for (int i = ill; i <= iuu; ++i) {

//           // Get position and separation
//           Real x1 = pco->x1v(i);
//           Real x2 = pco->x2f(j);
//           Real x3 = pco->x3f(k);
//           Real dx1 = pco->dx1f(i);

//           // Calculate metric coefficients
//           Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//           // Calculate lengths
//           Real det = Determinant(g);
//           pco->coord_len1_kji_(k,j,i) = std::sqrt(-det) * dx1;
//         }
//       }
//     }
//   }

//   // Calculate x2-edge-centered geometric quantities
//   if (not coarse_flag) {
//     for (int k = kll; k <= kuu+1; ++k) {
//       for (int j = jll; j <= juu; ++j) {
//         for (int i = ill; i <= iuu+1; ++i) {

//           // Get position and separation
//           Real x1 = pco->x1f(i);
//           Real x2 = pco->x2v(j);
//           Real x3 = pco->x3f(k);
//           Real dx2 = pco->dx2f(j);

//           // Calculate metric coefficients
//           Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//           // Calculate lengths
//           Real det = Determinant(g);
//           pco->coord_len2_kji_(k,j,i) = std::sqrt(-det) * dx2;
//         }
//       }
//     }
//   }

//   // Calculate x3-edge-centered geometric quantities
//   if (not coarse_flag) {
//     for (int k = kll; k <= kuu; ++k) {
//       for (int j = jll; j <= juu+1; ++j) {
//         for (int i = ill; i <= iuu+1; ++i) {

//           // Get position and separation
//           Real x1 = pco->x1f(i);
//           Real x2 = pco->x2f(j);
//           Real x3 = pco->x3v(k);
//           Real dx3 = pco->dx3f(k);

//           // Calculate metric coefficients
//           Metric(metric_t,x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3,dg_dt);

//           // Calculate lengths
//           Real det = Determinant(g);
//           pco->coord_len3_kji_(k,j,i) = std::sqrt(-det) * dx3;
//         }
//       }
//     }
//   }




//   // Free scratch arrays
//   g.DeleteAthenaArray();
//   g_inv.DeleteAthenaArray();
//   dg_dx1.DeleteAthenaArray();
//   dg_dx2.DeleteAthenaArray();
//   dg_dx3.DeleteAthenaArray();
//   dg_dt.DeleteAthenaArray();
//   if (not coarse_flag) {
//     transformation.DeleteAthenaArray();
// }

// #define DEL 1e-7
// void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
//     AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
//     AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
// {

//   // if  (Globals::my_rank == 0) fprintf(stderr,"Metric time in pgen file (GLOBAL RANK): %g \n", t);
//   // else fprintf(stderr,"Metric time in pgen file (RANK %d): %g \n", Globals::my_rank,t);
//   // Extract inputs
//   Real x = x1;
//   Real y = x2;
//   Real z = x3;

//   Real a_spin =a;

//   Real eta[4];

//   eta[0] = -1.0;
//   eta[1] = 1.0;
//   eta[2] = 1.0;
//   eta[3] = 1.0;



//   //////////////Perturber Black Hole//////////////////


//   Real xprime,yprime,zprime,rprime,Rprime;
//   get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);


//   Real dx_bh2_dt = 0.0;
//   Real dy_bh2_dt = 0.0;
//   Real dz_bh2_dt = v_bh2;




// /// prevent metric from getting nan sqrt(-gdet)
//   Real thprime  = std::acos(zprime/rprime);
//   Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
//                               (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

//   Real rhprime = ( q + std::sqrt(SQR(q)-SQR(aprime)) );
//   if (rprime < rhprime*0.8) {
//     rprime = rhprime*0.8;
//     xprime = rprime * std::cos(phiprime)*std::sin(thprime) - aprime * std::sin(phiprime)*std::sin(thprime);
//     yprime = rprime * std::sin(phiprime)*std::sin(thprime) + aprime * std::cos(phiprime)*std::sin(thprime);
//     zprime = rprime * std::cos(thprime);
//   }



//   //if (r<0.01) r = 0.01;


//   Real l_lowerprime[4],l_upperprime[4];

//   Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));
//   l_upperprime[0] = -1.0;
//   l_upperprime[1] = (rprime*xprime + aprime*yprime)/( SQR(rprime) + SQR(aprime) );
//   l_upperprime[2] = (rprime*yprime - aprime*xprime)/( SQR(rprime) + SQR(aprime) );
//   l_upperprime[3] = zprime/rprime;

//   l_lowerprime[0] = 1.0;
//   l_lowerprime[1] = l_upperprime[1];
//   l_lowerprime[2] = l_upperprime[2];
//   l_lowerprime[3] = l_upperprime[3];


//   //BOOST //

//   // Real Lorentz = std::sqrt(1.0/(1.0 - SQR(v_bh2)));
//   Real Lorentz = 1.0;

//   Real l0 = l_lowerprime[0];
//   Real l3 = l_lowerprime[3];

//   l_lowerprime[0] = Lorentz * (l0 - v_bh2 * l3);
//   l_lowerprime[3] = Lorentz * (l3 - v_bh2 * l0);





//   // Set covariant components
//   // g(I00) = eta[0] + fprime * l_lowerprime[0]*l_lowerprime[0] + v_bh2 * fprime * l_lowerprime[0]*l_lowerprime[3];
//   // g(I01) =          fprime * l_lowerprime[0]*l_lowerprime[1] + v_bh2 * fprime * l_lowerprime[1]*l_lowerprime[3];
//   // g(I02) =          fprime * l_lowerprime[0]*l_lowerprime[2] + v_bh2 * fprime * l_lowerprime[2]*l_lowerprime[3];
//   // g(I03) =          fprime * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * l_lowerprime[3]*l_lowerprime[3] + v_bh2;
//   g(I00) = eta[0] + fprime * l_lowerprime[0]*l_lowerprime[0]; // - 2.0*v_bh2 * fprime * l_lowerprime[0]*l_lowerprime[3]  
//                   //+ SQR(v_bh2)*fprime*l_lowerprime[3]*l_lowerprime[3] + SQR(v_bh2) ;
//   g(I01) =          fprime * l_lowerprime[0]*l_lowerprime[1]; // - v_bh2 * fprime * l_lowerprime[1]*l_lowerprime[3];
//   g(I02) =          fprime * l_lowerprime[0]*l_lowerprime[2]; // - v_bh2 * fprime * l_lowerprime[2]*l_lowerprime[3];
//   g(I03) =          fprime * l_lowerprime[0]*l_lowerprime[3]; // - v_bh2 * fprime * l_lowerprime[3]*l_lowerprime[3] - v_bh2;
//   g(I11) = eta[1] + fprime * l_lowerprime[1]*l_lowerprime[1];
//   g(I12) =          fprime * l_lowerprime[1]*l_lowerprime[2];
//   g(I13) =          fprime * l_lowerprime[1]*l_lowerprime[3];
//   g(I22) = eta[2] + fprime * l_lowerprime[2]*l_lowerprime[2];
//   g(I23) =          fprime * l_lowerprime[2]*l_lowerprime[3];
//   g(I33) = eta[3] + fprime * l_lowerprime[3]*l_lowerprime[3];

//   // Real det_test = Determinant(g);

//   // if (std::isnan( std::sqrt(-det_test))) {
//   //   fprintf(stderr,"NAN determinant in metric!! Det: %g \n xyz: %g %g %g \n r: %g \n",det_test,x,y,z,r);
//   //   exit(0);
//   // }


//   bool invertible = gluInvertMatrix(g,g_inv);

//   if (invertible==false) {
//     fprintf(stderr,"Non-invertible matrix at xyz: %g %g %g\n", x,y,z);
//   }


//   //expressioons for a = 0

//   // f = 2.0/R;
//   // l_lower[1] = x/R;
//   // l_lower[2] = y/R;
//   // l_lower[3] = z/R;
//   // df_dx1 = -2.0 * x/SQR(R)/R;
//   // df_dx2 = -2.0 * y/SQR(R)/R;
//   // df_dx3 = -2.0 * z/SQR(R)/R;

//   // dl1_dx1 = -SQR(x)/SQR(R)/R + 1.0/R;
//   // dl1_dx2 = -x*y/SQR(R)/R; 
//   // dl1_dx3 = -x*z/SQR(R)/R;

//   // dl2_dx1 = -x*y/SQR(R)/R;
//   // dl2_dx2 = -SQR(y)/SQR(R)/R + 1.0/R;
//   // dl2_dx3 = -y*z/SQR(R)/R;

//   // dl3_dx1 = -x*z/SQR(R)/R;
//   // dl3_dx2 = -y*z/SQR(R)/R;
//   // dl3_dx3 = -SQR(z)/SQR(R)/R;



//   // // Set x-derivatives of covariant components
//   dg_dx1(I00) = 0.0;
//   dg_dx1(I01) = 0.0;
//   dg_dx1(I02) = 0.0;
//   dg_dx1(I03) = 0.0;
//   dg_dx1(I11) = 0.0;
//   dg_dx1(I12) = 0.0;
//   dg_dx1(I13) = 0.0;
//   dg_dx1(I22) = 0.0;
//   dg_dx1(I23) = 0.0;
//   dg_dx1(I33) = 0.0;

//   // Set y-derivatives of covariant components
//   dg_dx2(I00) = 0.0;
//   dg_dx2(I01) = 0.0;
//   dg_dx2(I02) = 0.0;
//   dg_dx2(I03) = 0.0;
//   dg_dx2(I11) = 0.0;
//   dg_dx2(I12) = 0.0;
//   dg_dx2(I13) = 0.0;
//   dg_dx2(I22) = 0.0;
//   dg_dx2(I23) = 0.0;
//   dg_dx2(I33) = 0.0;

//   // Set z-derivatives of covariant components
//   dg_dx3(I00) = 0.0;
//   dg_dx3(I01) = 0.0;
//   dg_dx3(I02) = 0.0;
//   dg_dx3(I03) = 0.0;
//   dg_dx3(I11) = 0.0;
//   dg_dx3(I12) = 0.0;
//   dg_dx3(I13) = 0.0;
//   dg_dx3(I22) = 0.0;
//   dg_dx3(I23) = 0.0;
//   dg_dx3(I33) = 0.0;



// /////Secondary Black hole/////

//   Real sqrt_term =  2.0*SQR(rprime)-SQR(Rprime) + SQR(aprime);
//   Real rsq_p_asq = SQR(rprime) + SQR(aprime);

//   Real fprime_over_q = 2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));


//   Real dfprime_dx1 = q * SQR(fprime_over_q)*xprime/(2.0*std::pow(rprime,3)) * 
//                       ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
//   //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
//   Real dfprime_dx2 = q * SQR(fprime_over_q)*yprime/(2.0*std::pow(rprime,3)) * 
//                       ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
//   Real dfprime_dx3 = q * SQR(fprime_over_q)*zprime/(2.0*std::pow(rprime,5)) * 
//                       ( ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(aprime*rprime)) ;
//   //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
//   Real dl1prime_dx1 = xprime*rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + rprime/( rsq_p_asq );
//   // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
//   Real dl1prime_dx2 = yprime*rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ aprime/( rsq_p_asq );
//   Real dl1prime_dx3 = zprime/rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( (rsq_p_asq) * ( sqrt_term ) ) ;
//   Real dl2prime_dx1 = xprime*rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - aprime/( rsq_p_asq );
//   Real dl2prime_dx2 = yprime*rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + rprime/( rsq_p_asq );
//   Real dl2prime_dx3 = zprime/rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( (rsq_p_asq) * ( sqrt_term ) );
//   Real dl3prime_dx1 = - xprime*zprime/(rprime) /( sqrt_term );
//   Real dl3prime_dx2 = - yprime*zprime/(rprime) /( sqrt_term );
//   Real dl3prime_dx3 = - SQR(zprime)/(SQR(rprime)*rprime) * ( rsq_p_asq )/( sqrt_term ) + 1.0/rprime;

//   Real dl0prime_dx1 = 0.0;
//   Real dl0prime_dx2 = 0.0;
//   Real dl0prime_dx3 = 0.0;

//   AthenaArray<Real> dgprime_dx1, dgprime_dx2, dgprime_dx3;

//   dgprime_dx1.NewAthenaArray(NMETRIC);
//   dgprime_dx2.NewAthenaArray(NMETRIC);
//   dgprime_dx3.NewAthenaArray(NMETRIC);



  
//   Real dl0_dx1_tmp = dl0prime_dx1;
//   Real dl0_dx2_tmp = dl0prime_dx2;
//   Real dl0_dx3_tmp = dl0prime_dx3;

//   Real dl3_dx1_tmp = dl3prime_dx1;
//   Real dl3_dx2_tmp = dl3prime_dx2;
//   Real dl3_dx3_tmp = dl3prime_dx3;



//   dl0prime_dx1 = Lorentz * (dl0_dx1_tmp - v_bh2 * dl3_dx1_tmp); 
//   dl0prime_dx2 = Lorentz * (dl0_dx2_tmp - v_bh2 * dl3_dx2_tmp); 
//   dl0prime_dx3 = Lorentz * (dl0_dx3_tmp - v_bh2 * dl3_dx3_tmp); 


//   dl3prime_dx1 = Lorentz * (dl3_dx1_tmp - v_bh2 * dl0_dx1_tmp); 
//   dl3prime_dx2 = Lorentz * (dl3_dx2_tmp - v_bh2 * dl0_dx2_tmp); 
//   dl3prime_dx3 = Lorentz * (dl3_dx3_tmp - v_bh2 * dl0_dx3_tmp); 


//   // // Set x-derivatives of covariant components
//   // dgprime_dx1(I00) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx1 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx1
//   //                    + v_bh2 * dfprime_dx1 * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * dl0prime_dx1*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[0]*dl3prime_dx1;
//   // dgprime_dx1(I01) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx1;
//   //                    + v_bh2 * dfprime_dx1 * l_lowerprime[1]*l_lowerprime[3] + v_bh2 * fprime * dl1prime_dx1*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[1]*dl3prime_dx1;
//   // dgprime_dx1(I02) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx1
//   //                    + v_bh2 * dfprime_dx1 * l_lowerprime[2]*l_lowerprime[3] + v_bh2 * fprime * dl2prime_dx1*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[2]*dl3prime_dx1;
//   // dgprime_dx1(I03) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx1
//   //                    + v_bh2 * dfprime_dx1 * l_lowerprime[3]*l_lowerprime[3] + v_bh2 * fprime * dl3prime_dx1*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[3]*dl3prime_dx1;  

//   dgprime_dx1(I00) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx1 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx1;
//                     // - 2.0 * v_bh2 * (dfprime_dx1 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1*l_lowerprime[3]
//                     // +                fprime * l_lowerprime[0]*dl3prime_dx1) 
//                     // +  SQR(v_bh2) * (dfprime_dx1 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx1*l_lowerprime[3] 
//                     // +                fprime * l_lowerprime[3]*dl3prime_dx1);
//   dgprime_dx1(I01) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx1;
//                     // - v_bh2 * (dfprime_dx1 * l_lowerprime[1]*l_lowerprime[3] 
//                     // +          fprime * dl1prime_dx1*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[1]*dl3prime_dx1);
//   dgprime_dx1(I02) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx1;
//                     // - v_bh2 * (dfprime_dx1 * l_lowerprime[2]*l_lowerprime[3] 
//                     // +          fprime * dl2prime_dx1*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[2]*dl3prime_dx1);
//   dgprime_dx1(I03) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx1;
//                     // - v_bh2 * (dfprime_dx1 * l_lowerprime[3]*l_lowerprime[3] 
//                     // +          fprime * dl3prime_dx1*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[3]*dl3prime_dx1);  
//   dgprime_dx1(I11) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx1;
//   dgprime_dx1(I12) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx1;
//   dgprime_dx1(I13) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx1;
//   dgprime_dx1(I22) = dfprime_dx1*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx1;
//   dgprime_dx1(I23) = dfprime_dx1*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx1;
//   dgprime_dx1(I33) = dfprime_dx1*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx1;

//   // Set y-derivatives of covariant components
//   // dgprime_dx2(I00) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx2 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx2
//   //                    + v_bh2 * dfprime_dx2 * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * dl0prime_dx2*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[0]*dl3prime_dx2;
//   // dgprime_dx2(I01) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx2
//   //                    + v_bh2 * dfprime_dx2 * l_lowerprime[1]*l_lowerprime[3] + v_bh2 * fprime * dl1prime_dx2*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[1]*dl3prime_dx2;
//   // dgprime_dx2(I02) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx2
//   //                    + v_bh2 * dfprime_dx2 * l_lowerprime[2]*l_lowerprime[3] + v_bh2 * fprime * dl2prime_dx2*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[2]*dl3prime_dx2;
//   // dgprime_dx2(I03) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx2
//   //                    + v_bh2 * dfprime_dx2 * l_lowerprime[3]*l_lowerprime[3] + v_bh2 * fprime * dl3prime_dx2*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[3]*dl3prime_dx2;  
//   dgprime_dx2(I00) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx2 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx2;
//                     // - 2.0 * v_bh2 * (dfprime_dx2 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2*l_lowerprime[3]
//                     // +                fprime * l_lowerprime[0]*dl3prime_dx2) 
//                     // +  SQR(v_bh2) * (dfprime_dx2 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx2*l_lowerprime[3] 
//                     // +                fprime * l_lowerprime[3]*dl3prime_dx2);
//   dgprime_dx2(I01) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx2;
//                     // - v_bh2 * (dfprime_dx2 * l_lowerprime[1]*l_lowerprime[3] 
//                     // +          fprime * dl1prime_dx2*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[1]*dl3prime_dx2);
//   dgprime_dx2(I02) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx2;
//                     // - v_bh2 * (dfprime_dx2 * l_lowerprime[2]*l_lowerprime[3] 
//                     // +          fprime * dl2prime_dx2*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[2]*dl3prime_dx2);
//   dgprime_dx2(I03) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx2;
//                     // - v_bh2 * (dfprime_dx2 * l_lowerprime[3]*l_lowerprime[3] 
//                     // +          fprime * dl3prime_dx2*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[3]*dl3prime_dx2);  
//   dgprime_dx2(I11) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx2;
//   dgprime_dx2(I12) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx2;
//   dgprime_dx2(I13) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx2;
//   dgprime_dx2(I22) = dfprime_dx2*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx2;
//   dgprime_dx2(I23) = dfprime_dx2*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx2;
//   dgprime_dx2(I33) = dfprime_dx2*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx2;

//   // Set z-derivatives of covariant components
//   // dgprime_dx3(I00) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx3 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx3
//   //                    + v_bh2 * dfprime_dx3 * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * dl0prime_dx3*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[0]*dl3prime_dx3;
//   // dgprime_dx3(I01) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx3
//   //                     + v_bh2 * dfprime_dx3 * l_lowerprime[1]*l_lowerprime[3] + v_bh2 * fprime * dl1prime_dx3*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[1]*dl3prime_dx3;
//   // dgprime_dx3(I02) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx3
//   //                      + v_bh2 * dfprime_dx3 * l_lowerprime[2]*l_lowerprime[3] + v_bh2 * fprime * dl2prime_dx3*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[2]*dl3prime_dx3;;
//   // dgprime_dx3(I03) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx3
//   //                    + v_bh2 * dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] + v_bh2 * fprime * dl3prime_dx3*l_lowerprime[3]
//   //                    + v_bh2 * fprime * l_lowerprime[3]*dl3prime_dx3; ;

//   dgprime_dx3(I00) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx3 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx3;
//                     // - 2.0 * v_bh2 * (dfprime_dx3 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3*l_lowerprime[3]
//                     // +                fprime * l_lowerprime[0]*dl3prime_dx3) 
//                     // +  SQR(v_bh2) * (dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx3*l_lowerprime[3] 
//                     // +                fprime * l_lowerprime[3]*dl3prime_dx3);
//   dgprime_dx3(I01) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx3;
//                     // - v_bh2 * (dfprime_dx3 * l_lowerprime[1]*l_lowerprime[3] 
//                     // +          fprime * dl1prime_dx3*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[1]*dl3prime_dx3);
//   dgprime_dx3(I02) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx3;
//                     // - v_bh2 * (dfprime_dx3 * l_lowerprime[2]*l_lowerprime[3] 
//                     // +          fprime * dl2prime_dx3*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[2]*dl3prime_dx3);
//   dgprime_dx3(I03) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx3;
//                     // - v_bh2 * (dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] 
//                     // +          fprime * dl3prime_dx3*l_lowerprime[3]
//                     // +          fprime * l_lowerprime[3]*dl3prime_dx3);  
//   dgprime_dx3(I11) = dfprime_dx3*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx3;
//   dgprime_dx3(I12) = dfprime_dx3*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx3;
//   dgprime_dx3(I13) = dfprime_dx3*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx3;
//   dgprime_dx3(I22) = dfprime_dx3*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx3;
//   dgprime_dx3(I23) = dfprime_dx3*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx3;
//   dgprime_dx3(I33) = dfprime_dx3*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx3;





//   // // Set x-derivatives of covariant components
//   dg_dx1(I00) += dgprime_dx1(I00);
//   dg_dx1(I01) += dgprime_dx1(I01);
//   dg_dx1(I02) += dgprime_dx1(I02);
//   dg_dx1(I03) += dgprime_dx1(I03);
//   dg_dx1(I11) += dgprime_dx1(I11);
//   dg_dx1(I12) += dgprime_dx1(I12);
//   dg_dx1(I13) += dgprime_dx1(I13);
//   dg_dx1(I22) += dgprime_dx1(I22);
//   dg_dx1(I23) += dgprime_dx1(I23);
//   dg_dx1(I33) += dgprime_dx1(I33);

//   // Set y-derivatives of covariant components
//   dg_dx2(I00) += dgprime_dx2(I00);
//   dg_dx2(I01) += dgprime_dx2(I01);
//   dg_dx2(I02) += dgprime_dx2(I02);
//   dg_dx2(I03) += dgprime_dx2(I03);
//   dg_dx2(I11) += dgprime_dx2(I11);
//   dg_dx2(I12) += dgprime_dx2(I12);
//   dg_dx2(I13) += dgprime_dx2(I13);
//   dg_dx2(I22) += dgprime_dx2(I22);
//   dg_dx2(I23) += dgprime_dx2(I23);
//   dg_dx2(I33) += dgprime_dx2(I33);

//   // Set z-derivatives of covariant components
//   dg_dx3(I00) += dgprime_dx3(I00);
//   dg_dx3(I01) += dgprime_dx3(I01);
//   dg_dx3(I02) += dgprime_dx3(I02);
//   dg_dx3(I03) += dgprime_dx3(I03);
//   dg_dx3(I11) += dgprime_dx3(I11);
//   dg_dx3(I12) += dgprime_dx3(I12);
//   dg_dx3(I13) += dgprime_dx3(I13);
//   dg_dx3(I22) += dgprime_dx3(I22);
//   dg_dx3(I23) += dgprime_dx3(I23);
//   dg_dx3(I33) += dgprime_dx3(I33);





//   // Set t-derivatives of covariant components
//   dg_dt(I00) = -1.0 * (dx_bh2_dt * dgprime_dx1(I00) + dy_bh2_dt * dgprime_dx2(I00) + dz_bh2_dt * dgprime_dx3(I00) );
//   dg_dt(I01) = -1.0 * (dx_bh2_dt * dgprime_dx1(I01) + dy_bh2_dt * dgprime_dx2(I01) + dz_bh2_dt * dgprime_dx3(I01) );
//   dg_dt(I02) = -1.0 * (dx_bh2_dt * dgprime_dx1(I02) + dy_bh2_dt * dgprime_dx2(I02) + dz_bh2_dt * dgprime_dx3(I02) );
//   dg_dt(I03) = -1.0 * (dx_bh2_dt * dgprime_dx1(I03) + dy_bh2_dt * dgprime_dx2(I03) + dz_bh2_dt * dgprime_dx3(I03) );
//   dg_dt(I11) = -1.0 * (dx_bh2_dt * dgprime_dx1(I11) + dy_bh2_dt * dgprime_dx2(I11) + dz_bh2_dt * dgprime_dx3(I11) );
//   dg_dt(I12) = -1.0 * (dx_bh2_dt * dgprime_dx1(I12) + dy_bh2_dt * dgprime_dx2(I12) + dz_bh2_dt * dgprime_dx3(I12) );
//   dg_dt(I13) = -1.0 * (dx_bh2_dt * dgprime_dx1(I13) + dy_bh2_dt * dgprime_dx2(I13) + dz_bh2_dt * dgprime_dx3(I13) );
//   dg_dt(I22) = -1.0 * (dx_bh2_dt * dgprime_dx1(I22) + dy_bh2_dt * dgprime_dx2(I22) + dz_bh2_dt * dgprime_dx3(I22) );
//   dg_dt(I23) = -1.0 * (dx_bh2_dt * dgprime_dx1(I23) + dy_bh2_dt * dgprime_dx2(I23) + dz_bh2_dt * dgprime_dx3(I23) );
//   dg_dt(I33) = -1.0 * (dx_bh2_dt * dgprime_dx1(I33) + dy_bh2_dt * dgprime_dx2(I33) + dz_bh2_dt * dgprime_dx3(I33) );


//   dgprime_dx1.DeleteAthenaArray();
//   dgprime_dx2.DeleteAthenaArray();
//   dgprime_dx3.DeleteAthenaArray();


//   // AthenaArray<Real> gp,gm;
//   // AthenaArray<Real> delta_gp,delta_gm;



//   // gp.NewAthenaArray(NMETRIC);
//   // gm.NewAthenaArray(NMETRIC);
//   // delta_gp.NewAthenaArray(NMETRIC);
//   // delta_gm.NewAthenaArray(NMETRIC);



//   // cks_metric(x1,x2,x3,g);
//   // delta_cks_metric(pin,t,x1,x2,x3,delta_gp);

//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    g(n) += delta_gp(n);
//   // }


//   // cks_inverse_metric(x1,x2,x3,g_inv);
//   // delta_cks_metric_inverse(pin,t,x1,x2,x3,delta_gp);

//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    g_inv(n) += delta_gp(n);
//   // }

//   // Real x1p = x1 + DEL * rprime;
//   // Real x1m = x1 - DEL * rprime;

//   // cks_metric(x1p,x2,x3,gp);
//   // cks_metric(x1m,x2,x3,gm);
//   // delta_cks_metric(pin,t,x1p,x2,x3,delta_gp);
//   // delta_cks_metric(pin,t,x1m,x2,x3,delta_gm);

//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    gp(n) += delta_gp(n);
//   //    gm(n) += delta_gm(n);
//   // }

//   //   // // Set x-derivatives of covariant components
//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    dg_dx1(n) = (gp(n)-gm(n))/(x1p-x1m);
//   // }

//   // Real x2p = x2 + DEL * rprime;
//   // Real x2m = x2 - DEL * rprime;

//   // cks_metric(x1,x2p,x3,gp);
//   // cks_metric(x1,x2m,x3,gm);
//   // delta_cks_metric(pin,t,x1,x2p,x3,delta_gp);
//   // delta_cks_metric(pin,t,x1,x2m,x3,delta_gm);
//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    gp(n) += delta_gp(n);
//   //    gm(n) += delta_gm(n);
//   // }

//   //   // // Set y-derivatives of covariant components
//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    dg_dx2(n) = (gp(n)-gm(n))/(x2p-x2m);
//   // }
  
//   // Real x3p = x3 + DEL * rprime;
//   // Real x3m = x3 - DEL * rprime;

//   // cks_metric(x1,x2,x3p,gp);
//   // cks_metric(x1,x2,x3m,gm);
//   // delta_cks_metric(pin,t,x1,x2,x3p,delta_gp);
//   // delta_cks_metric(pin,t,x1,x2,x3m,delta_gm);
//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    gp(n) += delta_gp(n);
//   //    gm(n) += delta_gm(n);
//   // }

//   //   // // Set z-derivatives of covariant components
//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    dg_dx3(n) = (gp(n)-gm(n))/(x3p-x3m);
//   // }

//   // Real tp = t + DEL ;
//   // Real tm = t - DEL ;

//   // cks_metric(x1,x2,x3,gp);
//   // cks_metric(x1,x2,x3,gm);
//   // delta_cks_metric(pin,tp,x1,x2,x3,delta_gp);
//   // delta_cks_metric(pin,tm,x1,x2,x3,delta_gm);

//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    gp(n) += delta_gp(n);
//   //    gm(n) += delta_gm(n);
//   // }

//   //   // // Set t-derivatives of covariant components
//   // for (int n = 0; n < NMETRIC; ++n) {
//   //    dg_dt(n) = (gp(n)-gm(n))/(tp-tm);
//   // }

//   // gp.DeleteAthenaArray();
//   // gm.DeleteAthenaArray();
//   // delta_gm.DeleteAthenaArray();
//   // delta_gp.DeleteAthenaArray();
//   return;
// }