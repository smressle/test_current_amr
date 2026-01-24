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
void apply_inner_boundary_condition(MeshBlock *pmb,const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar, const FaceField &bb_old);
void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar);

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real ax, Real ay, Real az,Real *pr,
                                         Real *ptheta, Real *pphi);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real a, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void TransformAphi(Real a3_bl, Real x1,
                     Real x2, Real x3, Real a, Real *pa1, Real *pa2, Real *pa3);


int RefinementCondition(MeshBlock *pmb);
void  Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);
bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv);


void get_prime_coords(int BH_INDEX,Real x, Real y, Real z, AthenaArray<Real> &orbit_quantities,Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ);

void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
  AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt, bool take_derivatives);


void BoostVector(int BH_INDEX, Real t, Real a0, Real a1, Real a2, Real a3,AthenaArray<Real>&orbit_quantities, Real *pa0, Real *pa1, Real *pa2, Real *pa3);

Real DivergenceB(MeshBlock *pmb, int iout);

void set_orbit_arrays(std::string orbit_file_name);

void convert_spherical_to_cartesian_ks(Real r, Real th, Real phi, Real ax, Real ay, Real az,
    Real *x, Real *y, Real *z);

void get_orbit_quantities(Real t, AthenaArray<Real>&orbit_quantities);
void interp_orbits(Real t, int iorbit, AthenaArray<Real> &arr, Real *result);

void get_free_fall_solution(Real r, Real x1, Real x2, Real x3, Real ax_, Real ay_, Real az_, Real *uut, Real *uux1,
                                         Real *uux2, Real *uux3);
void unboosted_cks_metric(Real q_rat,Real xprime, Real yprime, Real zprime, Real rprime, Real Rprime, Real vx, Real vy, Real vz,Real ax, Real ay, Real az,AthenaArray<Real> &g_unboosted );
void ks_metric(Real r, Real th,Real a,AthenaArray<Real> &g_ks );
void boosted_BH_metric_addition(Real q_rat,Real xprime, Real yprime, Real zprime, Real rprime, Real Rprime, Real vx, Real vy, Real vz,Real ax, Real ay, Real az,AthenaArray<Real> &g_pert );


void NobleCooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);


Real MyTimeStep(MeshBlock *pmb);


// Global variables
static Real m;                                  // black hole parameters
static int sample_n_r, sample_n_theta;             // number of cells in 2D sample grid
static int sample_n_phi;                           // number of cells in 3D sample grid
static Real dfloor,pfloor;                         // density and pressure floors
static Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters


static Real q;          // black hole mass and spin
//Real q; 
// static Real r_inner_boundary,r_inner_boundary_2;
// static Real rh2;
// static Real eccentricity, tau, mean_angular_motion;
static Real t0; //time at which second BH is at polar axis
static Real orbit_inclination;
static Real field_norm;
static Real H_over_r_target;


static Real t0_orbits,dt_orbits;

static int nt;
AthenaArray<Real> t_orbits,orbit_array;


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

static Real SMALL = 1e-7;
#define DEL 1e-4;

Real gamma_max;


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


//----------------------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem-specific parameters from input file
    // Read problem-specific parameters from input file
  rho_min = pin->GetReal("hydro", "rho_min");
  rho_pow = pin->GetReal("hydro", "rho_pow");
  pgas_min = pin->GetReal("hydro", "pgas_min");
  pgas_pow = pin->GetReal("hydro", "pgas_pow");
  k_adi = pin->GetReal("problem", "k_adi");
  rin = pin->GetReal("problem", "rin");
  r_peak = pin->GetReal("problem", "r_peak");
  n_pow = pin->GetReal("problem", "n_pow");
  rho_max = pin->GetReal("problem", "rho_max");

  if (MAGNETIC_FIELDS_ENABLED) {
    std::string field_config_str = pin->GetString("problem",
                                                  "field_config");
    if (field_config_str == "normal") {
      field_config = normal;
    } 
    else if (field_config_str == "multi_loop"){
      field_config = multi_loop;
    }
      else if (field_config_str == "renorm") {
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

    potential_cutoff = pin->GetReal("problem", "potential_cutoff");
    potential_r_pow = pin->GetReal("problem", "potential_r_pow");
    potential_rho_pow = pin->GetReal("problem", "potential_rho_pow");
    potential_sinth_pow = pin->GetOrAddReal("problem", "potential_sinth_pow",0.0);
    potential_costh_pow = pin->GetOrAddReal("problem", "potential_costh_pow",0.0);

    potential_theta_min = pin->GetOrAddReal("problem", "potential_theta_min",0.0);
    potential_theta_max = pin->GetOrAddReal("problem", "potential_theta_max",PI);


    extra_field_norm = pin->GetOrAddReal("problem", "extra_field_norm",1.0);

    loop_radius = pin->GetOrAddReal("problem","loop_radius",10.0);
    potential_r_exp_cut  = pin->GetOrAddReal("problem","potential_r_exp_cut",1e6);
    potential_theta_scale_height = pin->GetOrAddReal("problem","potential_theta_scale_height",1e6);
    N_loops_theta = pin->GetOrAddReal("problem","N_loops_theta",1.0);

    beta_min = pin->GetReal("problem", "beta_min");

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




  if (MAGNETIC_FIELDS_ENABLED) field_norm =  pin->GetReal("problem", "field_norm");



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

    //Enroll metric
  EnrollUserMetric(Cartesian_GR);

  if (METRIC_EVOLUTION)  EnrollUserMetricWithoutPin(Binary_BH_Metric);

  EnrollUserTimeStepFunction(MyTimeStep);


  EnrollUserRadSourceFunction(inner_boundary_source_function);

  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, DivergenceB, "divB");
  }


  t0 = pin->GetOrAddReal("problem","t0", 0.0);
  m =pin->GetReal("coord", "m");

  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);


  std::string orbit_file_name;
  orbit_file_name =  pin->GetOrAddString("problem","orbit_filename", "orbits.in");
  set_orbit_arrays(orbit_file_name);


  // if (MAGNETIC_FIELDS_ENABLED) EnrollUserExplicitEMFSourceFunction(emf_source);

  // fprintf(stderr,"Done with set_orbit_arrays \n");


  gamma_max = pin->GetOrAddReal("hydro", "gamma_max", 1000.0);

      //SEE DE VILLIERS+ 2003 https://arxiv.org/pdf/astro-ph/0307260.pdf

  Real a = 0;
  Real rmb = 2.0 - a + 2.0 * std::sqrt(1.0-a);   //Equation 20 of https://arxiv.org/abs/1707.05680
  Real Z1 = 1.0 + std::pow( (1.0-SQR(a)), 0.33333) * ( std::pow( (1.0+a),0.3333) + std::pow( (1.0-a), 0.3333) );
  Real Z2 = std::sqrt(3.0*SQR(a) + SQR(Z1));
  Real rms = (3.0 + Z2 - std::sqrt( (3.0-Z1) * (3.0 + Z1 + 2.0*Z2) ) ); // Eq 1.136 in https://s3.cern.ch/inspire-prod-files-e/ebb8246d045759f2a7947d05492e894c ()Luciano Rezzolla An Introduction to Astrophysical Black Holes and Their Dynamical Production


  Real lmb = l_kep(a,rmb);
  Real lms = l_kep(a,rms);

  Real lc = l_kep(a,rc);


    // return 1.0/np.sqrt( - (gtphi(r,a,theta) + gtt(r,a,theta)*l) / (l*gphiphi(r,a,theta) + l**2.0*gtphi(r,a,theta) )  )

  Real lambda_in = std::sqrt(-gphiphi(rin,a,PI/2.0)/gtt(rin,a,PI/2.0) ); //lambda_func(rin,a,PI/2.0,lin)
  Real lambda_c = std::sqrt(-gphiphi(rc,a,PI/2.0)/gtt(rc,a,PI/2.0) ); //3lambda_func(rc,a,PI/2.0,lc)


  lin = lc/std::exp(n_pow*std::log(lambda_c/lambda_in) );
  c_const = lc/std::pow(lambda_c,n_pow);

  Real q_pow = 2.0-n_pow;
  Real alpha_pow = (2.0*n_pow-2.0)/n_pow; //q_pow/(q_pow-2.0);

  ud_t_in = -1.0/std::sqrt( - (gitt(rin,a,PI/2.0) - 2.0*lin*gitphi(rin,a,PI/2.0) + SQR(lin)*giphiphi(rin,a,PI/2.0) ) );



  // Compute Peak Density //
  Real denom_sq = -( gitt(rc,a,PI/2.0) - 2.0*lc*gitphi(rc,a,PI/2.0) + SQR(lc)*giphiphi(rc,a,PI/2.0) );
  Real ud_t_c = -1.0/std::sqrt(denom_sq);
  Real eps_c = 1.0/gam * (ud_t_in * f(lin,c_const,n_pow)/(ud_t_c * f(lc,c_const,n_pow)) -1.0);
  rho_peak = std::pow( (eps_c * (gam-1.0)/k_adi), (1.0/(gam-1.0)) );
  pgas_over_rho_peak = eps_c * (gam-1.0);
  Real gamma_adi = pin->GetReal("hydro", "gamma");
  kappa_init = k_adi * std::pow(rho_peak,gamma_adi-1.0);

  EnrollUserExplicitSourceFunction(NobleCooling);




  return;
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
  // q = pin->GetOrAddReal("problem", "q", 0.1);
  // aprime = q * pin->GetOrAddReal("problem", "a_bh2", 0.0);
  // r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);

  t0 = pin->GetOrAddReal("problem","t0", 0.0);

  // orbit_inclination = pin->GetOrAddReal("problem","orbit_inclination",0.0);


  // rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  //r_inner_boundary = rh/2.0;


    // Get mass of black hole
  // Real m2 = q;

  // rh2 =  ( m2 + std::sqrt( SQR(m2) - SQR(aprime)) );
  //r_inner_boundary_2 = rh2/2.0;

  // int N_user_vars = 7;
  // if (MAGNETIC_FIELDS_ENABLED) {
  //   AllocateUserOutputVariables(N_user_vars);
  // } else {
  //   AllocateUserOutputVariables(N_user_vars);
  // }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1+NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1+NGHOST);






  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));




  return;
}


int RefinementCondition(MeshBlock *pmb)
{

  // return 0;
  int refine = 0;

    Real DX,DY,DZ;
    Real dx,dy,dz;
  get_uniform_box_spacing(pmb->pmy_mesh->mesh_size,&DX,&DY,&DZ);
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);


  Real total_box_radius = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2.0;
  Real bh2_focus_radius = 3.125*q;
  // Real bh1_focus_radius = 3.125;

  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  // if (current_level >=max_refinement_level) return 0;

  int any_in_refinement_region = 0;
  int any_at_current_level=0;


  int max_level_required = 0;


  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->time,orbit_quantities);

  Real binary_separation_distance = std::sqrt( 
                                              SQR(orbit_quantities(IX1)-orbit_quantities(IX2)) + 
                                              SQR(orbit_quantities(IY1)-orbit_quantities(IY2)) +
                                              SQR(orbit_quantities(IZ1)-orbit_quantities(IZ2)) );

  Real binary_radius = binary_separation_distance/2.0;

  //this is the n such that total_box_radius/2**n is greater than binary_radius while for n+1 it is smaller
  int n_max_for_smr=  static_cast<int>(std::ceil(std::log2(total_box_radius/binary_radius) )) - 1;

  // fprintf(stderr,"current level: %d max_refinement_level: %d max_smr_refinement: %d max_bh2_refinement: %d \n",current_level,max_refinement_level,max_smr_refinement_level,max_second_bh_refinement_level);
  //first loop: check if any part of block is within refinement levels for secondary black hole

  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

          if (n_max_for_smr+2>max_second_bh_refinement_level) break;
          for (int n_level = n_max_for_smr+2; n_level<=max_second_bh_refinement_level; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(2,x,y,z, orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            //Real box_radius = bh2_focus_radius * std::pow(2.,max_second_bh_refinement_level - n_level)*0.9999;
            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

            
            Real mesh_block_widthx = pmb->block_size.nx1 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx1*1.0);
            Real mesh_block_widthy = pmb->block_size.nx2 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx2*1.0);
            Real mesh_block_widthz = pmb->block_size.nx3 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx3*1.0);

        
            //           if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
            // fprintf(stderr,"current level (AMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
            // }
            if (xprime<(box_radius-mesh_block_widthx/2.0) && xprime > -(box_radius-mesh_block_widthx/2.0) && 
              yprime<(box_radius-mesh_block_widthy/2.0) && yprime > -(box_radius-mesh_block_widthy/2.0) && 
              zprime<(box_radius-mesh_block_widthz/2.0) && zprime > -(box_radius-mesh_block_widthz/2.0) ){
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
          
          if (n_max_for_smr+2>max_smr_refinement_level) break;
          for (int n_level = n_max_for_smr+2; n_level<=max_smr_refinement_level; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(1,x,y,z,orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

            Real mesh_block_widthx = pmb->block_size.nx1 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx1*1.0);
            Real mesh_block_widthy = pmb->block_size.nx2 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx2*1.0);
            Real mesh_block_widthz = pmb->block_size.nx3 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx3*1.0);

            // Real fake_xprime = x - orbit_quantities(IX1);
            // Real fake_yprime = y - orbit_quantities(IY1);
            // Real fake_zprime = z - orbit_quantities(IZ1);
          

             // if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
             //   fprintf(stderr,"current level (SMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
             //    }
            // if (fake_xprime<box_radius && fake_xprime > -box_radius && fake_yprime<box_radius
            //   && fake_yprime > -box_radius && fake_zprime<box_radius && fake_zprime > -box_radius ){
            if (xprime<(box_radius-mesh_block_widthx/2.0) && xprime > -(box_radius-mesh_block_widthx/2.0) && 
              yprime<(box_radius-mesh_block_widthy/2.0) && yprime > -(box_radius-mesh_block_widthy/2.0) && 
              zprime<(box_radius-mesh_block_widthz/2.0) && zprime > -(box_radius-mesh_block_widthz/2.0) ){

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

  //third loop: default SMR levels that stop when it reaches the binary orbital radius

  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
          
          for (int n_level = 1; n_level<=n_max_for_smr; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

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

static Real k_adi;                      // hydro parameters
static Real rin, r_peak, l, rho_max;            // fixed torus parameters
static Real psi, sin_psi, cos_psi;                 // tilt parameters
static Real log_h_edge, log_h_peak;                // calculated torus parameters
static Real pgas_over_rho_peak, rho_peak;          // more calculated torus parameters
static Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
static b_configs field_config;                     // type of magnetic field
static Real potential_cutoff;                      // sets region of torus to magnetize
static Real potential_r_pow, potential_rho_pow;    // set how vector potential scales
static Real potential_sinth_pow,potential_costh_pow;
static Real potential_theta_min, potential_theta_max;
static Real loop_radius;
static Real potential_r_exp_cut, potential_theta_scale_height;
static Real N_loops_theta; 
static Real extra_field_norm;   
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
// static Real rh;                            
        // horizon radius
static Real n_pow;

// Constants Needed for Torus
static Real lin;
static Real c_const;
static Real n_pow;
static Real ud_t_in;
static Real rho_peak;
static Real pgas_over_rho_peak;
static Real kappa_init;


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

  Real gtphi(Real r, Real a, Real theta){
    Real cos2 =  SQR( std::cos(theta) );
    Real sin2 = SQR( std::sin(theta) );
    Real a2 = SQR(a) ;
    Real r2 = SQR(r);
    Real delta = r2 - 2.0*r + a2;
    Real sigma = r2 + a2 * cos2;

    return -2.0*a*r/sigma * sin2;
  }
  Real gtt(Real r, Real a, Real theta){
    Real cos2 =  SQR( std::cos(theta) );
    Real sin2 = SQR( std::sin(theta) );
    Real a2 = SQR(a) ;
    Real r2 = SQR(r);
    Real delta = r2 - 2.0*r + a2;
    Real sigma = r2 + a2 * cos2;

    return -(1.0 - 2.0*r/sigma);
  }
  Real gphiphi(Real r, Real a, Real theta){
    Real cos2 =  SQR( std::cos(theta) );
    Real sin2 = SQR( std::sin(theta) );
    Real a2 = SQR(a) ;
    Real r2 = SQR(r);
    Real delta = r2 - 2.0*r + a2;
    Real sigma = r2 + a2 * cos2;

    return (r2 + a2 + 2.0*a2*r/sigma * sin2) * sin2;
  }


  Real gitphi(Real r, Real a, Real theta){
    Real cos2 =  SQR( std::cos(theta) );
    Real sin2 = SQR( std::sin(theta) );
    Real a2 = SQR(a) ;
    Real r2 = SQR(r);
    Real delta = r2 - 2.0*r + a2;
    Real sigma = r2 + a2 * cos2;

    return -2.0*r/(sigma*delta)*a;
   } 

  Real gitt(Real r, Real a, Real theta){
    Real cos2 =  SQR( std::cos(theta) );
    Real sin2 = SQR( std::sin(theta) );
    Real a2 = SQR(a) ;
    Real r2 = SQR(r);
    Real delta = r2 - 2.0*r + a2;
    Real sigma = r2 + a2 * cos2;

    return -1.0/delta * (r2 + a2 +2*r*a2/sigma*sin2);
  }

  Real giphiphi(Real r, Real a, Real theta){
    Real cos2 =  SQR( std::cos(theta) );
    Real sin2 = SQR( std::sin(theta) );
    Real a2 = SQR(a) ;
    Real r2 = SQR(r);
    Real delta = r2 - 2.0*r + a2;
    Real sigma = r2 + a2 * cos2;

    return (delta - a2*sin2)/(sigma*delta*sin2);
  }


  Real  lambda_func(Real r,Real a,Real theta,Real l){
    return std::sqrt(-gphiphi(r,a,theta)/gtt(r,a,theta) );
  }
  Real l_kep(Real a,Real r){

    Real Omega = 1.0/(std::pow(r,1.5) + a);

    //Omega = - (gtphi + l gtt)/(gphiphi + l gtphi)
    //l (Omega gtphi + gtt) = -gtphi - Omega gphiphi
    // l = - (gtphi + Omega gphiphi)/(Omega gtphi + gtt)
    // Equation 2.4a in Chakrabarti
    Real l = - (gtphi(r,a,PI/2.0) + Omega * gphiphi(r,a,PI/2.0) ) / ( Omega * gtphi(r,a,PI/2.0) + gtt(r,a,PI/2.0) );
    return l;
  }


  Real f(Real l, Real c_const,Real n_pow){
    Real alpha_pow = (2.0*n_pow-2.0)/n_pow; //q_pow/(q_pow-2.0);
    return std::pow( std::fabs(1.0 - std::pow( c_const,(2.0/n_pow) ) * std::pow(l,(alpha_pow) ) ), (1.0/(alpha_pow) ) );
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


  // rin = 12.0
  // n_pow = 0.45
  Real rc = r_peak; //40.0

  //initialize random numbers
  std::mt19937_64 generator;
  std::uniform_real_distribution<Real> uniform(-0.02, std::nextafter(0.02, std::numeric_limits<Real>::max()));


  // Get ratio of specific heats
  Real gamma_adi = peos->GetGamma();
  Real gam = gamma_adi;

  Real a = 0;






  AthenaArray<bool> in_torus; 
  in_torus.NewAthenaArray(ku+1,ju+1,iu+1);
  
  // Prepare scratch arrays
  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);
  g_tmp.NewAthenaArray(NMETRIC);
  gi_tmp.NewAthenaArray(NMETRIC);
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {

        // Calculate Boyer-Lindquist coordinates of cell
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,0, &r,
            &theta, &phi);


        Real lambda_sol = std::sqrt(-gphiphi(r,a,theta)/gtt(r,a,theta) ) ;

        Real l_sol = c_const * std::pow( lambda_sol, n_pow);
        
        Real denom_sq = -( gitt(r,a,theta) - 2.0*l_sol*gitphi(r,a,theta) + SQR(l_sol)*giphiphi(r,a,theta) );
        Real ud_t,eps; 
        if (denom_sq>0){
          ud_t = -1.0/std::sqrt(denom_sq);
          eps = 1.0/gam * (ud_t_in * f(lin,c_const,n_pow)/(ud_t * f(l_sol,c_const,n_pow)) -1.0);

          if (std::isnan(eps)){
            fprintf(stderr,"eps is NAN! \n r theta phi: %g %g %g \n ud_t_in: %g f_in: %g ud_t: %g f: %g \n n_pow: %g c_const: %g l: %g lin: %g \n",r,theta,phi, ud_t_in,f(lin,c_const,n_pow),ud_t,f(l_sol,c_const,n_pow),n_pow,c_const,l_sol,lin);
            exit(0);
          }
        }
        else{
          ud_t = -1.0;
          eps = -1.0;
        }

         // Determine if we are in the torus
        Real rho_sol, ug_sol,pgas_sol;
        Real uu_t_sol,uu_phi_sol;
        if (eps<0 or r<rin) {
          in_torus(k,j,i) = false;

          rho_sol = 0.0;
          ug_sol = 0.0;
          pgas_sol = 0.0;
          uu_t_sol = 1.0;
          uu_phi_sol = 0.0;
        }
        else{
          in_torus(k,j,i) = true;

          rho_sol = std::pow( (eps * (gam-1.0)/k_adi), (1.0/(gam-1.0)) ) ;

          ug_sol = eps * rho_sol;
          pgas_sol = ug_sol * (gam-1.0);

          Real Omega = l_sol / SQR(  lambda_sol) ;

          //g_mu_nu u^mu u^nu = -1
          // g_tt u^t^2 + 2*g_tphi* u^t u^phi + g_phiphi * u^phi^2 = -1
          // g_tt + 2 g_tpih * Omega + g_phiphi*Omega**2 = -1/u^t^2 
          // u^t = sqrt( -1/ (g_tt + 2 g_tpih * Omega + g_phiphi*Omega**2)  )

          uu_t_sol = std::sqrt( -1.0/ (gtt(r,a,theta) + 2.0*gtphi(r,a,theta) * Omega + gphiphi(r,a,theta) * SQR( Omega) )  );
          uu_phi_sol = uu_t_sol * Omega;
        }

        // Calculate background primitives
        Real rho = rho_min * std::pow(r, rho_pow);
        Real pgas = pgas_min * std::pow(r, pgas_pow);
        Real uu1 = 0.0;
        Real uu2 = 0.0;
        Real uu3 = 0.0;

        Real perturbation = 0.0;
        // Overwrite primitives inside torus
        if (in_torus(k,j,i) ) {

          int seed = Globals::my_rank * block_size.nx1*block_size.nx2*block_size.nx3+ (k - ks) * block_size.nx2 * block_size.nx1 + (j - js) * block_size.nx1 + i - is;
          generator.seed(seed);
          perturbation = uniform(generator);

          // Calculate thermodynamic variables
          rho = rho_sol /rho_peak;
          pgas = pgas_sol / rho_peak;

          // Calculate velocities in Boyer-Lindquist coordinates
          Real u0_bl, u1_bl, u2_bl, u3_bl;
          // CalculateVelocityInTiltedTorus(r, theta, phi, &u0_bl, &u1_bl, &u2_bl, &u3_bl);

          u0_bl = uu_t_sol;
          u1_bl = 0.0;
          u2_bl = 0.0;
          u3_bl = uu_phi_sol;

          // Transform to preferred coordinates
          Real u0, u1, u2, u3;
          TransformVector(u0_bl, 0.0, u2_bl, u3_bl, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), a,&u0, &u1, &u2, &u3);
          uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
          uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
          uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;

          // fprintf(stderr,"In Torus\n r theta phi: %g %g %g \n rho %g press: %g uu: %g %g %g %g \n v: %g %g %g \n eps: %g rho_peak: %g \n ",r,theta,phi,rho,pgas,u0,u1,u2,u3,uu1,uu2,uu3,eps,rho_peak);
        }

        // Set primitive values, including cylindrically symmetric radial velocity
        // perturbations
        Real rr = r * std::sin(theta); 
        Real z = r * std::cos(theta);
        Real amp_rel = 0.0;
        if (in_torus(k,j,i)) {
          amp_rel = pert_amp * std::sin(pert_kr*rr) * std::cos(pert_kz*z);
        }
        Real amp_abs = amp_rel * uu3;
        Real pert_uur = rr/r * amp_abs;
        Real pert_uutheta = std::cos(theta)/r * amp_abs;
        //fprintf(stderr,"xyz: %g %g %g \n r th ph: %g %g %g in_torus: %d \n",pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),r,theta,phi, in_torus);
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas * (1 + perturbation);
        phydro->w(IVX,k,j,i) = phydro->w1(IM1,k,j,i) = uu1 + pert_uur;
        phydro->w(IVY,k,j,i) = phydro->w1(IM2,k,j,i) = uu2 + pert_uutheta;
        phydro->w(IVZ,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }
    }
  }

  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();
  g_tmp.DeleteAthenaArray();
  gi_tmp.DeleteAthenaArray();


  AthenaArray<Real> &g_ = ruser_meshblock_data[0];
  AthenaArray<Real> &gi_ = ruser_meshblock_data[1];


  // Initialize magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {

    // Prepare arrays of vector potential values
    AthenaArray<Real> a_phi_edges, a_phi_cells;
    AthenaArray<Real> a_theta_0, a_theta_1, a_theta_2, a_theta_3;
    AthenaArray<Real> a_phi_0, a_phi_1, a_phi_2, a_phi_3;
    a_phi_edges.NewAthenaArray(ku+2,ju+2, iu+2);
    a_phi_cells.NewAthenaArray(ku+1,ju+1, iu+1);
    Real normalization;

    // Calculate vector potential in normal case
    if (field_config == normal) {

      // Calculate edge-centered vector potential values for untilted disks
        for (int k = kl; k<=ku+1; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          for (int i = il; i <= iu+1; ++i) {
            Real r, theta, phi;
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));

                Real press = phydro->w(IPR,k,j,i);
                Real press_cutoff = std::max(press-potential_cutoff*pgas_over_rho_peak, static_cast<Real>(0.0));

                Real scaled_theta = (theta-potential_theta_min)/(potential_theta_max-potential_theta_min);
                if (theta<potential_theta_min || theta>potential_theta_max) a_phi_edges(k,j,i)=0.0;
                else a_phi_edges(k,j,i) = std::pow(r, potential_r_pow)
                    * std::pow(rho_cutoff, potential_rho_pow)
                    * std::pow(std::sin(PI * scaled_theta),potential_sinth_pow)
                    * std::pow(std::cos(PI * scaled_theta),potential_costh_pow);
                // else a_phi_edges(k,j,i) = std::pow(r, potential_r_pow)
                //     * std::pow(press_cutoff, potential_rho_pow)
                //     * std::pow(std::sin(PI * scaled_theta),potential_sinth_pow)
                //     * std::pow(std::cos(PI * scaled_theta),potential_costh_pow);
              }
             }
            }
          }
        }

      // Calculate cell-centered vector potential values for untilted disks
        for (int k = kl; k<=ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
          for (int i = il; i <= iu; ++i) {
            Real r, theta, phi;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));

                Real press = phydro->w(IPR,k,j,i);
                Real press_cutoff = std::max(press-potential_cutoff*pgas_over_rho_peak, static_cast<Real>(0.0));

                Real scaled_theta = (theta-potential_theta_min)/(potential_theta_max-potential_theta_min);
                if (theta<potential_theta_min || theta>potential_theta_max) a_phi_cells(k,j,i)=0.0;
                else a_phi_cells(k,j,i) = std::pow(r, potential_r_pow)
                    * std::pow(rho_cutoff, potential_rho_pow)
                    * std::pow(std::sin(PI * scaled_theta),potential_sinth_pow)
                    * std::pow(std::cos(PI * scaled_theta),potential_costh_pow);
                // else a_phi_cells(k,j,i) = std::pow(r, potential_r_pow)
                //     * std::pow(press_cutoff, potential_rho_pow)
                //     * std::pow(std::sin(PI * scaled_theta),potential_sinth_pow)
                //     * std::pow(std::cos(PI * scaled_theta),potential_costh_pow);
              }
            }
            }
          }
        }



      // Calculate magnetic field normalization
      // if (beta_min < 0.0) {
      //   normalization = 0.0;
      // } else {
      //   Real beta_min_actual = CalculateBetaMin();
      //   normalization = std::sqrt(beta_min_actual/beta_min);
      // }

        normalization = 1.0 * extra_field_norm;

    // Calculate vector potential in renormalized case
    } 
    else if (field_config == multi_loop) {

      // Calculate edge-centered vector potential values for untilted disks
        for (int k = kl; k<=ku+1; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          for (int i = il; i <= iu+1; ++i) {
            Real r, theta, phi;
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));

                Real scaled_theta = (theta-potential_theta_min)/(potential_theta_max-potential_theta_min);
                if (theta<potential_theta_min || theta>potential_theta_max) a_phi_edges(k,j,i)=0.0;
                a_phi_edges(k,j,i) = std::pow(r, potential_r_pow)
                    * std::pow(rho_cutoff, potential_rho_pow)
                    * std::pow(std::sin(N_loops_theta * PI * scaled_theta),1)
                    * std::pow(std::sin(PI * scaled_theta),potential_sinth_pow)
                    * std::pow(std::cos(PI * scaled_theta),potential_costh_pow)
                    * std::sin(PI * (r-rin)/loop_radius)
                    * std::exp(-r/potential_r_exp_cut)
                    * std::exp( -4*SQR(theta-PI/2.0)/SQR(potential_theta_scale_height));
              }
             }
            }
          }
        }

      // Calculate cell-centered vector potential values for untilted disks
        for (int k = kl; k<=ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
          for (int i = il; i <= iu; ++i) {
            Real r, theta, phi;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                Real scaled_theta = (theta-potential_theta_min)/(potential_theta_max-potential_theta_min);
                if (theta<potential_theta_min || theta>potential_theta_max) a_phi_cells(k,j,i)=0.0;
                a_phi_cells(k,j,i) = std::pow(r, potential_r_pow)
                    * std::pow(rho_cutoff, potential_rho_pow)
                    * std::pow(std::sin(N_loops_theta * PI * scaled_theta),1)
                    * std::pow(std::sin(PI * scaled_theta),potential_sinth_pow)
                    * std::pow(std::cos(PI * scaled_theta),potential_costh_pow) 
                    * std::sin(PI * (r-rin)/loop_radius)
                    * std::exp(-r/potential_r_exp_cut)
                    * std::exp( -4*SQR(theta-PI/2.0)/SQR(potential_theta_scale_height));
              }
            }
            }
          }
        }



      // Calculate magnetic field normalization
      // if (beta_min < 0.0) {
      //   normalization = 0.0;
      // } else {
      //   Real beta_min_actual = CalculateBetaMin();
      //   normalization = std::sqrt(beta_min_actual/beta_min);
      // }

        normalization = 1.0 * extra_field_norm;

    // Calculate vector potential in renormalized case
    } else if (field_config == MAD){
      // Calculate edge-centered vector potential values for untilted disks
        for (int k = kl; k<=ku+1; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          for (int i = il; i <= iu+1; ++i) {
            Real r, theta, phi;
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                a_phi_edges(k,j,i) = std::max( std::pow(r/20.0, 3.0) * std::pow(std::sin(theta),3.0) 
                    * rho * std::exp(-r/400.0)-0.2 ,static_cast<Real>(0.0)) ;
              }
             }
            }
          }
        }

      // Calculate cell-centered vector potential values for untilted disks
        for (int k = kl; k<=ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
          for (int i = il; i <= iu; ++i) {
            Real r, theta, phi;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                a_phi_cells(k,j,i) = std::max( std::pow(r/20.0, 3.0) * std::pow(std::sin(theta),3.0) 
                    * rho * std::exp(-r/400.0)-0.2 ,static_cast<Real>(0.0)) ;
              }
            }
            }
          }
        }



      // // Calculate magnetic field normalization
      // if (beta_min < 0.0) {
      //   normalization = 0.0;
      // } else {
      //   Real beta_min_actual = CalculateBetaMin();
      //   normalization = std::sqrt(beta_min_actual/beta_min);
      // }

      normalization = 0.5715/7.8780470912524105 * std::sqrt(10.0) * extra_field_norm ;

    }
    else {
      std::stringstream msg;
      msg << "### FATAL ERROR in Problem Generator\n"
          << "field_config must be \"normal\" or \"MAD\" or \"multi_loop\" " << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }


      // Set B^1
      for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
          pcoord->Face1Metric(k, j, il, iu+1,g_, gi_);
          for (int i = il; i <= iu+1; ++i) {


            // Prepare scratch arrays
            AthenaArray<Real> g_scratch;
            g_scratch.NewAthenaArray(NMETRIC);

            for (int n = 0; n < NMETRIC; ++n) g_scratch(n) = g_(n,i);
 
            Real det = Determinant(g_scratch); 

            g_scratch.DeleteAthenaArray();

            //d Az /dy
            Real tmp, Az_2,Az_1;
            TransformAphi(a_phi_edges(k,j+1,i),pcoord->x1f(i), pcoord->x2f(j+1),pcoord->x3v(k),a,
                &tmp,&tmp,&Az_2);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2f(j),pcoord->x3v(k),a,
                &tmp,&tmp,&Az_1);
                  

            pfield->b.x1f(k,j,i) = 1.0/std::sqrt(-det) * (Az_2-Az_1) / (pcoord->dx2f(j) );

            //d Ay/dz
            Real  Ay_2,Ay_1;
            TransformAphi(a_phi_edges(k+1,j,i),pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k+1), a,
                &tmp,&Ay_2,&tmp);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k), a,
                &tmp,&Ay_1,&tmp);

            pfield->b.x1f(k,j,i) -= 1.0/std::sqrt(-det) * (Ay_2-Ay_1) / (pcoord->dx3f(k) );

            pfield->b.x1f(k,j,i) *= normalization;

          }
        }
      }

      // Set B^2
      for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          pcoord->Face2Metric(k, j, il, iu,g_, gi_);
          for (int i = il; i <= iu; ++i) {

            // Prepare scratch arrays
            AthenaArray<Real> g_scratch; 
            g_scratch.NewAthenaArray(NMETRIC);

            for (int n = 0; n < NMETRIC; ++n) g_scratch(n) = g_(n,i);
 
            Real det = Determinant(g_scratch);

            g_scratch.DeleteAthenaArray();

            //d Ax /dz
            Real tmp, Ax_2,Ax_1;
            TransformAphi(a_phi_edges(k+1,j,i),pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k+1),a,
                &Ax_2,&tmp,&tmp);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k), a, 
                &Ax_1,&tmp,&tmp);
                  

            pfield->b.x2f(k,j,i) = 1.0/std::sqrt(-det) * (Ax_2-Ax_1) / (pcoord->dx3f(k) );

            //d Az/dx
            Real Az_2,Az_1;
            TransformAphi(a_phi_edges(k,j,i+1),pcoord->x1f(i+1), pcoord->x2f(j),pcoord->x3v(k), a, 
                &tmp,&tmp,&Az_2);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2f(j),pcoord->x3v(k),a,
                &tmp,&tmp,&Az_1);

            pfield->b.x2f(k,j,i) -= 1.0/std::sqrt(-det) * (Az_2-Az_1) / (pcoord->dx1f(i) );

            pfield->b.x2f(k,j,i) *= normalization;
                  
          }
        }
      }

      // Set B^3
      for (int k = kl; k <= ku+1; ++k) {
        for (int j = jl; j <= ju; ++j) {
          pcoord->Face3Metric(k, j, il, iu+1,g_, gi_);
          for (int i = il; i <= iu; ++i) {

            // Prepare scratch arrays
            AthenaArray<Real> g_scratch;
            g_scratch.NewAthenaArray(NMETRIC);

            for (int n = 0; n < NMETRIC; ++n) g_scratch(n) = g_(n,i);
 
            Real det = Determinant(g_scratch);

            g_scratch.DeleteAthenaArray();

            //d Ay /dx
            Real tmp, Ay_2,Ay_1;
            TransformAphi(a_phi_edges(k,j,i+1),pcoord->x1f(i+1), pcoord->x2v(j),pcoord->x3f(k),a,
                &tmp,&Ay_2,&tmp);
            TransformAphi(a_phi_edges(k,j,i),  pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k),a,
                &tmp,&Ay_1,&tmp);
                  

            pfield->b.x3f(k,j,i) = 1.0/std::sqrt(-det) * (Ay_2-Ay_1) / (pcoord->dx1f(i) );

            //d Ax/dy
            Real Ax_2,Ax_1;
            TransformAphi(a_phi_edges(k,j+1,i),pcoord->x1v(i), pcoord->x2f(j+1),pcoord->x3f(k),a,
                &Ax_2,&tmp,&tmp);
            TransformAphi(a_phi_edges(k,j,i),  pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k),a,
                &Ax_1,&tmp,&tmp);

            pfield->b.x3f(k,j,i) -= 1.0/std::sqrt(-det) * (Ax_2-Ax_1) / (pcoord->dx2f(j) );

            pfield->b.x3f(k,j,i) *= normalization;
              

          }
        }
      }

    

  

    // Free vector potential arrays
      a_phi_edges.DeleteAthenaArray();
      a_phi_cells.DeleteAthenaArray();
  }

  // Impose density and pressure floors
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,0, &r,
            &theta, &phi);
        Real &rho = phydro->w(IDN,k,j,i);
        Real &pgas = phydro->w(IEN,k,j,i);
        rho = std::max(rho, rho_min * std::pow(r, rho_pow));
        pgas = std::max(pgas, pgas_min * std::pow(r, pgas_pow));
        phydro->w1(IDN,k,j,i) = rho;
        phydro->w1(IEN,k,j,i) = pgas;
      }
    }
  }

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

  in_torus.DeleteAthenaArray();

  // Call user work function to set output variables
  UserWorkInLoop();
  return;
}

void  MeshBlock::PreserveDivbNewMetric(ParameterInput *pin){

return;
}


void set_orbit_arrays(std::string orbit_file_name){
      FILE *input_file;
        if ((input_file = fopen(orbit_file_name.c_str(), "r")) == NULL)   
               fprintf(stderr, "Cannot open %s, %s\n", "input_file",orbit_file_name.c_str());



    fscanf(input_file, "%i %f \n", &nt, &q);
    // int nt = 10;
    q = 1.0;

       
    fprintf(stderr,"nt in set_orbit_arrays: %d \n q in set_orbit_arrays: %g \n", nt,q);

    t_orbits.NewAthenaArray(nt);
    orbit_array.NewAthenaArray(Norbit,nt);



    int iorbit, it;
    for (it=0; it<nt; it++) {

      fread( &t_orbits(it), sizeof( Real ), 1, input_file );

      for (iorbit=0; iorbit<Norbit; iorbit++){

        fread( &orbit_array(iorbit,it), sizeof( Real ), 1, input_file );
      }

    }

    for (it=0; it<nt; it++) t_orbits(it) = t_orbits(it) + t0;

      for (it=0; it<nt; it++){
        orbit_array(IA2X,it) *= q;
        orbit_array(IA2Y,it) *= q;
        orbit_array(IA2Z,it) *= q;
      }

    t0_orbits = t_orbits(0);
    dt_orbits = t_orbits(1) - t_orbits(0);
        

  fclose(input_file);

  // fprintf(stderr,"Done reading orbit file \n");

}

void get_orbit_quantities(Real t, AthenaArray<Real>&orbit_quantities){

  for (int iorbit=0; iorbit<Norbit; iorbit++){
    interp_orbits(t,iorbit,orbit_array,&orbit_quantities(iorbit));
  }

  return;

}




void get_free_fall_solution(Real r, Real x1, Real x2, Real x3, Real ax_, Real ay_, Real az_, Real *uut, Real *uux1,
                                         Real *uux2, Real *uux3) {
   
    
    Real ax = ax_;
    Real ay = ay_;
    Real az = az_;

    Real amag = std::sqrt( SQR(ax) + SQR(ay) + SQR(az) );

    Real aperp = std::sqrt( SQR(ax) + SQR(ay) );

    Real a_dot_x = ax * (x1) + ay * (x2) + az * (x3);


    // these are in coordinates aligned with spin r,phi

    //(a^2*r^4 + r^6 + a^4*z^2 + a^2*r^2*z^2 + 2*a^2*r^3 - 2*a^2*r*z^2 - 2*sqrt(2)*sqrt(a^2 + r^2)*r^(7/2))/((r^4 + a^2*z^2)*(a^2 + r^2 - 2*r))
    Real numerator = std::pow(amag, 2) * std::pow(r, 4) + std::pow(r, 6) + std::pow(amag, 2) * std::pow(a_dot_x, 2) +  pow(r, 2) * pow(a_dot_x, 2) 
                        + 2 * std::pow(amag, 2) * std::pow(r, 3) - 2 *  r * std::pow(a_dot_x, 2) - 2 * std::sqrt(2) * std::sqrt(std::pow(amag, 2) + std::pow(r, 2)) * std::pow(r, 3.5);
    Real denominator = ( std::pow(r, 4) + SQR(a_dot_x) ) * ( std::pow(amag, 2) + std::pow(r, 2) - 2 * r + 1e-10);
    *uut =  numerator / denominator;

    Real uur = -std::sqrt(2.0)*std::sqrt( SQR(amag) + SQR(r) ) * std::pow(r,5.0/2.0)/( SQR(SQR(r)) + SQR(a_dot_x));
    Real uuphi = -2 * amag * SQR(r)*r /(( SQR(SQR(r)) + SQR(a_dot_x))*(std::sqrt(2.0)*std::sqrt( SQR(amag) + SQR(r))*std::sqrt(r) + 2*r));
    

    Real th_temp = std::acos( a_dot_x/ (amag * r) );

    // AthenaArray<Real> g_ks;
    // g_ks.NewAthenaArray(NMETRIC);
    // ks_metric(r,th_temp,amag,g_ks);

    //     // Extract metric coefficients
    // const Real &g_00 = g_ks(I00);
    // const Real &g_01 = g_ks(I01);
    // const Real &g_02 = g_ks(I02);
    // const Real &g_03 = g_ks(I03);
    // const Real &g_10 = g_ks(I01);
    // const Real &g_11 = g_ks(I11);
    // const Real &g_12 = g_ks(I12);
    // const Real &g_13 = g_ks(I13);
    // const Real &g_20 = g_ks(I02);
    // const Real &g_21 = g_ks(I12);
    // const Real &g_22 = g_ks(I22);
    // const Real &g_23 = g_ks(I23);
    // const Real &g_30 = g_ks(I03);
    // const Real &g_31 = g_ks(I13);
    // const Real &g_32 = g_ks(I23);
    // const Real &g_33 = g_ks(I33);

    // // Set lowered components
    // Real ud_0 = g_00*( *uut) + g_01*uur  + g_03*uuphi;
    // Real ud_1 = g_10*( *uut) + g_11*uur  + g_13*uuphi;
    // Real ud_2 = g_20*( *uut) + g_21*uur  + g_23*uuphi;
    // Real ud_3 = g_30*( *uut) + g_31*uur  + g_33*uuphi;

    // Real E = ud_0;
    // Real L = ud_3;
    // Real udotu = (*uut)*ud_0 + uur*ud_1 + uuphi*ud_3;


    // //  CHECK if this is actually a free fall solution!! //
    // Real rh = ( 1.0 + std::sqrt(SQR(1.0)-SQR(amag)) );
    // if (r> 0.8*rh){
    //   if ( ( std::fabs(E+1)>1e-2) or (std::fabs(L)>1e-2) or (fabs(udotu+1)>1e-2) ){

    //     fprintf(stderr, "Original KS coordinates \n E: %g L: %g udotu: %g \n  r: %g thprime: %g \n u: %g %g %g %g \n",
    //       E,L,udotu,r,th_temp, (*uut),uur,0,uuphi );
    //     exit(0);

    //   }
    // }

    // g_ks.DeleteAthenaArray();








    Real dx_du,dx_dv,dx_dw;
    Real dy_du,dy_dv,dy_dw;
    Real dz_du,dz_dv,dz_dw;
    Real u,v,w;
    if (aperp<1e-4){
      dx_du = 1.0;
      dx_dv = 0.0;
      dx_dw = 0.0;

      dy_du = 0.0;
      dy_dv = 1.0;
      dy_dw = 0.0;

      dz_du = 0.0;
      dz_dv = 0.0; 
      dz_dw = 1.0;

      u = x1;
      v = x2;
      w = x3;

    }
    else{
      dx_du = ay/aperp;
      dx_dv = ax*az/(aperp*amag);
      dx_dw = ax/amag;

      dy_du = -ax/aperp;
      dy_dv = ay*az/(aperp*amag);
      dy_dw = ay/amag;

      dz_du = 0.0;
      dz_dv = -aperp/amag; 
      dz_dw = az/amag;

      u = ay*x1/aperp - ax*x2/aperp;
      v = ax*az*x1/(aperp*amag) + ay*az*x2/(aperp*amag) - aperp*x3/amag;
      w = ax*x1/amag + ay*x2/amag + az*x3/amag;
    }


    // call u,v,w the coordinates of aligned frame
    Real rsq_p_asq = ( SQR(amag) + SQR(r) );
    Real du_dr = (r*u + amag*v)/rsq_p_asq;
    Real dv_dr = (r*v - amag*u)/rsq_p_asq;
    Real dw_dr =  w/(r + 1e-10);

    Real du_dphi = -v;
    Real dv_dphi = u;
    Real dw_dphi = 0.0;

    Real uuu = uur * du_dr + uuphi * du_dphi;
    Real uuv = uur * dv_dr + uuphi * dv_dphi;
    Real uuw = uur * dw_dr + uuphi * dw_dphi;


    // AthenaArray<Real> g_cks_unrotated;
    // g_cks_unrotated.NewAthenaArray(NMETRIC);

    // Real R_tmp = std::sqrt( SQR(u) + SQR(v) + SQR(w) );
    // Real r_tmp = std::sqrt( SQR(R_tmp) - SQR(amag) + std::sqrt( SQR(SQR(R_tmp) - SQR(amag)) + 4.0*SQR(amag*w) )  )/std::sqrt(2.0);

    // unboosted_cks_metric(1.0,u,v,w, r_tmp, R_tmp , 0,0,0,0,0,amag,g_cks_unrotated);
    // //unboosted_cks_metric(Real q_rat,Real xprime, Real yprime, Real zprime, Real rprime, Real Rprime, Real vx, Real vy, Real vz,Real ax, Real ay, Real az,AthenaArray<Real> &g_unboosted ){


    //     // Extract metric coefficients
    // const Real &g00 = g_cks_unrotated(I00);
    // const Real &g01 = g_cks_unrotated(I01);
    // const Real &g02 = g_cks_unrotated(I02);
    // const Real &g03 = g_cks_unrotated(I03);
    // const Real &g10 = g_cks_unrotated(I01);
    // const Real &g11 = g_cks_unrotated(I11);
    // const Real &g12 = g_cks_unrotated(I12);
    // const Real &g13 = g_cks_unrotated(I13);
    // const Real &g20 = g_cks_unrotated(I02);
    // const Real &g21 = g_cks_unrotated(I12);
    // const Real &g22 = g_cks_unrotated(I22);
    // const Real &g23 = g_cks_unrotated(I23);
    // const Real &g30 = g_cks_unrotated(I03);
    // const Real &g31 = g_cks_unrotated(I13);
    // const Real &g32 = g_cks_unrotated(I23);
    // const Real &g33 = g_cks_unrotated(I33);

    // // Set lowered components
    // ud_0 = g00*(*uut)  + g01*uuu + g02*uuv + g03*uuw;
    // ud_1 = g10*(*uut)  + g11*uuu + g12*uuv + g13*uuw;
    // ud_2 = g20*(*uut)  + g21*uuu + g22*uuv + g23*uuw;
    // ud_3 = g30*(*uut)  + g31*uuu + g32*uuv + g33*uuw;

    // E = ud_0;
    // L = ud_3;
    // udotu = (*uut)*ud_0 + uuu*ud_1 + uuv*ud_2 + uuw*ud_3;


    // //  CHECK if this is actually a free fall solution!! //
    // if (r_tmp> 0.8*rh){
    //   if ( ( std::fabs(E+1)>1e-2) or (fabs(udotu+1)>1e-2) ){

    //     fprintf(stderr, "Unrotated CKS coordinates \n E: %g L: %g udotu: %g \n  r: %g th: %g \n u: %g %g %g %g \n a: %g %g %g \n uvw: %g %g %g \n xyz: %g %g %g \ng: %g %g %g %g \n",
    //       E,L,udotu,r_tmp,th_temp, (*uut),uuu,uuv,uuw,ax,ay,az,u,v,w,x1,x2,x3,g_cks_unrotated(I00), g_cks_unrotated(I01),g_cks_unrotated(I02),g_cks_unrotated(I03));
    //     exit(0);

    //   }
    // }

    // g_cks_unrotated.DeleteAthenaArray();

    *uux1 = uuu * dx_du + uuv * dx_dv + uuw * dx_dw;
    *uux2 = uuu * dy_du + uuv * dy_dv + uuw * dy_dw;
    *uux3 = uuu * dz_du + uuv * dz_dv + uuw * dz_dw;


    return;

  }





/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar, const FaceField &bb_old){


  Real r,th,ph;
  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];

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
  // Real r_inner_boundary = rh*0.95;

  Real rh2 = ( q + std::sqrt( SQR(q) - SQR(a2)) );



   for (int k=kl; k<=ku; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=jl; j<=ju; ++j) {
      pmb->pcoord->CellMetric(k, j, il, iu, g, gi);
#pragma simd
      for (int i=il; i<=iu; ++i) {



          Real x = pmb->pcoord->x1v(i);
          Real y = pmb->pcoord->x2v(j);
          Real z = pmb->pcoord->x3v(k);
          Real t = pmb->pmy_mesh->metric_time;

          Real xprime,yprime,zprime,rprime,Rprime;

          get_prime_coords(1,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);

          Real thprime,phiprime;
          GetBoyerLindquistCoordinates(xprime,yprime,zprime,a1x,a1y,a1z, &rprime, &thprime, &phiprime);

          // Real xprime2,yprime2,zprime2,rprime2,Rprime2;

          // get_prime_coords(2,x,y,z, orbit_quantities,&xprime2,&yprime2, &zprime2, &rprime2,&Rprime2);

          // Real thprime2,phiprime2;
          // GetBoyerLindquistCoordinates(xprime2,yprime2,zprime2,a2x,a2y,a2z, &rprime2, &thprime2, &phiprime2);


          // const Real &a11 = g(I00,i);
          // const Real &a12 = g(I01,i);
          // const Real &a13 = g(I02,i);
          // const Real &a14 = g(I03,i);
          // const Real &a21 = g(I01,i);
          // const Real &a22 = g(I11,i);
          // const Real &a23 = g(I12,i);
          // const Real &a24 = g(I13,i);
          // const Real &a31 = g(I02,i);
          // const Real &a32 = g(I12,i);
          // const Real &a33 = g(I22,i);
          // const Real &a34 = g(I23,i);
          // const Real &a41 = g(I03,i);
          // const Real &a42 = g(I13,i);
          // const Real &a43 = g(I23,i);
          // const Real &a44 = g(I33,i);
          // Real det = a11 * Determinant(a22, a23, a24, a32, a33, a34, a42, a43, a44)
          //          - a12 * Determinant(a21, a23, a24, a31, a33, a34, a41, a43, a44)
          //          + a13 * Determinant(a21, a22, a24, a31, a32, a34, a41, a42, a44)
          //          - a14 * Determinant(a21, a22, a23, a31, a32, a33, a41, a42, a43);

          // if (prim(IPR,k,j,i)>1e10){
          //   fprintf(stderr,"rho: %g %g %g %g %g %g %g press: %g %g %g %g %g %g %g \n rho_old: %g %g %g %g %g %g %g \n press_old: %g %g %g %g %g %g %g \n xyz: %g %g %g \n xyzprime1: %g %g %g rprime1: %g \n xyzprime2: %g %g %g rprime2: %g \n fake_bsq: %g %g %g %g %g %g %g \n g: %g %g %g %g %g %g %g %g %g %g \n gi: %g %g %g %g %g %g %g %g %g %g \n fake_vs: %g %g %g %g %g %g %g \n DETERMINANT: %g \n",
          //     prim(IDN,k,j,i),prim(IDN,k+1,j,i),prim(IDN,k-1,j,i),prim(IDN,k,j+1,i),prim(IDN,k,j-1,i),
          //     prim(IDN,k,j,i+1),prim(IDN,k,j,i-1),
          //     prim(IPR,k,j,i),prim(IPR,k+1,j,i),prim(IPR,k-1,j,i),prim(IPR,k,j+1,i),prim(IPR,k,j-1,i),
          //     prim(IPR,k,j,i+1),prim(IPR,k,j,i-1),
          //     prim_old(IDN,k,j,i),prim_old(IDN,k+1,j,i),prim_old(IDN,k-1,j,i),prim_old(IDN,k,j+1,i),prim_old(IDN,k,j-1,i),
          //     prim_old(IDN,k,j,i+1),prim_old(IDN,k,j,i-1),
          //     prim_old(IPR,k,j,i),prim_old(IPR,k+1,j,i),prim_old(IPR,k-1,j,i),prim_old(IPR,k,j+1,i),prim_old(IPR,k,j-1,i),
          //     prim_old(IPR,k,j,i+1),prim_old(IPR,k,j,i-1),
          //     x,y,z,xprime,yprime,zprime,rprime,
          //     xprime2,yprime2,zprime2,rprime2,
          //     SQR(pmb->pfield->bcc(IB1,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)),
          //     SQR(pmb->pfield->bcc(IB1,k,j,i+1)) + SQR(pmb->pfield->bcc(IB2,k,j,i+1)) + SQR(pmb->pfield->bcc(IB3,k,j,i+1)),
          //     SQR(pmb->pfield->bcc(IB1,k,j,i-1)) + SQR(pmb->pfield->bcc(IB2,k,j,i-1)) + SQR(pmb->pfield->bcc(IB3,k,j,i-1)),
          //     SQR(pmb->pfield->bcc(IB1,k,j+1,i)) + SQR(pmb->pfield->bcc(IB2,k,j+1,i)) + SQR(pmb->pfield->bcc(IB3,k,j+1,i)),
          //     SQR(pmb->pfield->bcc(IB1,k,j-1,i)) + SQR(pmb->pfield->bcc(IB2,k,j-1,i)) + SQR(pmb->pfield->bcc(IB3,k,j-1,i)),
          //     SQR(pmb->pfield->bcc(IB1,k-1,j,i)) + SQR(pmb->pfield->bcc(IB2,k-1,j,i)) + SQR(pmb->pfield->bcc(IB3,k-1,j,i)),
          //     SQR(pmb->pfield->bcc(IB1,k+1,j,i)) + SQR(pmb->pfield->bcc(IB2,k+1,j,i)) + SQR(pmb->pfield->bcc(IB3,k+1,j,i)),
          //     g(I00,i),g(I01,i),g(I02,i),g(I03,i),g(I11,i),g(I12,i),g(I13,i),g(I22,i),g(I23,i), g(I33,i),
          //     gi(I00,i),gi(I01,i),gi(I02,i),gi(I03,i),gi(I11,i),gi(I12,i),gi(I13,i),gi(I22,i),gi(I23,i), gi(I33,i),
          //     SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)),
          //     SQR(prim(IVX,k+1,j,i)) + SQR(prim(IVY,k+1,j,i)) + SQR(prim(IVZ,k+1,j,i)),
          //     SQR(prim(IVX,k-1,j,i)) + SQR(prim(IVY,k-1,j,i)) + SQR(prim(IVZ,k-1,j,i)),
          //     SQR(prim(IVX,k,j+1,i)) + SQR(prim(IVY,k,j+1,i)) + SQR(prim(IVZ,k,j+1,i)),
          //     SQR(prim(IVX,k,j-1,i)) + SQR(prim(IVY,k,j-1,i)) + SQR(prim(IVZ,k,j-1,i)),
          //     SQR(prim(IVX,k,j,i+1)) + SQR(prim(IVY,k,j,i+1)) + SQR(prim(IVZ,k,j,i+1)),
          //     SQR(prim(IVX,k,j,i-1)) + SQR(prim(IVY,k,j,i-1)) + SQR(prim(IVZ,k,j,i-1)),
          //     det
              // );


          //     fprintf(stderr,"u_rho: %g %g %g %g %g %g %g u_en: %g %g %g %g %g %g %g \n rho_old: %g %g %g %g %g %g %g \n press_old: %g %g %g %g %g %g %g \n xyz: %g %g %g \n xyzprime1: %g %g %g rprime1: %g \n xyzprime2: %g %g %g rprime2: %g \n fake_bsq: %g %g %g %g %g %g %g \n g: %g %g %g %g %g %g %g %g %g %g \n gi: %g %g %g %g %g %g %g %g %g %g \n fake_Msq: %g %g %g %g %g %g %g \n",
          //     pmb->phydro->u(IDN,k,j,i),pmb->phydro->u(IDN,k+1,j,i),pmb->phydro->u(IDN,k-1,j,i),pmb->phydro->u(IDN,k,j+1,i),pmb->phydro->u(IDN,k,j-1,i),
          //     pmb->phydro->u(IDN,k,j,i+1),pmb->phydro->u(IDN,k,j,i-1),
          //     pmb->phydro->u(IPR,k,j,i),pmb->phydro->u(IPR,k+1,j,i),pmb->phydro->u(IPR,k-1,j,i),pmb->phydro->u(IPR,k,j+1,i),pmb->phydro->u(IPR,k,j-1,i),
          //     pmb->phydro->u(IPR,k,j,i+1),pmb->phydro->u(IPR,k,j,i-1),
          //     prim_old(IDN,k,j,i),prim_old(IDN,k+1,j,i),prim_old(IDN,k-1,j,i),prim_old(IDN,k,j+1,i),prim_old(IDN,k,j-1,i),
          //     prim_old(IDN,k,j,i+1),prim_old(IDN,k,j,i-1),
          //     prim_old(IPR,k,j,i),prim_old(IPR,k+1,j,i),prim_old(IPR,k-1,j,i),prim_old(IPR,k,j+1,i),prim_old(IPR,k,j-1,i),
          //     prim_old(IPR,k,j,i+1),prim_old(IPR,k,j,i-1),
          //     x,y,z,xprime,yprime,zprime,rprime,
          //     xprime2,yprime2,zprime2,rprime2,
          //     SQR(pmb->pfield->bcc(IB1,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)),
          //     SQR(pmb->pfield->bcc(IB1,k,j,i+1)) + SQR(pmb->pfield->bcc(IB2,k,j,i+1)) + SQR(pmb->pfield->bcc(IB3,k,j,i+1)),
          //     SQR(pmb->pfield->bcc(IB1,k,j,i-1)) + SQR(pmb->pfield->bcc(IB2,k,j,i-1)) + SQR(pmb->pfield->bcc(IB3,k,j,i-1)),
          //     SQR(pmb->pfield->bcc(IB1,k,j+1,i)) + SQR(pmb->pfield->bcc(IB2,k,j+1,i)) + SQR(pmb->pfield->bcc(IB3,k,j+1,i)),
          //     SQR(pmb->pfield->bcc(IB1,k,j-1,i)) + SQR(pmb->pfield->bcc(IB2,k,j-1,i)) + SQR(pmb->pfield->bcc(IB3,k,j-1,i)),
          //     SQR(pmb->pfield->bcc(IB1,k-1,j,i)) + SQR(pmb->pfield->bcc(IB2,k-1,j,i)) + SQR(pmb->pfield->bcc(IB3,k-1,j,i)),
          //     SQR(pmb->pfield->bcc(IB1,k+1,j,i)) + SQR(pmb->pfield->bcc(IB2,k+1,j,i)) + SQR(pmb->pfield->bcc(IB3,k+1,j,i)),
          //     g(I00,i),g(I01,i),g(I02,i),g(I03,i),g(I11,i),g(I12,i),g(I13,i),g(I22,i),g(I23,i), g(I33,i),
          //     gi(I00,i),gi(I01,i),gi(I02,i),gi(I03,i),gi(I11,i),gi(I12,i),gi(I13,i),gi(I22,i),gi(I23,i), gi(I33,i),
          //     SQR(pmb->phydro->u(IVX,k,j,i)) + SQR(pmb->phydro->u(IVY,k,j,i)) + SQR(pmb->phydro->u(IVZ,k,j,i)),
          //     SQR(pmb->phydro->u(IVX,k+1,j,i)) + SQR(pmb->phydro->u(IVY,k+1,j,i)) + SQR(pmb->phydro->u(IVZ,k+1,j,i)),
          //     SQR(pmb->phydro->u(IVX,k-1,j,i)) + SQR(pmb->phydro->u(IVY,k-1,j,i)) + SQR(pmb->phydro->u(IVZ,k-1,j,i)),
          //     SQR(pmb->phydro->u(IVX,k,j+1,i)) + SQR(pmb->phydro->u(IVY,k,j+1,i)) + SQR(pmb->phydro->u(IVZ,k,j+1,i)),
          //     SQR(pmb->phydro->u(IVX,k,j-1,i)) + SQR(pmb->phydro->u(IVY,k,j-1,i)) + SQR(pmb->phydro->u(IVZ,k,j-1,i)),
          //     SQR(pmb->phydro->u(IVX,k,j,i+1)) + SQR(pmb->phydro->u(IVY,k,j,i+1)) + SQR(pmb->phydro->u(IVZ,k,j,i+1)),
          //     SQR(pmb->phydro->u(IVX,k,j,i-1)) + SQR(pmb->phydro->u(IVY,k,j,i-1)) + SQR(pmb->phydro->u(IVZ,k,j,i-1))
          //     );
          //   exit(0);
          // }


          // if is_face_in_boundary(1, i,j,k, a1x,a1y,a1z,a2x,a2y, a2z,rh,rh2,pmb ){
          //   pmb->pfield->b.x1f(k,j,i) = bb_old.x1f(k,j,i);
          // }
          // if is_face_in_boundary(2, i,j,k, a1x,a1y,a1z,a2x,a2y, a2z,rh,rh2,pmb ){
          //   pmb->pfield->b.x2f(k,j,i) = bb_old.x2f(k,j,i);
          // }
          // if is_face_in_boundary(3, i,j,k, a1x,a1y,a1z,a2x,a2y, a2z,rh,rh2,pmb ){
          //   pmb->pfield->b.x3f(k,j,i) = bb_old.x3f(k,j,i);
          // }



          if (rprime < rh*0.8) {
            rprime = rh*0.8;
            convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a1x,a1y,a1z,&xprime,&yprime,&zprime);
          }

          if (rprime < rh){

              Real bsq_over_rho_max = 1.0;
              Real beta_floor = 0.2;
              

              //u^r partial/partialr   partial/partialr = partial x/partialr partial/partialx + ...

              //light 2g_r_t u^r u^t + g_tt u^t^2 + g_rr u^r^2 = 0

              //v_r^2 g_rr + 2 v_r g_t_r + g_tt  = 0

              // v_r = (- 2 g_tr +/ sqrt(4g_tr^2 - 4g_rr g_tt))/g_rr

              // uu_cks  = (A, B cos(phi)sin(th), B sin(phi)sin(th),Bcos(phi) )
              // g_munu uu_cks^mu uu_cks^nu = -1

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


              get_free_fall_solution(rprime, xprime,yprime, zprime, a1x,a1y,a1z, &u0, &u1,&u2,&u3);

              // AthenaArray<Real> g_unboosted;
              // g_unboosted.NewAthenaArray(NMETRIC);
              // unboosted_cks_metric(1.0,xprime,yprime, zprime, rprime, Rprime, orbit_quantities(IV1X), orbit_quantities(IV1Y), orbit_quantities(IV1Z),a1x,a1y,a1z,g_unboosted );

              // // Extract metric coefficients
              // const Real &g_00 = g_unboosted(I00);
              // const Real &g_01 = g_unboosted(I01);
              // const Real &g_02 = g_unboosted(I02);
              // const Real &g_03 = g_unboosted(I03);
              // const Real &g_10 = g_unboosted(I01);
              // const Real &g_11 = g_unboosted(I11);
              // const Real &g_12 = g_unboosted(I12);
              // const Real &g_13 = g_unboosted(I13);
              // const Real &g_20 = g_unboosted(I02);
              // const Real &g_21 = g_unboosted(I12);
              // const Real &g_22 = g_unboosted(I22);
              // const Real &g_23 = g_unboosted(I23);
              // const Real &g_30 = g_unboosted(I03);
              // const Real &g_31 = g_unboosted(I13);
              // const Real &g_32 = g_unboosted(I23);
              // const Real &g_33 = g_unboosted(I33);

              // // Set lowered components
              // Real ud_0 = g_00*u0 + g_01*u1 + g_02*u2 + g_03*u3;
              // Real ud_1 = g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3;
              // Real ud_2 = g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3;
              // Real ud_3 = g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3;

              // Real E = ud_0;
              // Real L = ud_3;
              // Real udotu = u0*ud_0 + u1*ud_1 + u2*ud_2 + u3*ud_3;


              // //  CHECK if this is actually a free fall solution!! //
              // if (rprime > 0.8*rh){
              //   if ( ( std::fabs(E+1)>1e-2)  or (fabs(udotu+1)>1e-2) ){

              //     fprintf(stderr, "E: %g L: %g udotu: %g \n xyz: %g %g %g\n rprime: %g thprime: %g phiprime: %g \n u: %g %g %g %g \n",
              //       E,L,udotu,xprime,yprime,zprime,rprime,thprime,phiprime, u0,u1,u2,u3 );
              //     exit(0);

              //   }
              // }

              // g_unboosted.DeleteAthenaArray();

              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(1,t,u0,u1,u2,u3, orbit_quantities,&u0prime,&u1prime,&u2prime,&u3prime);

              // Extract metric coefficients
              const Real &g00_ = g(I00,i);
              const Real &g01_ = g(I01,i);
              const Real &g02_ = g(I02,i);
              const Real &g03_ = g(I03,i);
              const Real &g10_ = g(I01,i);
              const Real &g11_  = g(I11,i);
              const Real &g12_  = g(I12,i);
              const Real &g13_  = g(I13,i);
              const Real &g20_  = g(I02,i);
              const Real &g21_  = g(I12,i);
              const Real &g22_  = g(I22,i);
              const Real &g23_  = g(I23,i);
              const Real &g30_  = g(I03,i);
              const Real &g31_  = g(I13,i);
              const Real &g32_  = g(I23,i);
              const Real &g33_  = g(I33,i);

              // Set lowered components
              Real ud_0 = g00_ *u0prime + g01_ *u1prime + g02_ *u2prime + g03_ *u3prime;
              Real ud_1 = g10_ *u0prime + g11_ *u1prime + g12_ *u2prime + g13_ *u3prime;
              Real ud_2 = g20_ *u0prime + g21_ *u1prime + g22_ *u2prime + g23_ *u3prime;
              Real ud_3 = g30_ *u0prime + g31_ *u1prime + g32_ *u2prime + g33_ *u3prime;

              // Real E = ud_0;
              // Real L = ud_3;
              // Real udotu = u0prime*ud_0 + u1prime*ud_1 + u2prime*ud_2 + u3prime*ud_3;


              Real git_ui = g01_ *u1prime + g02_ *u2prime + g03_ *u3prime;
              Real gij_ui_uj = g(I11,i)*u1prime*u1prime + 2.0*g(I12,i)*u1prime*u2prime + 2.0*g(I13,i)*u1prime*u3prime
                       + g(I22,i)*u2prime*u2prime + 2.0*g(I23,i)*u2prime*u3prime
                       + g(I33,i)*u3prime*u3prime;
              Real a_const = g00_*SQR(u0prime) -2.0*g00_*SQR(u0prime) + SQR(g00_*u0prime) * gij_ui_uj/SQR(git_ui);
              Real b_const = 2.0 * g00_*u0prime * gij_ui_uj/SQR(git_ui) - 2.0*u0prime;
              Real c_const = (gij_ui_uj/SQR(git_ui) + 1.0);

              Real A_const = (- b_const - std::sqrt( SQR(b_const) - 4.0 * a_const*c_const ) )/ (2*a_const);
              Real B_const = -1.0 / (git_ui) * (1.0 + A_const * g00_ * u0prime);

              // Real constant = g00_*SQR(A_const*u0prime) + 2.0*A_const*B_const *git_ui*u0prime + SQR(B_const)*gij_ui_uj;

              u0prime *= A_const; //1.0/std::sqrt(-udotu) ;
              u1prime *= B_const; //1.0/std::sqrt(-udotu) ;
              u2prime *= B_const; //1.0/std::sqrt(-udotu) ;
              u3prime *= B_const; //1.0/std::sqrt(-udotu) ;


              // ud_0 = g00_ *u0prime + g01_ *u1prime + g02_ *u2prime + g03_ *u3prime;
              // ud_1 = g10_ *u0prime + g11_ *u1prime + g12_ *u2prime + g13_ *u3prime;
              // ud_2 = g20_ *u0prime + g21_ *u1prime + g22_ *u2prime + g23_ *u3prime;
              // ud_3 = g30_ *u0prime + g31_ *u1prime + g32_ *u2prime + g33_ *u3prime;


              // E = ud_0;
              // L = ud_3;
              // udotu = u0prime*ud_0 + u1prime*ud_1 + u2prime*ud_2 + u3prime*ud_3;


              // //  CHECK if this is actually a free fall solution!! //
              // if (rprime > 0.8*rh){
              //   // if ( ( std::fabs(E+1)>1e-2)  or (fabs(udotu+1)>1e-2) ){

              //     fprintf(stderr, "First Boosted BH E: %g L: %g udotu: %g \n xyz: %g %g %g\n rprime: %g thprime: %g phiprime: %g \n u: %g %g %g %g \n a_const: %g b_const: %g c_const: %g A_const: %g B_const: %g Equation_constant: %g \n ",
              //       E,L,udotu,xprime,yprime,zprime,rprime,thprime,phiprime, u0prime,u1prime,u2prime,u3prime,a_const,b_const,c_const,A_const,B_const ,constant);

              //   // }
              // }



              uu1 = u1prime - gi(I01,i) / gi(I00,i) * u0prime;
              uu2 = u2prime - gi(I02,i) / gi(I00,i) * u0prime;
              uu3 = u3prime - gi(I03,i) / gi(I00,i) * u0prime;

              
              prim(IDN,k,j,i) = dfloor;

              // Real dceiling = 1e3;
              // Real Pceiling = 1e3;

              // if (prim(IDN,k,j,i)>dceiling)prim(IDN,k,j,i)=dceiling;
              // if (prim(IPR,k,j,i)>Pceiling)prim(IPR,k,j,i)=Pceiling;

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


              if (gamma>gamma_max){

                fprintf(stderr,"gamma: %g rprime: %g xyzprime: %g %g %g \n a1: %g %g %g \n",gamma,rprime,xprime,yprime,zprime,a1x,a1y,a1z);

                // Real ratio = gamma_max/gamma;

                // u0prime *= ratio;
                // u1prime *= ratio;
                // u2prime *= ratio;
                // u3prime *= ratio;

                // gamma = gamma_max;


                // uu1 = u1prime - gi(I01,i) / gi(I00,i) * u0prime;
                // uu2 = u2prime - gi(I02,i) / gi(I00,i) * u0prime;
                // uu3 = u3prime - gi(I03,i) / gi(I00,i) * u0prime;


                // prim(IVX,k,j,i) = uu1;
                // prim(IVY,k,j,i) = uu2;
                // prim(IVZ,k,j,i) = uu3;

              }


              // if (gamma>1e3){
              //   fprintf(stderr, "HUGE gamma in horizon 1: %g \n xyzprime: %g %g %g rprime: %g \n", gamma,xprime,yprime,zprime,rprime);
              // }
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



          get_prime_coords(2,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);

          // Real thprime,phiprime;
          GetBoyerLindquistCoordinates(xprime,yprime,zprime,a2x,a2y,a2z, &rprime, &thprime, &phiprime);


          if (rprime < rh2*0.8) {
            rprime = rh2*0.8;
            convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a2x,a2y,a2z,&xprime,&yprime,&zprime);
          }

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



              get_free_fall_solution(rprime, xprime,yprime, zprime, a2x,a2y,a2z, &u0, &u1,&u2,&u3);

              // AthenaArray<Real> g_unboosted;
              // g_unboosted.NewAthenaArray(NMETRIC);
              // unboosted_cks_metric(1.0,xprime,yprime, zprime, rprime, Rprime, orbit_quantities(IV2X), orbit_quantities(IV2Y), orbit_quantities(IV2Z),a2x,a2y,a2z,g_unboosted );

              // // Extract metric coefficients
              // const Real &g00 = g_unboosted(I00);
              // const Real &g01 = g_unboosted(I01);
              // const Real &g02 = g_unboosted(I02);
              // const Real &g03 = g_unboosted(I03);
              // const Real &g10 = g_unboosted(I01);
              // const Real &g11 = g_unboosted(I11);
              // const Real &g12 = g_unboosted(I12);
              // const Real &g13 = g_unboosted(I13);
              // const Real &g20 = g_unboosted(I02);
              // const Real &g21 = g_unboosted(I12);
              // const Real &g22 = g_unboosted(I22);
              // const Real &g23 = g_unboosted(I23);
              // const Real &g30 = g_unboosted(I03);
              // const Real &g31 = g_unboosted(I13);
              // const Real &g32 = g_unboosted(I23);
              // const Real &g33 = g_unboosted(I33);

              // // Set lowered components
              // Real ud_0 = g00*u0 + g01*u1 + g02*u2 + g03*u3;
              // Real ud_1 = g10*u0 + g11*u1 + g12*u2 + g13*u3;
              // Real ud_2 = g20*u0 + g21*u1 + g22*u2 + g23*u3;
              // Real ud_3 = g30*u0 + g31*u1 + g32*u2 + g33*u3;

              // Real E = ud_0;
              // Real L = ud_3;
              // Real udotu = u0*ud_0 + u1*ud_1 + u2*ud_2 + u3*ud_3;


              // //  CHECK if this is actually a free fall solution!! //
              // if (rprime > 0.8*rh2){
              //   if ( ( std::fabs(E+1)>1e-2)  or (fabs(udotu+1)>1e-2) ){

              //     fprintf(stderr, "Second BH E: %g L: %g udotu: %g \n xyz: %g %g %g\n rprime: %g thprime: %g phiprime: %g \n u: %g %g %g %g \n",
              //       E,L,udotu,xprime,yprime,zprime,rprime,thprime,phiprime, u0,u1,u2,u3 );
              //     exit(0);

              //   }
              // }

              // g_unboosted.DeleteAthenaArray();

              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(2,t,u0,u1,u2,u3, orbit_quantities,&u0prime,&u1prime,&u2prime,&u3prime);

              // AthenaArray<Real> g_boosted;
              // g_boosted.NewAthenaArray(NMETRIC);
              // boosted_BH_metric_addition(1.0,xprime,yprime,zprime, rprime, Rprime, orbit_quantities(IV2X), orbit_quantities(IV2Y), orbit_quantities(IV2Z),a2x,a2y,a2z, g_boosted );

              // g_boosted(I00) += -1.0;
              // g_boosted(I11) += 1.0;
              // g_boosted(I22) += 1.0;
              // g_boosted(I33) += 1.0;

              //               // Extract metric coefficients
              // const Real &g_00 = g_boosted(I00);
              // const Real &g_01 = g_boosted(I01);
              // const Real &g_02 = g_boosted(I02);
              // const Real &g_03 = g_boosted(I03);
              // const Real &g_10 = g_boosted(I01);
              // const Real &g_11 = g_boosted(I11);
              // const Real &g_12 = g_boosted(I12);
              // const Real &g_13 = g_boosted(I13);
              // const Real &g_20 = g_boosted(I02);
              // const Real &g_21 = g_boosted(I12);
              // const Real &g_22 = g_boosted(I22);
              // const Real &g_23 = g_boosted(I23);
              // const Real &g_30 = g_boosted(I03);
              // const Real &g_31 = g_boosted(I13);
              // const Real &g_32 = g_boosted(I23);
              // const Real &g_33 = g_boosted(I33);

              // // Set lowered components
              // ud_0 = g_00*u0prime + g_01*u1prime + g_02*u2prime + g_03*u3prime;
              // ud_1 = g_10*u0prime + g_11*u1prime + g_12*u2prime + g_13*u3prime;
              // ud_2 = g_20*u0prime + g_21*u1prime + g_22*u2prime + g_23*u3prime;
              // ud_3 = g_30*u0prime + g_31*u1prime + g_32*u2prime + g_33*u3prime;

              // E = ud_0;
              // L = ud_3;
              // udotu = u0prime*ud_0 + u1prime*ud_1 + u2prime*ud_2 + u3prime*ud_3;


              // //  CHECK if this is actually a free fall solution!! //
              // // if (rprime > 0.8*rh2){
              // //   if ( ( std::fabs(E+1)>1e-2)  or (fabs(udotu+1)>1e-2) ){

              // //     fprintf(stderr, "Second BH boosted and isolated E: %g L: %g udotu: %g \n xyz: %g %g %g\n rprime: %g thprime: %g phiprime: %g \n u: %g %g %g %g \n",
              // //       E,L,udotu,xprime,yprime,zprime,rprime,thprime,phiprime, u0prime,u1prime,u2prime,u3prime );
              // //     exit(0);

              // //   }
              // // }
              // g_boosted.DeleteAthenaArray();

              // //Make sure four vector is normalized
              // Real c_const = 1.0 + g(I11,i)*u1prime*u1prime + 2.0*g(I12,i)*u1prime*u2prime+ 2.0*g(I13,i)*u1prime*u3prime
              //          + g(I22,i)*u2prime*u2prime + 2.0*g(I23,i)*u2prime*u3prime
              //          + g(I33,i)*u3prime*u3prime;

              // Real b_const = 2.0 * ( g(I01,i)*u1prime + g(I02,i)*u2prime + g(I03,i)*u3prime );

              // Real a_const = g(I00,i);

              // if (std::fabs(a_const)<std::numeric_limits<double>::epsilon()){
              //   u0prime = -c_const/b_const;

              // }
              // else{
              //   u0prime = (-b_const + std::sqrt( SQR(b_const) - 4.0*a_const*c_const ) )/(2.0*a_const);
              // }

 

               // Extract metric coefficients
              const Real &g00_ = g(I00,i);
              const Real &g01_ = g(I01,i);
              const Real &g02_ = g(I02,i);
              const Real &g03_ = g(I03,i);
              const Real &g10_ = g(I01,i);
              const Real &g11_  = g(I11,i);
              const Real &g12_  = g(I12,i);
              const Real &g13_  = g(I13,i);
              const Real &g20_  = g(I02,i);
              const Real &g21_  = g(I12,i);
              const Real &g22_  = g(I22,i);
              const Real &g23_  = g(I23,i);
              const Real &g30_  = g(I03,i);
              const Real &g31_  = g(I13,i);
              const Real &g32_  = g(I23,i);
              const Real &g33_  = g(I33,i);

              // Set lowered components
              Real ud_0 = g00_ *u0prime + g01_ *u1prime + g02_ *u2prime + g03_ *u3prime;
              Real ud_1 = g10_ *u0prime + g11_ *u1prime + g12_ *u2prime + g13_ *u3prime;
              Real ud_2 = g20_ *u0prime + g21_ *u1prime + g22_ *u2prime + g23_ *u3prime;
              Real ud_3 = g30_ *u0prime + g31_ *u1prime + g32_ *u2prime + g33_ *u3prime;

              // E = ud_0;
              // L = ud_3;
              // udotu = u0prime*ud_0 + u1prime*ud_1 + u2prime*ud_2 + u3prime*ud_3;


              Real git_ui = g01_ *u1prime + g02_ *u2prime + g03_ *u3prime;
              Real gij_ui_uj = g(I11,i)*u1prime*u1prime + 2.0*g(I12,i)*u1prime*u2prime + 2.0*g(I13,i)*u1prime*u3prime
                       + g(I22,i)*u2prime*u2prime + 2.0*g(I23,i)*u2prime*u3prime
                       + g(I33,i)*u3prime*u3prime;
              Real a_const = g00_*SQR(u0prime) -2.0*g00_*SQR(u0prime) + SQR(g00_*u0prime) * gij_ui_uj/SQR(git_ui);
              Real b_const = 2.0 * g00_*u0prime * gij_ui_uj/SQR(git_ui) - 2.0*u0prime;
              Real c_const = (gij_ui_uj/SQR(git_ui) + 1.0);

              Real A_const = (- b_const - std::sqrt( SQR(b_const) - 4.0 * a_const*c_const ) )/ (2*a_const);
              Real B_const = -1.0 / (git_ui) * (1.0 + A_const * g00_ * u0prime);

              // Real constant = g00_*SQR(A_const*u0prime) + 2.0*A_const*B_const *git_ui*u0prime + SQR(B_const)*gij_ui_uj;

              u0prime *= A_const; //1.0/std::sqrt(-udotu) ;
              u1prime *= B_const; //1.0/std::sqrt(-udotu) ;
              u2prime *= B_const; //1.0/std::sqrt(-udotu) ;
              u3prime *= B_const; //1.0/std::sqrt(-udotu) ;


              // ud_0 = g00_ *u0prime + g01_ *u1prime + g02_ *u2prime + g03_ *u3prime;
              // ud_1 = g10_ *u0prime + g11_ *u1prime + g12_ *u2prime + g13_ *u3prime;
              // ud_2 = g20_ *u0prime + g21_ *u1prime + g22_ *u2prime + g23_ *u3prime;
              // ud_3 = g30_ *u0prime + g31_ *u1prime + g32_ *u2prime + g33_ *u3prime;

              // E = ud_0;
              // L = ud_3;
              // udotu = u0prime*ud_0 + u1prime*ud_1 + u2prime*ud_2 + u3prime*ud_3;


              // //  CHECK if this is actually a free fall solution!! //
              // if (rprime > 0.8*rh){
              //   // if ( ( std::fabs(E+1)>1e-2)  or (fabs(udotu+1)>1e-2) ){

              //     fprintf(stderr, "Second Boosted BH E: %g L: %g udotu: %g \n xyz: %g %g %g\n rprime: %g thprime: %g phiprime: %g \n u: %g %g %g %g \n a_const: %g b_const: %g c_const: %g A_const: %g B_const: %g Equation_constant: %g \n ",
              //       E,L,udotu,xprime,yprime,zprime,rprime,thprime,phiprime, u0prime,u1prime,u2prime,u3prime,a_const,b_const,c_const,A_const,B_const ,constant);

              //   // }
              // }



              uu1 = u1prime - gi(I01,i) / gi(I00,i) * u0prime;
              uu2 = u2prime - gi(I02,i) / gi(I00,i) * u0prime;
              uu3 = u3prime - gi(I03,i) / gi(I00,i) * u0prime;

              

              // Real dceiling = 1e3;
              // Real Pceiling = 1e3;

              // if (prim(IDN,k,j,i)>dceiling)prim(IDN,k,j,i)=dceiling;
              // if (prim(IPR,k,j,i)>Pceiling)prim(IPR,k,j,i)=Pceiling;
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


              if (gamma>gamma_max){

                fprintf(stderr,"gamma: %g rprime: %g xyzprime: %g %g %g \n a2: %g %g %g \n",gamma,rprime,xprime,yprime,zprime,a2x,a2y,a2z);

                // Real ratio = gamma_max/gamma;

                // u0prime *= ratio;
                // u1prime *= ratio;
                // u2prime *= ratio;
                // u3prime *= ratio;

                // gamma = gamma_max;


                // uu1 = u1prime - gi(I01,i) / gi(I00,i) * u0prime;
                // uu2 = u2prime - gi(I02,i) / gi(I00,i) * u0prime;
                // uu3 = u3prime - gi(I03,i) / gi(I00,i) * u0prime;


                // prim(IVX,k,j,i) = uu1;
                // prim(IVY,k,j,i) = uu2;
                // prim(IVZ,k,j,i) = uu3;

              }



              // if (gamma>1e3){
              //   fprintf(stderr, "HUGE gamma in horizon 2: %g \n xyzprime: %g %g %g rprime: %g \n", gamma,xprime,yprime,zprime,rprime);
              // }
              // user_out_var(0,k,j,i) = gamma;

              // Calculate 4-velocity
              alpha = std::sqrt(-1.0/gi(I00,i));
              u0 = gamma/alpha;
              u1 = uu1 - alpha * gamma * gi(I01,i);
              u2 = uu2 - alpha * gamma * gi(I02,i);
              u3 = uu3 - alpha * gamma * gi(I03,i);
              Real u_0, u_1, u_2, u_3;

              // pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &ud_0, &ud_1, &ud_2, &ud_3);

              // E = ud_0;
              // L = ud_3;
              // udotu = u0*ud_0 + u1*ud_1 + u2*ud_2 + u3*ud_3;


              // // //  CHECK if this is actually a free fall solution!! //
              // if (rprime > 0.8*rh){
              //   // if ( ( std::fabs(E+1)>1e-2)  or (fabs(udotu+1)>1e-2) ){

              //     fprintf(stderr, "Resulting velocities! E: %g L: %g udotu: %g \n xyz: %g %g %g\n rprime: %g thprime: %g phiprime: %g \n u: %g %g %g %g \n gamma: %g vxyz: %g %g %g\n ",
              //       E,L,udotu,xprime,yprime,zprime,rprime,thprime,phiprime, u0,u1,u2,u3,gamma,u1/u0,u2/u0,u3/u0 );

              //   // }
              // }


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


  apply_inner_boundary_condition(pmb,prim_old,prim,prim_scalar,bb_half);

  return;
}

double risco_calc_general( int do_prograde, double spin, double mass )
{
  double Z1,Z2,sign,term1,term2 ;
  double spinsq;

  spinsq = spin*spin;
  sign = (do_prograde) ? 1. : -1. ; 

  term1 = pow(1. + spin,1./3.);
  term2 = pow(1. - spin,1./3.);
  
  Z1 = 1. + term1*term2*(term1 + term2);

  Z2 = sqrt(3.*spinsq + Z1*Z1) ;

  return( mass * (3. + Z2-sign*sqrt((3. - Z1)*(3. + Z1 + 2.*Z2)))  );

}
double target_temperature_func( double r_loc, double h_o_r ,double spin, double mass)
{
  Real r, x, rm1, xm3, R_z, term1;
  Real C_f, F_f, G_f;  /* NT functions */
  Real ut_sq, uphi_sq;
  static double ut_sq_isco, uphi_sq_isco;
  static int local_first_time = 1;

  /* Set ISCO values for future use if we need them */
  Real r_isco = risco_calc_general( 1, spin, mass )/(mass+SMALL);
  r = r_isco;
  rm1 = 1./r;     x = sqrt(r);    xm3 = 1./(x*x*x);    term1 = spin*spin*rm1;

  C_f = 1. - 3*rm1 + 2*spin*xm3;
  F_f = 1. - 2*spin*xm3 + term1*rm1;
  G_f = 1. - 2*rm1 + spin*xm3 ;

  ut_sq_isco   = G_f * G_f     / C_f ;
  uphi_sq_isco = F_f * F_f * r / C_f;
  local_first_time = 0;

  r = r_loc;


  rm1 = 1./r;     x = sqrt(r);    xm3 = 1./(x*x*x);    term1 = spin*spin*rm1;

  if( r > r_isco ) { 
    C_f = 1. - 3*rm1 + 2*spin*xm3;
    F_f = 1. - 2*spin*xm3 + term1*rm1;
    G_f = 1. - 2*rm1 + spin*xm3 ;
    ut_sq   = G_f * G_f     / C_f ;
    uphi_sq = F_f * F_f * r / C_f;
    R_z = uphi_sq*rm1 - term1*(ut_sq - 1.) ;
    term1 = 0.5*M_PI*R_z*h_o_r*h_o_r*rm1;
  }
  else { 
    R_z = uphi_sq_isco*rm1 - term1*(ut_sq_isco - 1.) ;
    term1 = 0.5*M_PI*R_z*h_o_r*h_o_r*rm1;
  }
  
  return( term1 ) ;
}


void NobleCooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){



  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];

  Real gamma_adi = pmb->peos->GetGamma();

  Real k_target = 0.0005;



  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);

  Real a1 = std::sqrt( SQR(orbit_quantities(IA1X)) + SQR(orbit_quantities(IA1Y)) + SQR(orbit_quantities(IA1Z)));
  Real a2 = std::sqrt( SQR(orbit_quantities(IA2X)) + SQR(orbit_quantities(IA2Y)) + SQR(orbit_quantities(IA2Z)));



  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real radius, theta,phi;
        GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), orbit_quantities(IA1X),orbit_quantities(IA1Y),orbit_quantities(IA1Z),
          &radius,&theta, &phi);

        Real xprime,yprime,zprime,rprime,Rprime;
        get_prime_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);
        Real rhprime = ( q + std::sqrt(SQR(q)-SQR(a2)) );

        Real ug = prim(IPR,k,j,i)/(gamma_adi-1.0);

        Real Omega = 1.0/( std::pow(radius,1.5) + a1);
        Real Omega_secondary = 1.0/(q+SMALL) * 1.0/( std::pow(rprime/(q+SMALL),1.5) + a2/(q+SMALL));

        // Omega_secondary = Omega_secondary*q;

        Real r_isco = risco_calc_general( 1, a1, m );
        Real r_isco_secondary = risco_calc_general( 1, a2/(q+SMALL), q ); //neads a/M, returns isco in units of M_1

        if (radius<r_isco) Omega = 1.0/( std::pow(r_isco,1.5) + a1);
        if (rprime<r_isco_secondary) Omega_secondary = 1.0/(q+SMALL) * 1.0/( std::pow(r_isco_secondary/(q+SMALL),1.5) + a2/(q+SMALL));

        Real Target_Temperature = target_temperature_func( radius,H_over_r_target,a1,m);
        Real Target_Temperature_secondary = target_temperature_func( rprime/(q+SMALL),H_over_r_target,a2/(q+SMALL),q);

        Real Y = prim(IPR,k,j,i)/prim(IDN,k,j,i)/Target_Temperature;
        Real Y_secondary = prim(IPR,k,j,i)/prim(IDN,k,j,i)/Target_Temperature_secondary;


        Real L_cool;

        if (Target_Temperature>=Target_Temperature_secondary){
          L_cool = Omega * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        }
        else{
          L_cool = Omega_secondary * ug * std::sqrt( Y_secondary-1.0 +  std::fabs(Y_secondary-1.0) );
        }

        // if (Target_Temperature>=Target_Temperature_secondary){
        //   L_cool = Omega * ug * std::pow( Y-1.0 +  std::fabs(Y-1.0),1.0 );
        // }
        // else{
        //   L_cool = Omega_secondary * ug * std::pow( Y_secondary-1.0 +  std::fabs(Y_secondary-1.0),1.0 );
        // }
        // Real L_cool = Omega * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        // Real L_cool_secondary = 0.0; //Omega_secondary * ug * std::sqrt( Y_secondary-1.0 +  std::fabs(Y_secondary-1.0) );
        if (L_cool<0) L_cool = 0.0;
        // if (L_cool_secondary<0) L_cool_secondary = 0.0;

        // L_cool = L_cool/5.0;
        // L_cool = L_cool * 10;


        // L_cool = std::max(L_cool,L_cool_secondary);


          // Calculate normal frame Lorentz factor
        Real uu1 = prim(IM1,k,j,i);
        Real uu2 = prim(IM2,k,j,i);
        Real uu3 = prim(IM3,k,j,i);
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
        Real u_0, u_1, u_2, u_3;

        pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);



        // Do not include bsq in enthalpy
        Real Be = - ( 1.0 + ug/prim(IDN,k,j,i) + prim(IPR,k,j,i)/prim(IDN,k,j,i) ) * u_0 -1.0; 



        if (Be>0) L_cool = 0.0;

        // Calculate Boyer-Lindquist coordinates of cell
        Real rh = ( m + std::sqrt(SQR(m)-SQR(a1)) );
        if (radius < rh) L_cool = 0.0;

        if (rprime < rhprime) L_cool = 0.0;

        // Real L_cool_max = 0.2 * ug /dt; /// L_cool * dt < 0.2 * ug
        // if (L_cool> L_cool_max) L_cool = L_cool_max;

        cons(IEN,k,j,i) += -dt * L_cool * u_0; 
        cons(IM1,k,j,i) += -dt * L_cool * u_1;
        cons(IM2,k,j,i) += -dt * L_cool * u_2;
        cons(IM3,k,j,i) += -dt * L_cool * u_3;

        Real ug_frac = dt * L_cool/ug;

        pmb->user_out_var(0,k,j,i) = L_cool * u_0;

        // pmb->user_out_var(1,k,j,i) = pmb->pcoord->GetCellVolume(k,j,i)/
        //                   (pmb->pcoord->dx1f(i)*pmb->pcoord->dx2f(j)*pmb->pcoord->dx3f(k));



      }
    }
  }

  orbit_quantities.DeleteAthenaArray();

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

static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real x1,
                     Real x2, Real x3, Real a, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

  if (COORDINATE_SYSTEM == "schwarzschild") {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (COORDINATE_SYSTEM == "kerr-schild") {
    Real r = x1;
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl + a/delta * a1_bl;
  }
    else if (COORDINATE_SYSTEM == "gr_user"){
    Real x = x1;
    Real y = x2;
    Real z = x3;

    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    *pa1 = a1_bl * ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
           a2_bl * x*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
           a3_bl * y; 
    *pa2 = a1_bl * ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
           a2_bl * y*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
           a3_bl * x;
    *pa3 = a1_bl * z/r - 
           a2_bl * r * std::sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  }
  return;
}

//Transform vector potential, A_\mu, from KS to CKS coordinates assuming A_r = A_theta = 0
// A_\mu (cks) = A_nu (ks)  dx^nu (ks)/dx^\mu (cks) = A_phi (ks) dphi (ks)/dx^\mu
// phi_ks = arctan((r*y + a*x)/(r*x - a*y) ) 
//
static void TransformAphi(Real a3_ks, Real x1,
                     Real x2, Real x3, Real a, Real *pa1, Real *pa2, Real *pa3) {

  if (COORDINATE_SYSTEM == "gr_user"){
    Real x = x1;
    Real y = x2;
    Real z = x3;

    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    Real sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a);

    //dphi/dx =  partial phi/partial x + partial phi/partial r partial r/partial x 
    *pa1 = a3_ks * ( -y/(SQR(x)+SQR(y))  + a*x*r/( (SQR(a)+SQR(r))*sqrt_term ) ); 
    //dphi/dx =  partial phi/partial y + partial phi/partial r partial r/partial y 
    *pa2 = a3_ks * (  x/(SQR(x)+SQR(y))  + a*y*r/( (SQR(a)+SQR(r))*sqrt_term ) ); 
    //dphi/dx =   partial phi/partial r partial r/partial z 
    *pa3 = a3_ks * ( a*z/(r*sqrt_term) );
  }
  else{
          std::stringstream msg;
      msg << "### FATAL ERROR in TransformAphi\n"
          << "this function only works for CKS coordinates"
          <<  std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  return;
}


void interp_orbits(Real t, int iorbit,AthenaArray<Real> &arr, Real *result){

    int it = (int) ((t - t0_orbits) / dt_orbits + 1000) - 1000; //Rounds down

    if (it<= 0) it = 0;
    if (it>=nt-1) it = nt-1;

    Real slope;


   if (t<t0_orbits){
      slope = (arr(iorbit,it+1)-arr(iorbit,it))/dt_orbits;
      *result = (t - t_orbits(it) ) * slope + arr(iorbit,it);
   }
   else if (it==nt-1){
      slope = (arr(iorbit,it)-arr(iorbit,it-1))/dt_orbits;
      *result = (t - t_orbits(it) ) * slope + arr(iorbit,it);
    }
    else{
      slope = (arr(iorbit,it+1)-arr(iorbit,it))/dt_orbits;
      *result = (t - t_orbits(it) ) * slope + arr(iorbit,it);
    }

    return;

}


void get_prime_coords(int BH_INDEX, Real x, Real y, Real z, AthenaArray<Real> &orbit_quantities, Real *xprime, Real *yprime, Real *zprime, Real *rprime, Real *Rprime){

  Real xbh,ybh,zbh,ax,ay,az,vxbh,vybh,vzbh;

  
  if (BH_INDEX ==1){
      xbh = orbit_quantities(IX1);
      ybh = orbit_quantities(IY1);
      zbh = orbit_quantities(IZ1);


      ax = orbit_quantities(IA1X);
      ay = orbit_quantities(IA1Y);
      az = orbit_quantities(IA1Z);

      vxbh = orbit_quantities(IV1X);
      vybh = orbit_quantities(IV1Y);
      vzbh = orbit_quantities(IV1Z);
  }
  else if (BH_INDEX ==2){
      xbh = orbit_quantities(IX2);
      ybh = orbit_quantities(IY2);
      zbh = orbit_quantities(IZ2);


      ax = orbit_quantities(IA2X);
      ay = orbit_quantities(IA2Y);
      az = orbit_quantities(IA2Z);

      vxbh = orbit_quantities(IV2X);
      vybh = orbit_quantities(IV2Y);
      vzbh = orbit_quantities(IV2Z);
  }
  else {
    fprintf(stderr,"Choose a valid BH_INDEX!!: %d \n",BH_INDEX);
    exit(0);
  }
  Real a_mag = std::sqrt( SQR(ax) + SQR(ay) + SQR(az) );


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
  if (std::fabs(a_dot_x_prime))
  *rprime = std::sqrt(*rprime/2.0);


  // if (std::isnan(*rprime) or std::isnan(*xprime) or std::isnan(*yprime) or std::isnan(*zprime) ){
  //     fprintf(stderr,"ISNAN in GetBoyer!!! \n xyz: %g %g %g \n xbh ybh zbh: %g %g %g \n ax ay az a: %g %g %g %g \n adotx: %g \n xyzprime: %g %g %g \n ",
  //       x,y,z,xbh, ybh, zbh, ax,ay,az,a_mag, a_dot_x_prime,*xprime,*yprime,*zprime );
  //     exit(0);
  //   }

  return;

}

//From BHframe to lab frame

void BoostVector(int BH_INDEX, Real t,Real a0, Real a1, Real a2, Real a3, AthenaArray<Real> &orbit_quantities, Real *pa0, Real *pa1, Real *pa2, Real *pa3){


  Real vxbh,vybh,vzbh;
  if (BH_INDEX==1){
    vxbh = orbit_quantities(IV1X);
    vybh = orbit_quantities(IV1Y);
    vzbh = orbit_quantities(IV1Z);

  }
  else if (BH_INDEX==2){
    vxbh = orbit_quantities(IV2X);
    vybh = orbit_quantities(IV2Y);
    vzbh = orbit_quantities(IV2Z);
  }
  else{
    fprintf(stderr,"Choose a valid BH_INDEX!!!: %g",BH_INDEX);
    exit(0);
  }



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



void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{


  m = pin->GetReal("coord", "m");

  //////////////Perturber Black Hole//////////////////

  t0 = pin->GetOrAddReal("problem","t0", 0.0);

  Binary_BH_Metric(t,x1,x2,x3,g,g_inv,dg_dx1,dg_dx2,dg_dx3,dg_dt,true);

  return;

}


void boosted_BH_metric_addition(Real q_rat,Real xprime, Real yprime, Real zprime, Real rprime, Real Rprime, Real vx, Real vy, Real vz,Real ax, Real ay, Real az,AthenaArray<Real> &g_pert ){

  Real a_dot_x_prime = ax * xprime + ay * yprime + az * zprime;
  Real a_mag = std::sqrt( SQR(ax) + SQR(ay) + SQR(az) );
  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime>=0)){

    Real diff = SMALL - a_dot_x_prime/(a_mag+SMALL);
    a_dot_x_prime =  SMALL;

    xprime = xprime + diff*ax/(a_mag+SMALL);
    yprime = yprime + diff*ay/(a_mag+SMALL);
    zprime = zprime + diff*az/(a_mag+SMALL);
  }
  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime <0)){

    Real diff = -SMALL - a_dot_x_prime/(a_mag+SMALL);
    a_dot_x_prime =  -SMALL;

    xprime = xprime + diff*ax/(a_mag+SMALL);
    yprime = yprime + diff*ay/(a_mag+SMALL);
    zprime = zprime + diff*az/(a_mag+SMALL);
  } 
  
  Real thprime,phiprime;
  GetBoyerLindquistCoordinates(xprime,yprime,zprime,ax,ay,az, &rprime, &thprime, &phiprime);


/// prevent metric from getting nan sqrt(-gdet)

  Real rhprime = ( q_rat + std::sqrt(SQR(q_rat)-SQR(a_mag)) );
  if (rprime < rhprime*0.8) {
    rprime = rhprime*0.8;
    convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, ax,ay,az,&xprime,&yprime,&zprime);
  }

  a_dot_x_prime = ax * xprime + ay * yprime + az * zprime;

  Real a_cross_x_prime[3];


  a_cross_x_prime[0] = ay * zprime - az * yprime;
  a_cross_x_prime[1] = az * xprime - ax * zprime;
  a_cross_x_prime[2] = ax * yprime - ay * xprime;


  Real rsq_p_asq_prime = SQR(rprime) + SQR(a_mag);

  //First calculated all quantities in BH rest (primed) frame

  Real l_lowerprime[4],l_upperprime[4];
  Real l_lowerprime_transformed[4];
  AthenaArray<Real> Lambda;

  Lambda.NewAthenaArray(NMETRIC);

  Real fprime = q_rat *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(a_dot_x_prime));
  l_upperprime[0] = -1.0;
  l_upperprime[1] = (rprime * xprime - a_cross_x_prime[0] + a_dot_x_prime * ax/rprime)/(rsq_p_asq_prime);
  l_upperprime[2] = (rprime * yprime - a_cross_x_prime[1] + a_dot_x_prime * ay/rprime)/(rsq_p_asq_prime);
  l_upperprime[3] = (rprime * zprime - a_cross_x_prime[2] + a_dot_x_prime * az/rprime)/(rsq_p_asq_prime);

  l_lowerprime[0] = 1.0;
  l_lowerprime[1] = l_upperprime[1];
  l_lowerprime[2] = l_upperprime[2];
  l_lowerprime[3] = l_upperprime[3];

  //Terms for the boost //

  Real vsq = SQR(vx) + SQR(vy) + SQR(vz);
  Real beta_mag = std::sqrt(vsq);
  Real Lorentz = std::sqrt(1.0/(1.0 - vsq));
  ///Real Lorentz = 1.0;
  Real nx = vx/beta_mag;
  Real ny = vy/beta_mag;
  Real nz = vz/beta_mag;


  // This is the inverse transformation since l_mu is lowered.  This 
  // takes a lowered vector from BH frame to lab frame.   
  Lambda(I00) =  Lorentz;
  Lambda(I01) = -Lorentz * vx;
  Lambda(I02) = -Lorentz * vy;
  Lambda(I03) = -Lorentz * vz;
  Lambda(I11) = ( 1.0 + (Lorentz - 1.0) * nx * nx );
  Lambda(I12) = (       (Lorentz - 1.0) * nx * ny ); 
  Lambda(I13) = (       (Lorentz - 1.0) * nx * nz );
  Lambda(I22) = ( 1.0 + (Lorentz - 1.0) * ny * ny ); 
  Lambda(I23) = (       (Lorentz - 1.0) * ny * nz );
  Lambda(I33) = ( 1.0 + (Lorentz - 1.0) * nz * nz );




  // Boost l_mu
  matrix_multiply_vector_lefthandside(Lambda,l_lowerprime,l_lowerprime_transformed);


  // Set covariant components
  g_pert(I00) = fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[0];
  g_pert(I01) = fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[1];
  g_pert(I02) = fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[2];
  g_pert(I03) = fprime * l_lowerprime_transformed[0]*l_lowerprime_transformed[3];
  g_pert(I11) = fprime * l_lowerprime_transformed[1]*l_lowerprime_transformed[1];
  g_pert(I12) = fprime * l_lowerprime_transformed[1]*l_lowerprime_transformed[2];
  g_pert(I13) = fprime * l_lowerprime_transformed[1]*l_lowerprime_transformed[3];
  g_pert(I22) = fprime * l_lowerprime_transformed[2]*l_lowerprime_transformed[2];
  g_pert(I23) = fprime * l_lowerprime_transformed[2]*l_lowerprime_transformed[3];
  g_pert(I33) = fprime * l_lowerprime_transformed[3]*l_lowerprime_transformed[3];


  Lambda.DeleteAthenaArray();
  return;

}


void unboosted_cks_metric(Real q_rat,Real xprime, Real yprime, Real zprime, Real rprime, Real Rprime, Real vx, Real vy, Real vz,Real ax, Real ay, Real az,AthenaArray<Real> &g_unboosted ){

  Real a_dot_x_prime = ax * xprime + ay * yprime + az * zprime;
  Real a_mag = std::sqrt( SQR(ax) + SQR(ay) + SQR(az) );
  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime>=0)){

    Real diff = SMALL - a_dot_x_prime/(a_mag+SMALL);
    a_dot_x_prime =  SMALL;

    xprime = xprime + diff*ax/(a_mag+SMALL);
    yprime = yprime + diff*ay/(a_mag+SMALL);
    zprime = zprime + diff*az/(a_mag+SMALL);
  }
  if ((std::fabs(a_dot_x_prime)<SMALL) && (a_dot_x_prime <0)){

    Real diff = -SMALL - a_dot_x_prime/(a_mag+SMALL);
    a_dot_x_prime =  -SMALL;

    xprime = xprime + diff*ax/(a_mag+SMALL);
    yprime = yprime + diff*ay/(a_mag+SMALL);
    zprime = zprime + diff*az/(a_mag+SMALL);
  } 
  
  Real thprime,phiprime;
  GetBoyerLindquistCoordinates(xprime,yprime,zprime,ax,ay,az, &rprime, &thprime, &phiprime);


/// prevent metric from getting nan sqrt(-gdet)

  Real rhprime = ( q_rat + std::sqrt(SQR(q_rat)-SQR(a_mag)) );
  if (rprime < rhprime*0.8) {
    rprime = rhprime*0.8;
    convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, ax,ay,az,&xprime,&yprime,&zprime);
  }

  a_dot_x_prime = ax * xprime + ay * yprime + az * zprime;

  Real a_cross_x_prime[3];


  a_cross_x_prime[0] = ay * zprime - az * yprime;
  a_cross_x_prime[1] = az * xprime - ax * zprime;
  a_cross_x_prime[2] = ax * yprime - ay * xprime;


  Real rsq_p_asq_prime = SQR(rprime) + SQR(a_mag);

  //First calculated all quantities in BH rest (primed) frame

  Real l_lowerprime[4],l_upperprime[4];

  Real fprime = q_rat *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(a_dot_x_prime));
  l_upperprime[0] = -1.0;
  l_upperprime[1] = (rprime * xprime - a_cross_x_prime[0] + a_dot_x_prime * ax/rprime)/(rsq_p_asq_prime);
  l_upperprime[2] = (rprime * yprime - a_cross_x_prime[1] + a_dot_x_prime * ay/rprime)/(rsq_p_asq_prime);
  l_upperprime[3] = (rprime * zprime - a_cross_x_prime[2] + a_dot_x_prime * az/rprime)/(rsq_p_asq_prime);

  l_lowerprime[0] = 1.0;
  l_lowerprime[1] = l_upperprime[1];
  l_lowerprime[2] = l_upperprime[2];
  l_lowerprime[3] = l_upperprime[3];


  // Set covariant components
  g_unboosted(I00) = -1.0 + fprime * l_lowerprime[0]*l_lowerprime[0];
  g_unboosted(I01) = fprime * l_lowerprime[0]*l_lowerprime[1];
  g_unboosted(I02) = fprime * l_lowerprime[0]*l_lowerprime[2];
  g_unboosted(I03) = fprime * l_lowerprime[0]*l_lowerprime[3];
  g_unboosted(I11) = 1.0 + fprime * l_lowerprime[1]*l_lowerprime[1];
  g_unboosted(I12) = fprime * l_lowerprime[1]*l_lowerprime[2];
  g_unboosted(I13) = fprime * l_lowerprime[1]*l_lowerprime[3];
  g_unboosted(I22) = 1.0 + fprime * l_lowerprime[2]*l_lowerprime[2];
  g_unboosted(I23) = fprime * l_lowerprime[2]*l_lowerprime[3];
  g_unboosted(I33) = 1.0 + fprime * l_lowerprime[3]*l_lowerprime[3];


  return;

}


void ks_metric(Real r, Real th,Real a,AthenaArray<Real> &g_ks ){

  Real m = 1.0;

  Real a2 = SQR(a);
  Real sin2 = SQR( std::sin(th) ) ;
  Real cos2 = SQR( std::cos(th) );

  // Go through 1D block of cells

    // Extract remaining useful quantities
  Real r2 = SQR(r);
  Real delta = r2 - 2.0*m*r + a2;
  Real sigma = r2 + a2 * cos2;

  // Set covariant metric coefficients
  g_ks(I00) = -(1.0 - 2.0*m*r/sigma);
  g_ks(I01) = 2.0*m*r/sigma;
  g_ks(I03) = -2.0*m*a*r/sigma * sin2;
  g_ks(I11) = 1.0 + 2.0*m*r/sigma;
  g_ks(I13) = -(1.0 + 2.0*m*r/sigma) * a * sin2;
  g_ks(I22) = sigma;
  g_ks(I33) = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2;

  //First calculated all quantities in BH rest (primed) frame

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

  Real eta[4];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;

  //////////////First Black Hole//////////////////
  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(1,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);


  AthenaArray<Real> g_pert;

  g_pert.NewAthenaArray(NMETRIC);

  boosted_BH_metric_addition(1.0,xprime,yprime,zprime,rprime,Rprime, v1x,v1y,v1z, a1x,a1y,a1z,g_pert );


    // Set covariant components
  g(I00) = eta[0] + g_pert(I00);
  g(I01) =          g_pert(I01);
  g(I02) =          g_pert(I02);
  g(I03) =          g_pert(I03);
  g(I11) = eta[1] + g_pert(I11);
  g(I12) =          g_pert(I12);
  g(I13) =          g_pert(I13);
  g(I22) = eta[2] + g_pert(I22);
  g(I23) =          g_pert(I23);
  g(I33) = eta[3] + g_pert(I33);

  //////////////Second Black Hole//////////////////


  get_prime_coords(2,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);

  boosted_BH_metric_addition(q,xprime,yprime,zprime,rprime,Rprime, v2x,v2y,v2z, a2x,a2y,a2z,g_pert );

    // Set covariant components
  g(I00) += g_pert(I00);
  g(I01) += g_pert(I01);
  g(I02) += g_pert(I02);
  g(I03) += g_pert(I03);
  g(I11) += g_pert(I11);
  g(I12) += g_pert(I12);
  g(I13) += g_pert(I13);
  g(I22) += g_pert(I22);
  g(I23) += g_pert(I23);
  g(I33) += g_pert(I33);


  g_pert.DeleteAthenaArray();



  // Real det = Determinant(g);
  // if (det>=0){
  //   fprintf(stderr, "sqrt -g is nan!! xyz: %g %g %g xyzbh: %g %g %g \n xyzprime: %g %g %g \n r th phi: %g %g %g \n r th phi prime: %g %g %g \n",
  //     x,y,z,orbit_quantities(IX2),orbit_quantities(IY2),orbit_quantities(IZ2),
  //     xprime,yprime,zprime,r,th,phi,rprime,thprime,phiprime);
  //   exit(0);
  // }



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



void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt, bool take_derivatives)
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
    for (int n = 0; n < NMETRIC; ++n) {
      fprintf(stderr,"nmetric: %d metric: %g \n", n,g(n));
    }
    exit(0);

  }


  if (take_derivatives){

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
      get_prime_coords(2,x,y,z,orbit_quantities,&xprime,&yprime,&zprime,&rprime,&Rprime);

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

         // if (std::fabs(dg_dx1(n))>1e2 ){
         //  fprintf(stderr,"large dg_dx1!: %g for n= %d\n x: %g %g y: %g z: %g t: %g \n",dg_dx1(n),n,x1,x1p,x2,x3,t);
         // }
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

        // if (std::fabs(dg_dx2(n))>1e2 ){
        //   fprintf(stderr,"large dg_dx2!: %g for n= %d\n x: %g y: %g %g z: %g t: %g \n",dg_dx2(n),n,x1,x2,x2p,x3,t);
        //  }
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
        // if (std::fabs(dg_dx3(n))>1e2 ){
        //   fprintf(stderr,"large dg_dx2!: %g for n= %d\n x: %g  y: %g z: %g %g t: %g \n",dg_dx3(n),n,x1,x2,x3,x3p,t);
        //  }
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

        // if (std::fabs(dg_dt(n))>1e2 ){
        //   fprintf(stderr,"large dg_dt!: %g for n= %d\n x1: %g y: %g z: %g t: %g %g \n",dg_dt(n),n,x1,x2,x3,t,tp);
        //  }
      }

      gp.DeleteAthenaArray();
      // gm.DeleteAthenaArray();

}

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
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(n,k,j,is-i) = prim(n,k,j,is);
      }
    }}
  }

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        if (prim(IVX,k,j,is-i)>0) prim(IVX,k,j,is-i)=0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = b.x1f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = b.x2f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
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
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(n,k,j,ie+i) = prim(n,k,j,ie);
      }
    }}
  }

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        if (prim(IVX,k,j,ie+i)<0) prim(IVX,k,j,ie+i)=0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
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
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(n,k,js-j,i) = prim(n,k,js,i);
      }
    }}
  }

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (prim(IVY,k,js-j,i)>0) prim(IVY,k,js-j,i)=0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) = b.x1f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = b.x2f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) = b.x3f(k,js,i);
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
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(n,k,je+j,i) = prim(n,k,je,i);
      }
    }}
  }

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (prim(IVY,k,je+j,i)<0) prim(IVY,k,je+j,i)=0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) = b.x1f(k,(je  ),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = b.x2f(k,(je+1),i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) = b.x3f(k,(je  ),i);
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
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(n,ks-k,j,i) = prim(n,ks,j,i);
      }
    }}
  }

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (prim(IVZ,ks-k,j,i)>0) prim(IVZ,ks-k,j,i)=0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) = b.x1f(ks,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) = b.x2f(ks,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = b.x3f(ks,j,i);
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
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(n,ke+k,j,i) = prim(n,ke,j,i);
      }
    }}
  }

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (prim(IVZ,ke+k,j,i)<0) prim(IVZ,ke+k,j,i)=0;

      }
    }}
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) = b.x1f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) = b.x2f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = b.x3f((ke+1),j,i);
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


//THIS IS SUPER INEFFICIENT 
void EquationOfState::GetRadii(Real t, Real x1, Real x2, Real x3,  Real a, Real *r, Real *r2){


  // AthenaArray<Real> orbit_quantities;
  // orbit_quantities.NewAthenaArray(Norbit);

  // get_orbit_quantities(t,orbit_quantities);

  // Real x = x1;
  // Real y = x2;
  // Real z = x3;
  // Real xprime1,yprime1,zprime1,rprime1,Rprime1;
  // get_prime_coords(1,x,y,z, orbit_quantities, &xprime1,&yprime1, &zprime1, &rprime1,&Rprime1);

  // Real xprime2,yprime2,zprime2,rprime2,Rprime2;
  // get_prime_coords(2,x,y,z, orbit_quantities, &xprime2,&yprime2, &zprime2, &rprime2,&Rprime2);

  // orbit_quantities.DeleteAthenaArray();

  // (*r) = rprime1;
  // (*r2) = rprime2;


  (*r) = -1.0;
  (*r2) = -1.0;

  return;
  // Real r, th, phi;
  // GetBoyerLindquistCoordinates(x1,x2,x3,0,0,a, &r, &th, &phi);
  // return r;
}

// Real EquationOfState::GetRadius2(Real t, Real x1, Real x2, Real x3){

//   AthenaArray<Real> orbit_quantities;
//   orbit_quantities.NewAthenaArray(Norbit);

//   get_orbit_quantities(t,orbit_quantities);

//   Real x = x1;
//   Real y = x2;
//   Real z = x3;
//   Real xprime2,yprime2,zprime2,rprime2,Rprime2;
//   get_prime_coords(2,x,y,z, orbit_quantities, &xprime2,&yprime2, &zprime2, &rprime2,&Rprime2);

//   orbit_quantities.DeleteAthenaArray();

//   return rprime2;
//   // Real xprime,yprime,zprime,rprime,Rprime;
//   // get_prime_coords(2,x1,x2,x3, pmy_block_->pmy_mesh->time, &xprime,&yprime,&zprime,&rprime, &Rprime);

//   // return rprime;
// }
Real max_wave_speed_gr(int DIR, int i, int j, int k,MeshBlock *pmb,AthenaArray<Real> &w,AthenaArray<Real> &g_,AthenaArray<Real> &gi_,AthenaArray<Real> &bcc,FaceField b ) {
  Real Acov[4],Acon[4],Bcon[4],Bcov[4];

  for (int mu=0; mu<=3; ++mu){
    Acov[mu] = 0.0;
    Bcov[mu] = 0.0;
    Acon[mu] = 0.0;
    Bcon[mu] = 0.0;
  }

  Acov[DIR] = 1.0;
  Bcov[0] = 1.0;


  Acon[0] = gi_(I00,i)*Acov[0] + gi_(I01,i)*Acov[1] + gi_(I02,i)*Acov[2] + gi_(I03,i)*Acov[3];
  Acon[1] = gi_(I01,i)*Acov[0] + gi_(I11,i)*Acov[1] + gi_(I12,i)*Acov[2] + gi_(I13,i)*Acov[3];
  Acon[2] = gi_(I02,i)*Acov[0] + gi_(I12,i)*Acov[1] + gi_(I22,i)*Acov[2] + gi_(I23,i)*Acov[3];
  Acon[3] = gi_(I03,i)*Acov[0] + gi_(I13,i)*Acov[1] + gi_(I23,i)*Acov[2] + gi_(I33,i)*Acov[3];

  Bcon[0] = gi_(I00,i)*Bcov[0] + gi_(I01,i)*Bcov[1] + gi_(I02,i)*Bcov[2] + gi_(I03,i)*Bcov[3];
  Bcon[1] = gi_(I01,i)*Bcov[0] + gi_(I11,i)*Bcov[1] + gi_(I12,i)*Bcov[2] + gi_(I13,i)*Bcov[3];
  Bcon[2] = gi_(I02,i)*Bcov[0] + gi_(I12,i)*Bcov[1] + gi_(I22,i)*Bcov[2] + gi_(I23,i)*Bcov[3];
  Bcon[3] = gi_(I03,i)*Bcov[0] + gi_(I13,i)*Bcov[1] + gi_(I23,i)*Bcov[2] + gi_(I33,i)*Bcov[3];

  Real Asq = Acon[0]*Acov[0] + Acon[1]*Acov[1] + Acon[2]*Acov[2] + Acon[3]*Acov[3];
  Real Bsq = Bcon[0]*Bcov[0] + Bcon[1]*Bcov[1] + Bcon[2]*Bcov[2] + Bcon[3]*Bcov[3];


  Real uu1 = w(IVX,k,j,i);
  Real uu2 = w(IVY,k,j,i);
  Real uu3 = w(IVZ,k,j,i);
  Real tmp = g_(I11,i)*uu1*uu1 + 2.0*g_(I12,i)*uu1*uu2 + 2.0*g_(I13,i)*uu1*uu3
           + g_(I22,i)*uu2*uu2 + 2.0*g_(I23,i)*uu2*uu3
           + g_(I33,i)*uu3*uu3;
  Real gamma = std::sqrt(1.0 + tmp);

  // Calculate 4-velocity
  Real alpha = std::sqrt(-1.0/gi_(I00,i));
  Real u0 = gamma/alpha;
  Real u1 = uu1 - alpha * gamma * gi_(I01,i);
  Real u2 = uu2 - alpha * gamma * gi_(I02,i);
  Real u3 = uu3 - alpha * gamma * gi_(I03,i);

  Real Au = u0*Acov[0] + u1*Acov[1] + u2*Acov[2] + u3*Acov[3];


  Real Bu = u0*Bcov[0] + u1*Bcov[1] + u2*Bcov[2] + u3*Bcov[3];

  Real AB = Acon[0]*Bcov[0] + Acon[1]*Bcov[1] + Acon[2]*Bcov[2] + Acon[3]*Bcov[3];

  Real Au2 = Au*Au;
  Real Bu2 = Bu*Bu;
  Real AuBu = Au*Bu;

  Real b_sq;

  if (MAGNETIC_FIELDS_ENABLED) {


    Real u_0,u_1,u_2,u_3;
    pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

    // Calculate 4-magnetic field
    Real bb1 = bcc(IB1,k,j,i); /// + std::abs(b_x1f(k,j,i) - bcc(IB1,k,j,i));
    Real bb2 = bcc(IB2,k,j,i);
    Real bb3 = bcc(IB3,k,j,i);

    if (DIR==1) bb1 += std::abs(b.x1f(k,j,i) - bcc(IB1,k,j,i));
    if (DIR==2) bb2 += std::abs(b.x2f(k,j,i) - bcc(IB2,k,j,i));
    if (DIR==3) bb3 += std::abs(b.x3f(k,j,i) - bcc(IB3,k,j,i));

    Real b0 = g_(I01,i)*u0*bb1 + g_(I02,i)*u0*bb2 + g_(I03,i)*u0*bb3
            + g_(I11,i)*u1*bb1 + g_(I12,i)*u1*bb2 + g_(I13,i)*u1*bb3
            + g_(I12,i)*u2*bb1 + g_(I22,i)*u2*bb2 + g_(I23,i)*u2*bb3
            + g_(I13,i)*u3*bb1 + g_(I23,i)*u3*bb2 + g_(I33,i)*u3*bb3;
    Real b1 = (bb1 + b0 * u1) / u0;
    Real b2 = (bb2 + b0 * u2) / u0;
    Real b3 = (bb3 + b0 * u3) / u0;
    Real b_0, b_1, b_2, b_3;
    pmb->pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

    // Calculate bsq
    b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  }
  else{
    b_sq = 0.0;
  }
  Real gam = pmb->peos->GetGamma();
  // Find fast magnetosonic speed
  Real rho = w(IDN,k,j,i);
  Real u = w(IPR,k,j,i) / (gam-1.0); 
  Real ef = rho + gam*u;
  Real ee = b_sq + ef;
  Real va2 = b_sq/ee;
  Real cs2 = gam*(gam - 1.)*u/ef;

  Real cms2 = cs2 + va2 - cs2*va2;

  // Real SMALL = 1e-10;

  cms2 = (cms2 < 0) ? SMALL : cms2;
  cms2 = (cms2 > 1) ? 1 : cms2;

  // Require that speed of wave measured by observer q->ucon is cms2

  Real A = Bu2 - (Bsq + Bu2)*cms2;
  Real B = 2.*(AuBu - (AB + AuBu)*cms2);
  Real C = Au2 - (Asq + Au2)*cms2;

  Real discr = B*B - 4.*A*C;
  discr = (discr < 0.) ? 0. : discr;
  discr = std::sqrt(discr);

  Real vp = -(-B + discr)/(2.*A);
  Real vm = -(-B - discr)/(2.*A);

  Real cmax = (vp > vm) ? vp : vm;
  Real cmin = (vp > vm) ? vm : vp;

    // (*cmax)[k][j][i] = fabs(MY_MAX(MY_MAX(0., (*cmaxL)[k][j][i]), (*cmaxR)[k][j][i]));
    // (*cmin)[k][j][i] = fabs(MY_MAX(MY_MAX(0., -(*cminL)[k][j][i]), -(*cminR)[k][j][i]));
    // (*ctop)[dir][k][j][i] = MY_MAX((*cmax)[k][j][i], (*cmin)[k][j][i]);
  Real ctop = std::max(std::fabs(cmax),std::fabs(cmin));


  // AthenaArray<Real> orbit_quantities;
  // orbit_quantities.NewAthenaArray(Norbit);

  // get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);

  // Real x = pmb->pcoord->x1v(i);
  // Real y = pmb->pcoord->x2v(j);
  // Real z = pmb->pcoord->x3v(k);
  // Real xprime1,yprime1,zprime1,rprime1,Rprime1;
  // get_prime_coords(1,x,y,z, orbit_quantities, &xprime1,&yprime1, &zprime1, &rprime1,&Rprime1);
  // Real xprime2,yprime2,zprime2,rprime2,Rprime2;
  // get_prime_coords(2,x,y,z, orbit_quantities, &xprime2,&yprime2, &zprime2, &rprime2,&Rprime2);

  // if (ctop>2)
  // fprintf(stderr,"dir: %d ijk: %d %d %d \n xyz: %g %g %g \n cms2: %g ABC: %g %g %g \n Bu2: %g Au2: %g vp: %g vm: %g \n ctop: %g xyzprime1: %g %g %g rprime1: %g \n xyzprime2: %g %g %g rprime2: %g \n U: %g %g %g %g \n cmax: %g cmin: %g AB: %g Asq: %g Bsq: %g\n gis: %g %g %g %g %g %g %g \n Acov: %g %g %g %g \n Acon: %g %g %g %g \n Bcov: %g %g %g %g \n Bcon: %g %g %g %g \n",
  //   DIR, i,j,k, pmb->pcoord->x1v(i),pmb->pcoord->x2v(j),pmb->pcoord->x3v(k),cms2,A,B,C,Bu2,Au2,vp,vm,ctop,xprime1,yprime1,zprime1,rprime1,xprime2,yprime2,zprime2,rprime2,u0,u1,u2,u3,cmax,cmin,AB,Asq,Bsq,
  //   gi_(I00,i),gi_(I01,i),gi_(I11,i),gi_(I02,i),gi_(I22,i),gi_(I03,i),gi_(I33,i),
  //   Acov[0],Acov[1],Acov[2],Acov[3], Acon[0],Acon[1],Acon[2],Acon[3],
  //   Bcov[0],Bcov[1],Bcov[2],Bcov[3], Bcon[0],Bcon[1],Bcon[2],Bcon[3]);


  // orbit_quantities.DeleteAthenaArray();
  return ctop;



}


Real MyTimeStep(MeshBlock *pmb)
{
  Real min_dt=FLT_MAX;

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

  AthenaArray<Real> g, gi,dt1,dt2,dt3;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);
  dt1.NewAthenaArray(pmb->ncells1);
  dt2.NewAthenaArray(pmb->ncells1);
  dt3.NewAthenaArray(pmb->ncells1);
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CenterWidth1(k, j, pmb->is, pmb->ie, dt1);
      pmb->pcoord->CenterWidth2(k, j, pmb->is, pmb->ie, dt2);
      pmb->pcoord->CenterWidth3(k, j, pmb->is, pmb->ie, dt3);
      pmb->pcoord->CellMetric(k,j,pmb->is,pmb->ie,g,gi); 
        for (int i=pmb->is; i<=pmb->ie; ++i) {


          // Real cl1 = ( -g_(I01,i) + std::sqrt( SQR(g_(I01,i)) - g_(I00,i)*g_(I11,i) ) ) / g_(I11,i);
          // Real cl2 = ( -g_(I02,i) + std::sqrt( SQR(g_(I02,i)) - g_(I00,i)*g_(I22,i) ) ) / g_(I22,i);
          // Real cl3 = ( -g_(I03,i) + std::sqrt( SQR(g_(I03,i)) - g_(I00,i)*g_(I33,i) ) ) / g_(I33,i);

          // Real cl1 = max_wave_speed_gr(1,i,j,k,pmb,pmb->phydro->w,g,gi,pmb->pfield->bcc,pmb->pfield->b);
          // Real cl2 = max_wave_speed_gr(2,i,j,k,pmb,pmb->phydro->w,g,gi,pmb->pfield->bcc,pmb->pfield->b);
          // Real cl3 = max_wave_speed_gr(3,i,j,k,pmb,pmb->phydro->w,g,gi,pmb->pfield->bcc,pmb->pfield->b);


           // SR case: do nothing (assume maximum characteristic is c = 1)
      // GR case: divide cell widths by coordinate speed of light (not necessarily unity)

          Real speed1a = -(std::sqrt(SQR(gi(I01,i)) - gi(I00,i) * gi(I11,i))
              - gi(I01,i)) / gi(I00,i);

          Real speed1b = (std::sqrt(SQR(gi(I01,i)) - gi(I00,i) * gi(I11,i))
              - gi(I01,i)) / gi(I00,i);

          Real speed1 = std::max(std::abs(speed1a),std::abs(speed1b));

          Real speed2a = -(std::sqrt(SQR(gi(I02,i)) - gi(I00,i) * gi(I22,i))
              - gi(I02,i)) / gi(I00,i);
          Real speed2b = (std::sqrt(SQR(gi(I02,i)) - gi(I00,i) * gi(I22,i))
              - gi(I02,i)) / gi(I00,i);

          Real speed2 = std::max(std::abs(speed2a),std::abs(speed2b));

          Real speed3a = -(std::sqrt(SQR(gi(I03,i)) - gi(I00,i) * gi(I33,i))
              - gi(I03,i)) / gi(I00,i);
          Real speed3b = (std::sqrt(SQR(gi(I03,i)) - gi(I00,i) * gi(I33,i))
              - gi(I03,i)) / gi(I00,i);


          Real speed3 = std::max(std::abs(speed3a),std::abs(speed3b));

          // dt1(i) = pmb->pcoord->dx1f(i) /speed1;
          // dt2(i) = pmb->pcoord->dx2f(j) /speed2;
          // dt3(i) = pmb->pcoord->dx3f(k) /speed3;

          dt1(i) /= speed1;
          dt2(i) /=speed2;
          dt3(i) /=speed3;


          // AthenaArray<Real> orbit_quantities;
          // orbit_quantities.NewAthenaArray(Norbit);

          // get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);

          // Real x = pmb->pcoord->x1v(i);
          // Real y = pmb->pcoord->x2v(j);
          // Real z = pmb->pcoord->x3v(k);
          // Real xprime1,yprime1,zprime1,rprime1,Rprime1;
          // get_prime_coords(1,x,y,z, orbit_quantities, &xprime1,&yprime1, &zprime1, &rprime1,&Rprime1);
          // Real xprime2,yprime2,zprime2,rprime2,Rprime2;
          // get_prime_coords(2,x,y,z, orbit_quantities, &xprime2,&yprime2, &zprime2, &rprime2,&Rprime2);

          // Real a1x = orbit_quantities(IA1X);
          // Real a1y = orbit_quantities(IA1Y);
          // Real a1z = orbit_quantities(IA1Z);

          // Real a2x = orbit_quantities(IA2X);
          // Real a2y = orbit_quantities(IA2Y);
          // Real a2z = orbit_quantities(IA2Z);

          // Real a1 = std::sqrt( SQR(a1x) + SQR(a1y) + SQR(a1z) );
          // Real a2 = std::sqrt( SQR(a2x) + SQR(a2y) + SQR(a2z) );


          // Real rh1 =  ( m + std::sqrt( SQR(m) -SQR(a1)) );
          // Real rh2 =  ( m + std::sqrt( SQR(m) -SQR(a2)) );


          // Real uu1 = pmb->phydro->w(IVX,k,j,i);
          // Real uu2 = pmb->phydro->w(IVY,k,j,i);
          // Real uu3 = pmb->phydro->w(IVZ,k,j,i);
          // Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
          //          + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
          //          + g(I33,i)*uu3*uu3;
          // Real gamma = std::sqrt(1.0 + tmp);

          // // Calculate 4-velocity
          // Real alpha = std::sqrt(-1.0/gi(I00,i));
          // Real u0 = gamma/alpha;
          // Real u1 = uu1 - alpha * gamma * gi(I01,i);
          // Real u2 = uu2 - alpha * gamma * gi(I02,i);
          // Real u3 = uu3 - alpha * gamma * gi(I03,i);
          // // if (( rprime1<rh1 and rprime1>0.8*rh1) or ( rprime2<rh2 and rprime2>0.8*rh2))
          // // fprintf(stderr,"dx: %g %g %g \n rprime1: %g rprime 2: %g \n speed1: %g speed2: %g speed3: %g \n gii: %g %g %g \n xyzprime1: %g %g %g \n xyzprime2: %g %g %G \n ginvers: %g %g %g \n v: %g %g %g \n",
          // // pmb->pcoord->dx1f(i),pmb->pcoord->dx2f(j),pmb->pcoord->dx1f(3), rprime1,rprime2, speed1,speed2,speed3,std::sqrt(g(I11,i)),std::sqrt(g(I22,i)),std::sqrt(g(I33,i)),xprime1,yprime1,zprime1,xprime2,yprime2,zprime2,
          // //  gi(I01,i),gi(I00,i),gi(I11,i),u1/u0,u2/u0,u3/u0);


          // orbit_quantities.DeleteAthenaArray();

          // //(cour*dx[mu]/(*ctop)[mu][k][j][i]);
          // dt1(i) = pmb->pcoord->dx1f(i) / cl1;
          // dt2(i) = pmb->pcoord->dx2f(j) / cl2;
          // dt3(i) = pmb->pcoord->dx3f(k) / cl3;
      }

            // compute minimum of (v1 +/- C)
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real& dt_1 = dt1(i);
        min_dt = std::min(min_dt, dt_1);
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real& dt_2 = dt2(i);
          min_dt= std::min(min_dt, dt_2);
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real& dt_3 = dt3(i);
          min_dt = std::min(min_dt, dt_3);
        }
      }
    }
  }

  // calculate the timestep limited by the diffusion processes


  min_dt *= pmb->pmy_mesh->cfl_number;
  // scale the theoretical stability limit by a safety factor = the hyperbolic CFL limit
  // (user-selected or automaticlaly enforced). May add independent parameter "cfl_diff"
  // in the future (with default = cfl_number).

  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();
  dt1.DeleteAthenaArray();
  dt2.DeleteAthenaArray();
  dt3.DeleteAthenaArray();
  return min_dt;
}

