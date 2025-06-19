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
enum b_configs {vertical, normal, renorm, MAD,multi_loop};
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
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real a, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void TransformAphi(Real a3_bl, Real x1,
                     Real x2, Real x3, Real a, Real *pa1, Real *pa2, Real *pa3);
static Real CalculateLFromRPeak(Real r);
static Real CalculateRPeakFromL(Real l_target);
static Real LogHAux(Real r, Real sin_theta);
static void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3);
static void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);
int RefinementCondition(MeshBlock *pmb);
void  Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);
bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv);


void get_prime_coords(Real x, Real y, Real z, AthenaArray<Real> &orbit_quantities,Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ);

void single_bh_metric(Real x1, Real x2, Real x3, ParameterInput *pin,AthenaArray<Real> &g);

void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
  AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt, bool take_derivatives);


void BoostVector(Real t, Real a0, Real a1, Real a2, Real a3,AthenaArray<Real>&orbit_quantities, Real *pa0, Real *pa1, Real *pa2, Real *pa3);

Real DivergenceB(MeshBlock *pmb, int iout);

void set_orbit_arrays(std::string orbit_file_name);

void convert_spherical_to_cartesian_ks(Real r, Real th, Real phi, Real ax, Real ay, Real az,
    Real *x, Real *y, Real *z);

void get_orbit_quantities(Real t, AthenaArray<Real>&orbit_quantities);
void interp_orbits(Real t, int iorbit, AthenaArray<Real> &arr, Real *result);

Real Luminosity(MeshBlock *pmb, int iout);

void NobleCooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);


// Global variables
static Real m, a;                                  // black hole parameters
static Real gamma_adi, k_adi;                      // hydro parameters
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
// static Real rh;                                    // horizon radius
static Real n_pow;


static Real q;          // black hole mass and spin
//Real q; 
// static Real r_inner_boundary,r_inner_boundary_2;
// static Real rh2;
// static Real eccentricity, tau, mean_angular_motion;
static Real t0; //time at which second BH is at polar axis
static Real orbit_inclination;

static Real t0_orbits,dt_orbits;

static Real H_over_r_target;

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


//----------------------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
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


  H_over_r_target = pin->GetOrAddReal("problem", "H_over_r", 0.1);

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

  EnrollUserRadSourceFunction(inner_boundary_source_function);

  AllocateUserHistoryOutput(2);

  EnrollUserHistoryOutput(0, DivergenceB, "divB");
  EnrollUserHistoryOutput(1, Luminosity, "Lum");

  t0 = pin->GetOrAddReal("problem","t0", 1e5);
  m =pin->GetReal("coord", "m");

  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);


  std::string orbit_file_name;
  orbit_file_name =  pin->GetOrAddString("problem","orbit_filename", "orbits.in");
  set_orbit_arrays(orbit_file_name);



  EnrollUserExplicitSourceFunction(NobleCooling);


  // fprintf(stderr,"Done with set_orbit_arrays \n");

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


  int N_user_vars = 1;

  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1+NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1+NGHOST);






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
  Real bh2_focus_radius = 12*q;
  //Real bh2_focus_radius = 3.125*0.1;

  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  // if (current_level >=max_refinement_level) return 0;

  int any_in_refinement_region = 0;
  int any_at_current_level=0;


  int max_level_required = 0;


  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);


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
            get_prime_coords(x,y,z, pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

            Real z_radius;

            // if (n_level==1) z_radius = 250.0*0.9999;
            // if (n_level==2) z_radius = 125.0*0.9999;
            // if (n_level==3) z_radius = 62.5*0.9999;
            // if (n_level==4) z_radius = 31.25*0.9999;
            // if (n_level==5) z_radius = 13.5*0.9999;
            // if (n_level==6) z_radius = 6.3*0.9999;
            // if (n_level==7) z_radius = 3.2*0.9999;
            // if (n_level==8) z_radius = 1.6*0.9999;

            // if (n_level>=5) box_radius = total_box_radius/std::pow(2.,n_level-2)*0.9999;
            // Real z_radius = 0.8* std::pow(2.0,max_smr_refinement_level-n_level+1);


            if (n_level==1) z_radius = 76.8*0.9999;
            if (n_level==2) z_radius = 38.4*0.9999;
            if (n_level==3) z_radius = 19.2*0.9999;
            if (n_level==4) z_radius = 9.6*0.9999;
            if (n_level==5) z_radius = 4.8*0.9999;
            if (n_level==6) z_radius = 2.4*0.9999;
            if (n_level==7) z_radius = 1.2*0.9999;

            if (n_level>=4) box_radius = total_box_radius/std::pow(2.,n_level-2)*0.9999;

            if (n_level==4) box_radius = 75.0 * 0.9999;

          

             // if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
             //   fprintf(stderr,"current level (SMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
             //    }
            if (x<box_radius && x > -box_radius && y<box_radius
              && y > -box_radius && z<z_radius && z > -z_radius ){


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

  Real a = pin->GetOrAddReal("problem", "a", 0.0);
  Real rh =  ( m + std::sqrt( SQR(m) -SQR(a)) );




  //SEE DE VILLIERS+ 2003 https://arxiv.org/pdf/astro-ph/0307260.pdf

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


  Real lin = lc/std::exp(n_pow*std::log(lambda_c/lambda_in) );
  Real c_const = lc/std::pow(lambda_c,n_pow);

  Real q_pow = 2.0-n_pow;
  Real alpha_pow = (2.0*n_pow-2.0)/n_pow; //q_pow/(q_pow-2.0);

  Real ud_t_in = -1.0/std::sqrt( - (gitt(rin,a,PI/2.0) - 2.0*lin*gitphi(rin,a,PI/2.0) + SQR(lin)*giphiphi(rin,a,PI/2.0) ) );



  // Compute Peak Density //
  Real denom_sq = -( gitt(rc,a,PI/2.0) - 2.0*lc*gitphi(rc,a,PI/2.0) + SQR(lc)*giphiphi(rc,a,PI/2.0) );
  Real ud_t_c = -1.0/std::sqrt(denom_sq);
  Real eps_c = 1.0/gam * (ud_t_in * f(lin,c_const,n_pow)/(ud_t_c * f(lc,c_const,n_pow)) -1.0);
  rho_peak = std::pow( (eps_c * (gam-1.0)/k_adi), (1.0/(gam-1.0)) );
  pgas_over_rho_peak = eps_c * (gam-1.0);
  Real pgas_peak = pgas_over_rho_peak * rho_peak;


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
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,a, &r,
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



    // Determine limits of sample grid
    Real r_vals[8], theta_vals[8], phi_vals[8];
    for (int p = 0; p < 8; ++p) {
      Real x1_val = (p%2 == 0) ? x1_min : x1_max;
      Real x2_val = ((p/2)%2 == 0) ? x2_min : x2_max;
      Real x3_val = ((p/4)%2 == 0) ? x3_min : x3_max;
      GetBoyerLindquistCoordinates(x1_val, x2_val, x3_val,0,0,a, r_vals+p, theta_vals+p,
          phi_vals+p);
    }
    // r_min = *std::min_element(r_vals, r_vals+8);
    r_max = *std::max_element(r_vals, r_vals+8);
    // theta_min = *std::min_element(theta_vals, theta_vals+8);
    // theta_max = *std::max_element(theta_vals, theta_vals+8);
    // phi_min = *std::min_element(phi_vals, phi_vals+8);
    // phi_max = *std::max_element(phi_vals, phi_vals+8);

    r_min = rh;
    theta_min = 0.01; 
    theta_max = PI-0.01;
    phi_min = 0.0;
    phi_max = 2.0*PI;

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
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),0,0,a,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));

                Real press = phydro->w(IPR,k,j,i);
                Real press_cutoff = std::max(press-potential_cutoff*pgas_peak, static_cast<Real>(0.0));

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
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,a,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));

                Real press = phydro->w(IPR,k,j,i);
                Real press_cutoff = std::max(press-potential_cutoff*pgas_peak, static_cast<Real>(0.0));

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
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),0,0,a,
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
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,a,
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
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),0,0,a,
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
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,a,
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
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0,0,a, &r,
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
  int SCALE_DIVERGENCE = true; 
  //int SCALE_DIVERGENCE = pin->GetOrAddBoolean("problem","scale_divergence",false);


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


void set_orbit_arrays(std::string orbit_file_name){
      FILE *input_file;
        if ((input_file = fopen(orbit_file_name.c_str(), "r")) == NULL)   
               fprintf(stderr, "Cannot open %s, %s\n", "input_file",orbit_file_name.c_str());



    fscanf(input_file, "%i %lf \n", &nt, &q);
    // int nt = 10;
    fprintf(stderr,"nt in set_orbit_arrays: %d \n q in set_orbit_arrays: %g \n", nt,q);
    // q = 0.01;

       
    // fprintf(stderr,"nt in set_orbit_arrays: %d \n q in set_orbit_arrays: %g \n", nt,q);

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

  fprintf(stderr,"Done reading orbit file \n");

}

void get_orbit_quantities(Real t, AthenaArray<Real>&orbit_quantities){

  for (int iorbit=0; iorbit<Norbit; iorbit++){
    interp_orbits(t,iorbit,orbit_array,&orbit_quantities(iorbit));
  }

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
          Real t = pmb->pmy_mesh->metric_time;

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


  apply_inner_boundary_condition(pmb,prim,prim_scalar);

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
  Real r_isco = risco_calc_general( 1, spin, mass );
  r = r_isco;
  rm1 = 1./r;     x = sqrt(r);    xm3 = 1./(x*x*x);    term1 = a*a*rm1;

  C_f = 1. - 3*rm1 + 2*a*xm3;
  F_f = 1. - 2*a*xm3 + term1*rm1;
  G_f = 1. - 2*rm1 + a*xm3 ;

  ut_sq_isco   = G_f * G_f     / C_f ;
  uphi_sq_isco = F_f * F_f * r / C_f;
  local_first_time = 0;

  r = r_loc;


  rm1 = 1./r;     x = sqrt(r);    xm3 = 1./(x*x*x);    term1 = a*a*rm1;

  if( r > r_isco ) { 
    C_f = 1. - 3*rm1 + 2*a*xm3;
    F_f = 1. - 2*a*xm3 + term1*rm1;
    G_f = 1. - 2*rm1 + a*xm3 ;
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

        Real r_isco = risco_calc_general( 1, a1, m );
        Real r_isco_secondary = risco_calc_general( 1, a2/(q+SMALL), q )/(q+SMALL); //neads a/M, returns isco in units of M

        if (radius<r_isco) Omega = 1.0/( std::pow(r_isco,1.5) + a1);
        if (rprime<r_isco_secondary) Omega_secondary = 1.0/(q+SMALL) * 1.0/( std::pow(r_isco_secondary/(q+SMALL),1.5) + a2/(q+SMALL));

        Real Target_Temperature = target_temperature_func( radius,H_over_r_target,a1,m);
        Real Target_Temperature_secondary = target_temperature_func( rprime/(q+SMALL),H_over_r_target,a2/(q+SMALL),q);

        Real Y = prim(IPR,k,j,i)/prim(IDN,k,j,i)/Target_Temperature;
        Real Y_secondary = prim(IPR,k,j,i)/prim(IDN,k,j,i)/Target_Temperature_secondary;



        Real L_cool = Omega * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        Real L_cool_secondary = Omega_secondary * ug * std::sqrt( Y_secondary-1.0 +  std::fabs(Y_secondary-1.0) );
        if (L_cool<0) L_cool = 0.0;
        if (L_cool_secondary<0) L_cool_secondary = 0.0;


        L_cool = std::max(L_cool,L_cool_secondary);


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


        cons(IEN,k,j,i) += -dt * L_cool * u_0; 
        cons(IM1,k,j,i) += -dt * L_cool * u_1;
        cons(IM2,k,j,i) += -dt * L_cool * u_2;
        cons(IM3,k,j,i) += -dt * L_cool * u_3;

        Real ug_frac = dt * L_cool/ug;

        pmb->user_out_var(0,k,j,i) = L_cool;



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

Real Luminosity(MeshBlock *pmb, int iout)
{
  Real Lum=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {

        Real volume = pmb->pcoord->GetCellVolume(k,j,i);

        Lum += pmb->user_out_var(0,k,j,i) * volume;
      }
    }
  }



  return Lum;
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


  // // Go through all cells
  // for (int k = ks; k <= ke; ++k) {
  //   for (int j = js; j <= je; ++j) {
  //     pcoord->CellMetric(k, j, is, ie, g, gi);
  //     for (int i = is; i <= ie; ++i) {

  //       // Calculate normal frame Lorentz factor
  //       Real uu1 = phydro->w(IM1,k,j,i);
  //       Real uu2 = phydro->w(IM2,k,j,i);
  //       Real uu3 = phydro->w(IM3,k,j,i);
  //       Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
  //                + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
  //                + g(I33,i)*uu3*uu3;
  //       Real gamma = std::sqrt(1.0 + tmp);
  //       user_out_var(0,k,j,i) = gamma;

  //       // Calculate 4-velocity
  //       Real alpha = std::sqrt(-1.0/gi(I00,i));
  //       Real u0 = gamma/alpha;
  //       Real u1 = uu1 - alpha * gamma * gi(I01,i);
  //       Real u2 = uu2 - alpha * gamma * gi(I02,i);
  //       Real u3 = uu3 - alpha * gamma * gi(I03,i);
  //       Real u_0, u_1, u_2, u_3;

  //       user_out_var(1,k,j,i) = u0;
  //       user_out_var(2,k,j,i) = u1;
  //       user_out_var(3,k,j,i) = u2;
  //       user_out_var(4,k,j,i) = u3;
  //       if (not MAGNETIC_FIELDS_ENABLED) {
  //         continue;
  //       }

  //       pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

  //       // Calculate 4-magnetic field
  //       Real bb1 = pfield->bcc(IB1,k,j,i);
  //       Real bb2 = pfield->bcc(IB2,k,j,i);
  //       Real bb3 = pfield->bcc(IB3,k,j,i);
  //       Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
  //               + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
  //               + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
  //               + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
  //       Real b1 = (bb1 + b0 * u1) / u0;
  //       Real b2 = (bb2 + b0 * u2) / u0;
  //       Real b3 = (bb3 + b0 * u3) / u0;
  //       Real b_0, b_1, b_2, b_3;
  //       pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

  //       // Calculate magnetic pressure
  //       Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
  //       user_out_var(5,k,j,i) = b_sq/2.0;

  //       if (std::isnan(b_sq)) {
  //         Real r, th,tmp;
  //         GetBoyerLindquistCoordinates(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),&r,&th,&tmp);
  //         fprintf(stderr,"BSQ IS NAN!! \n x y z: %g %g %g r th  %g %g \n g: %g %g %g %g %g %g %g %g %g %g\n bb: %g %g %g u: %g %g %g %g \n",
  //           pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),r,th,g(I00,i),g(I01,i),g(I02,i),g(I03,i),
  //           g(I11,i),g(I12,i),g(I13,i),g(I22,i),g(I23,i),g(I33,i),bb1,bb2,bb3,u0,u1,u2,u3) ;

  //         exit(0);
  //       }
  //     }
  //   }
  // }

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
// Function for calculating angular momentum variable l
// Inputs:
//   r: desired radius of pressure maximum
// Outputs:
//   returned value: l = u^t u_\phi such that pressure maximum occurs at r_peak
// Notes:
//   beware many different definitions of l abound
//     this is *not* -u_phi/u_t
//   Harm has a similar function: lfish_calc() in init.c
//     Harm's function assumes M = 1 and that corotation is desired
//     it is equivalent to this, though seeing this requires much manipulation
//   implements (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation
//   see CalculateRPeakFromL()

static Real CalculateLFromRPeak(Real r) {
  Real num = SQR(SQR(r)) + SQR(a*r) - 2.0*m*SQR(a)*r - a*(SQR(r)-SQR(a))*std::sqrt(m*r);
  Real denom = SQR(r) - 3.0*m*r + 2.0*a*std::sqrt(m*r);
  return 1.0/r * std::sqrt(m/r) * num/denom;
}

//----------------------------------------------------------------------------------------
// Function for calculating pressure maximum radius r_peak
// Inputs:
//   l_target: desired u^t u_\phi
// Outputs:
//   returned value: location of pressure maximum given l_target
// Notes:
//   beware many different definitions of l abound
//     this is *not* -u_phi/u_t
//   uses (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation
//   uses bisection to find r such that formula for l agrees with given value
//   proceeds until either absolute tolerance is met
//   returns best value after max_iterations reached if tolerances not met
//   returns NAN in case of failure (e.g. root not bracketed)
//   see CalculateLFromRPeak()

static Real CalculateRPeakFromL(Real l_target) {
  // Parameters
  const Real tol_r = 1.0e-10;      // absolute tolerance on abscissa r_peak
  const Real tol_l = 1.0e-10;      // absolute tolerance on ordinate l
  const int max_iterations = 100;  // maximum number of iterations before best res

  // Prepare initial values
  Real r_a = r_min;
  Real r_b = r_max;
  Real r_c = 0.5 * (r_min + r_max);
  Real l_a = CalculateLFromRPeak(r_a);
  Real l_b = CalculateLFromRPeak(r_b);
  Real l_c = CalculateLFromRPeak(r_c);
  if (not ((l_a < l_target and l_b > l_target) or (l_a > l_target and l_b < l_target))) {
    return NAN;
  }

  // Find root
  for (int n = 0; n < max_iterations; ++n) {
    if (std::abs(r_b-r_a) <= 2.0*tol_r or std::abs(l_c-l_target) <= tol_l) {
      break;
    }
    if ((l_a < l_target and l_c < l_target) or (l_a > l_target and l_c > l_target)) {
      r_a = r_c;
      l_a = l_c;
    } else {
      r_b = r_c;
      l_b = l_c;
    }
    r_c = 0.5 * (r_min + r_max);
    l_c = CalculateLFromRPeak(r_c);
  }
  return r_c;
}

//----------------------------------------------------------------------------------------
// Function for helping to calculate enthalpy
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)

static Real LogHAux(Real r, Real sin_theta) {
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*m*r + SQR(a);                    // \Delta
  Real sigma = SQR(r) + SQR(a)*cos_sq_theta;                 // \Sigma
  Real aa = SQR(SQR(r)+SQR(a)) - delta*SQR(a)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // \exp(-2\chi) (cf. FM 2.15)
  Real omega = 2.0*m*a*r/aa;                                 // \omega (FM 3.5)
  Real var_a = std::sqrt(1.0 + 4.0*SQR(l)*exp_neg2chi);
  Real var_b = 0.5 * std::log((1.0+var_a)
      / (sigma*delta/aa));
  Real var_c = -0.5 * var_a;
  Real var_d = -l * omega;
  return var_b + var_c + var_d;                              // (FM 3.4)
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside untilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   sin_theta: sine of Boyer-Lindquist theta
// Outputs:
//   pu0: u^t set (Boyer-Lindquist coordinates)
//   pu3: u^\phi set (Boyer-Lindquist coordinates)
// Notes:
//   The formula for u^3 as a function of u_{(\phi)} is tedious to derive, but this
//       matches the formula used in Harm (init.c).

static void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3) {
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*m*r + SQR(a);                    // \Delta
  Real sigma = SQR(r) + SQR(a)*cos_sq_theta;                 // \Sigma
  Real aa = SQR(SQR(r)+SQR(a)) - delta*SQR(a)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // \exp(-2\chi) (cf. FM 2.15)
  Real u_phi_proj_a = 1.0 + 4.0*SQR(l)*exp_neg2chi;
  Real u_phi_proj_b = -1.0 + std::sqrt(u_phi_proj_a);
  Real u_phi_proj = std::sqrt(0.5 * u_phi_proj_b);           // (FM 3.3)
  Real u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
  Real u3_b = 2.0*m*a*r * std::sqrt(u3_a);
  Real u3_c = std::sqrt(sigma/aa) / sin_theta;
  Real u3 = u3_b + u3_c * u_phi_proj;
  Real g_00 = -(1.0 - 2.0*m*r/sigma);
  Real g_03 = -2.0*m*a*r/sigma * sin_sq_theta;
  Real g_33 = (sigma + (1.0 + 2.0*m*r/sigma) * SQR(a)
      * sin_sq_theta) * sin_sq_theta;
  Real u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
  Real u0_b = std::sqrt(u0_a - g_00);
  Real u0 = -1.0/g_00 * (g_03*u3 + u0_b);
  *pu0 = u0;
  *pu3 = u3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside tilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   theta,phi: Boyer-Lindquist theta and phi in BH-aligned coordinates
// Outputs:
//   pu0,pu1,pu2,pu3: u^\mu set (Boyer-Lindquist coordinates)
// Notes:
//   first finds corresponding location in untilted torus
//   next calculates velocity at that point in untilted case
//   finally transforms that velocity into coordinates in which torus is tilted

static void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3) {
  // Calculate corresponding location
  Real sin_theta = std::sin(theta);
  Real cos_theta = std::cos(theta);
  Real sin_phi = std::sin(phi);
  Real cos_phi = std::cos(phi);
  Real sin_vartheta, cos_vartheta, varphi;
  if (psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = cos_psi * x - sin_psi * z;
    Real vary = y;
    Real varz = sin_psi * x + cos_psi * z;
    sin_vartheta = std::sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = std::atan2(vary, varx);
  } else {
    sin_vartheta = std::abs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? phi-PI : phi;
  }
  Real sin_varphi = std::sin(varphi);
  Real cos_varphi = std::cos(varphi);

  // Calculate untilted velocity
  Real u0_tilt, u3_tilt;
  CalculateVelocityInTorus(r, sin_vartheta, &u0_tilt, &u3_tilt);
  Real u1_tilt = 0.0;
  Real u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (psi != 0.0) {
    Real dtheta_dvartheta =
        (cos_psi * sin_vartheta + sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    Real dtheta_dvarphi = -sin_psi * sin_vartheta * sin_varphi / sin_theta;
    Real dphi_dvartheta = sin_psi * sin_varphi / SQR(sin_theta);
    Real dphi_dvarphi = sin_vartheta / SQR(sin_theta)
        * (cos_psi * sin_vartheta + sin_psi * cos_vartheta * cos_varphi);
    *pu2 = dtheta_dvartheta * u2_tilt + dtheta_dvarphi * u3_tilt;
    *pu3 = dphi_dvartheta * u2_tilt + dphi_dvarphi * u3_tilt;
  } else {
    *pu2 = u2_tilt;
    *pu3 = u3_tilt;
  }
  if (sin_theta < 0.0) {
    *pu2 *= -1.0;
    *pu3 *= -1.0;
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


  // if (std::isnan(*rprime) or std::isnan(*xprime) or std::isnan(*yprime) or std::isnan(*zprime) ){
  //     fprintf(stderr,"ISNAN in GetBoyer!!! \n xyz: %g %g %g \n xbh ybh zbh: %g %g %g \n ax ay az a: %g %g %g %g \n adotx: %g \n xyzprime: %g %g %g \n ",
  //       x,y,z,xbh, ybh, zbh, ax,ay,az,a_mag, a_dot_x_prime,*xprime,*yprime,*zprime );
  //     exit(0);
  //   }

  return;

}

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



void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{


  m = pin->GetReal("coord", "m");

  //////////////Perturber Black Hole//////////////////

  t0 = pin->GetOrAddReal("problem","t0", 1e5);

  Binary_BH_Metric(t,x1,x2,x3,g,g_inv,dg_dx1,dg_dx2,dg_dx3,dg_dt,true);

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


  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x = 0.1;
    y = 0.1;
    z = 0.1;
  }

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

  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a_dot_x));
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



#define DEL 1e-7
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
#define DEL 1e-7
void single_bh_metric(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a = pin->GetReal("coord", "a");
  Real a_spin = a;

  if ((std::fabs(z)<SMALL) && ( z>=0 )) z=  SMALL;
  if ((std::fabs(z)<SMALL) && ( z<0  )) z= -SMALL;

  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x = 0.1;
    y = 0.1;
    z = 0.1;
  }

  Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = std::sqrt(r/2.0);


  //if (r<0.01) r = 0.01;


  Real eta[4],l_lower[4],l_upper[4];

  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  l_upper[0] = -1.0;
  l_upper[1] = (r*x + a_spin*y)/( SQR(r) + SQR(a) );
  l_upper[2] = (r*y - a_spin*x)/( SQR(r) + SQR(a) );
  l_upper[3] = z/r;

  l_lower[0] = 1.0;
  l_lower[1] = l_upper[1];
  l_lower[2] = l_upper[2];
  l_lower[3] = l_upper[3];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;




  // Set covariant components
  g(I00) = eta[0] + f * l_lower[0]*l_lower[0] ;
  g(I01) =          f * l_lower[0]*l_lower[1] ;
  g(I02) =          f * l_lower[0]*l_lower[2] ;
  g(I03) =          f * l_lower[0]*l_lower[3] ;
  g(I11) = eta[1] + f * l_lower[1]*l_lower[1] ;
  g(I12) =          f * l_lower[1]*l_lower[2] ;
  g(I13) =          f * l_lower[1]*l_lower[3] ;
  g(I22) = eta[2] + f * l_lower[2]*l_lower[2] ;
  g(I23) =          f * l_lower[2]*l_lower[3] ;
  g(I33) = eta[3] + f * l_lower[3]*l_lower[3] ;



  return;
}


// Real EquationOfState::GetRadius(Real x1, Real x2, Real x3,  Real a){

//   Real r, th, phi;
//   GetBoyerLindquistCoordinates(x1,x2,x3,0,0,a, &r, &th, &phi);
//   return r;
// }

// Real EquationOfState::GetRadius2(Real x1, Real x2, Real x3){

//   return -1.0;
//   // Real xprime,yprime,zprime,rprime,Rprime;
//   // get_prime_coords(x1,x2,x3, pmy_block_->pmy_mesh->time, &xprime,&yprime,&zprime,&rprime, &Rprime);

//   // return rprime;
// }
