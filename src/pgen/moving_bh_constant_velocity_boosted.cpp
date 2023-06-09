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

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void TransformAphi(Real a3_bl, Real x1,
                     Real x2, Real x3, Real *pa1, Real *pa2, Real *pa3);
static Real CalculateLFromRPeak(Real r);
static Real CalculateRPeakFromL(Real l_target);
static Real LogHAux(Real r, Real sin_theta);
static void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3);
static void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);
static Real CalculateBetaMin();
static bool CalculateBeta(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
                          Real theta_p, Real phi_m, Real phi_c, Real phi_p, Real *pbeta);
static bool CalculateBetaFromA(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
              Real theta_p, Real a_cm, Real a_cp, Real a_mc, Real a_pc, Real *pbeta);
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


void get_prime_coords(Real x, Real y, Real z, Real t, Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);
void get_bh_position(Real t, Real *xbh, Real *ybh, Real *zbh);
void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ);

void single_bh_metric(Real x1, Real x2, Real x3, ParameterInput *pin,AthenaArray<Real> &g);

Real DivergenceB(MeshBlock *pmb, int iout);
void UpdateMetricFunction(Real metric_t, MeshBlock *pmb);


// Global variables
static Real m, a;                                  // black hole parameters
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
static Real rh;                                    // horizon radius

static Real aprime,q;          // black hole mass and spin
static Real r_inner_boundary,r_inner_boundary_2;
static Real rh2;
static Real v_bh2;
static Real Omega_bh2;
static Real eccentricity, tau, mean_angular_motion;
static Real t0; //time at which second BH is at polar axis

// Real rotation_matrix[3][3];


int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
int max_second_bh_refinement_level = 0;  /*Maximum allowed level of refinement for AMR on secondary BH */
int max_smr_refinement_level = 0; /*Maximum allowed level of refinement for SMR on primary BH */

static Real SMALL = 1e-5;


/* A structure defining the properties of each of the source 'stars' */
typedef struct secondary_bh_s{
  Real q;     /* mass ratio */
  Real spin;    /* dimensionless spin in units of ???s*/
  Real x1;      /* position in X,Y,Z (in pc) */
  Real x2;
  Real x3;
  Real v1;      /* velocity in X,Y,Z */
  Real v2;
  Real v3;
  Real alpha;     /* euler angles for ZXZ rotation*/
  Real beta;
  Real gamma;
  Real tau;
  Real mean_angular_motion;
  Real eccentricity;
  Real rotation_matrix[3][3];
  Real period;
}secondary_bh;

secondary_bh bh2;          /* The stars structure used throughout */

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
  a = pcoord->GetSpin();
  q = pin->GetOrAddReal("problem", "q", 0.1);
  aprime = q * pin->GetOrAddReal("problem", "a_bh2", 0.0);

  v_bh2 = pin->GetOrAddReal("problem", "vbh", 0.05);



  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  r_inner_boundary = rh/2.0;


    // Get mass of black hole
  Real m2 = q;

  rh2 =  ( m2 + std::sqrt( SQR(m2) - SQR(aprime)) );
  r_inner_boundary_2 = rh2/2.0;

  int N_user_vars = 7;
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
  Real bh2_focus_radius = 3.125*q;
  //Real bh2_focus_radius = 3.125*0.08;

  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  // if (current_level >=max_refinement_level) return 0;

  int any_in_refinement_region = 0;
  int any_at_current_level=0;


  int max_level_required = 0;


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
            get_prime_coords(x,y,z, pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
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


  // Prepare scratch arrays
  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {

        Real rho = 1.0;
        Real pgas = 1e-3;
        Real uu1 = 0.0;
        Real uu2 = 0.0;
        Real uu3 = 0.0;


        Real xprime,yprime,zprime,rprime,Rprime;

        get_prime_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);

        if (rprime<=5.0){
          rho = 0.0;
          pgas = 0.0;
        }

          Real beta_init = 5.0;
          Real B_const = 0.0;

        if (MAGNETIC_FIELDS_ENABLED)
            Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                     + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                     + g(I33,i)*uu3*uu3;
            Real gamma = std::sqrt(1.0 + tmp);
            // user_out_var(0,k,j,i) = gamma;

            // Calculate 4-velocity
            Real alpha = std::sqrt(-1.0/gi(I00,i));
            Real u0 = gamma/alpha;
            Real u1 = uu1 - alpha * gamma * gi(I01,i);
            Real u2 = uu2 - alpha * gamma * gi(I02,i);
            Real u3 = uu3 - alpha * gamma * gi(I03,i);
            Real u_0, u_1, u_2, u_3;

            pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

            // Calculate 4-magnetic field
            Real bb1 = 0.0;
            Real bb2 = 0.1;
            Real bb3 = 0.0;
            Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
                    + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
                    + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
                    + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
            Real b1 = (bb1 + b0 * u1) / u0;
            Real b2 = (bb2 + b0 * u2) / u0;
            Real b3 = (bb3 + b0 * u3) / u0;
            Real b_0, b_1, b_2, b_3;
            pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

            // Calculate magnetic pressure
            Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

            Real beta_act = pgas / b_sq * 2.0;

            B_const = std::sqrt(beta_act/beta_init*b_sq);
       }


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

            get_prime_coords(x_coord,pcoord->x2f(j),pcoord->x3f(k), pmy_mesh->t, &xprime,&yprime, &zprime, &rprime,&Rprime);

            Real Ax = Ax_func(xprime,yprime,zprime);
            Real Ay = Ay_func(xprime,yprime,zprime);
            Real Az = Az_func(xprime,yprime,zprime);

            // Real Ar,Ath,Aphi,A0;;


            A1(k,j,i) = Ax * exp_cut_off(rprime);

            Real y_coord;
            if (j<= ju) y_coord = pcoord->x2v(j);
            else y_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);

            get_prime_coords(pcoord->x1f(i),y_coord,pcoord->x3f(k), pmy_mesh->t, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Ax = Ax_func(xprime,yprime,zprime) ;
            Ay = Ay_func(xprime,yprime,zprime) ;
            Az = Az_func(xprime,yprime,zprime) ;

            A2(k,j,i) = Ay * exp_cut_off(rprime);

            Real z_coord;
            if (k<= ku) z_coord = pcoord->x3v(k);
            else z_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
            get_prime_coords(pcoord->x1f(i),pcoord->x2f(j),z_coord, pmy_mesh->t, &xprime,&yprime, &zprime, &rprime,&Rprime);
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
  return;
}

void  MeshBlock::PreserveDivbNewMetric(ParameterInput *pin){
  int SCALE_DIVERGENCE = false; //true; //pin->GetOrAddBoolean("problem","scale_divergence",false);

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



/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar){


  Real r,th,ph;
  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];



   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {


         GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);

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

          get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime < r_inner_boundary_2){
              

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



}}}




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


  // Go through all cells
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pcoord->CellMetric(k, j, is, ie, g, gi);
      for (int i = is; i <= ie; ++i) {

        // Calculate normal frame Lorentz factor
        Real uu1 = phydro->w(IM1,k,j,i);
        Real uu2 = phydro->w(IM2,k,j,i);
        Real uu3 = phydro->w(IM3,k,j,i);
        Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                 + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                 + g(I33,i)*uu3*uu3;
        Real gamma = std::sqrt(1.0 + tmp);
        // user_out_var(0,k,j,i) = gamma;

        // Calculate 4-velocity
        Real alpha = std::sqrt(-1.0/gi(I00,i));
        Real u0 = gamma/alpha;
        Real u1 = uu1 - alpha * gamma * gi(I01,i);
        Real u2 = uu2 - alpha * gamma * gi(I02,i);
        Real u3 = uu3 - alpha * gamma * gi(I03,i);
        Real u_0, u_1, u_2, u_3;

        // user_out_var(1,k,j,i) = u0;
        // user_out_var(2,k,j,i) = u1;
        // user_out_var(3,k,j,i) = u2;
        // user_out_var(4,k,j,i) = u3;
        if (not MAGNETIC_FIELDS_ENABLED) {
          continue;
        }

        pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

        // Calculate 4-magnetic field
        Real bb1 = pfield->bcc(IB1,k,j,i);
        Real bb2 = pfield->bcc(IB2,k,j,i);
        Real bb3 = pfield->bcc(IB3,k,j,i);
        Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
                + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
                + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
                + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
        Real b1 = (bb1 + b0 * u1) / u0;
        Real b2 = (bb2 + b0 * u2) / u0;
        Real b3 = (bb3 + b0 * u3) / u0;
        Real b_0, b_1, b_2, b_3;
        pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

        // Calculate magnetic pressure
        Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
        user_out_var(5,k,j,i) = b_sq/2.0;

        if (std::isnan(b_sq)) {
          Real r, th,tmp;
          GetBoyerLindquistCoordinates(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),&r,&th,&tmp);
          fprintf(stderr,"BSQ IS NAN!! \n x y z: %g %g %g r th  %g %g \n g: %g %g %g %g %g %g %g %g %g %g\n bb: %g %g %g u: %g %g %g %g \n",
            pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),r,th,g(I00,i),g(I01,i),g(I02,i),g(I03,i),
            g(I11,i),g(I12,i),g(I13,i),g(I22,i),g(I23,i),g(I33,i),bb1,bb2,bb3,u0,u1,u2,u3) ;

          exit(0);
        }
      }
    }
  }

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
// Function for finding approximate minimum value of plasma beta expected
// Inputs: (none)
// Outputs:
//   returned value: minimum beta found by sampling grid covering whole domain
// Notes:
//   constructs grid over entire mesh, not just block
//   grid is not necessarily the same as used for the problem proper
//   calculation is done entirely in Boyer-Lindquist coordinates

static Real CalculateBetaMin() {
  // Prepare container to hold minimum
  Real beta_min_actual = std::numeric_limits<Real>::max();

  // Go through sample grid in phi
  for (int k = 0; k < sample_n_phi; ++k) {

    // Calculate phi values
    Real phi_m = phi_min + static_cast<Real>(k)/static_cast<Real>(sample_n_phi)
        * (phi_max-phi_min);
    Real phi_p = phi_min + static_cast<Real>(k+1)/static_cast<Real>(sample_n_phi)
        * (phi_max-phi_min);
    Real phi_c = 0.5 * (phi_m + phi_p);

    // Go through sample grid in theta
    for (int j = 0; j < sample_n_theta; ++j) {

      // Calculate theta values
      Real theta_m = theta_min + static_cast<Real>(j)/static_cast<Real>(sample_n_theta)
          * (theta_max-theta_min);
      Real theta_p = theta_min + static_cast<Real>(j+1)/static_cast<Real>(sample_n_theta)
          * (theta_max-theta_min);
      Real theta_c = 0.5 * (theta_m + theta_p);

      // Go through sample grid in r
      Real r_m, delta_r;
      Real r_p = 0.0;
      for (int i = 0; i < sample_n_r; ++i) {

        // Calculate r values
        if (i == 0) {
          r_m = r_min;
          Real ratio_power = 1.0;
          Real ratio_sum = 1.0;
          for (int ii = 1; ii < sample_n_r; ++ii) {
            ratio_power *= sample_r_rat;
            ratio_sum += ratio_power;
          }
          delta_r = (r_max-r_min) / ratio_sum;
        } else {
          r_m = r_p;
          delta_r *= sample_r_rat;
        }
        r_p = r_m + delta_r;
        Real r_c = 0.5 * (r_m + r_p);

        // Calculate beta
        Real beta;
        //fprintf(stderr,"r: %g %g %g\n theta: %g %g %g \n phi: %g %g %g \n r_min: %g r_max: %g \n theta_min: %g theta_max: %g \n phi_min : %g phi_max: %g \n",
          //r_m,r_c,r_p,theta_m,theta_c,theta_p,phi_m,phi_c,phi_p,r_min,r_max,theta_min,theta_max,phi_min,phi_max);
        
        bool value_set = CalculateBeta(r_m, r_c, r_p, theta_m, theta_c, theta_p, phi_m,
            phi_c, phi_p, &beta);
        if (value_set) {
          beta_min_actual = std::min(beta_min_actual, beta);
        }
      }
    }
  }
  return beta_min_actual;
}

//----------------------------------------------------------------------------------------
// Function for calculating beta from four nearby points
// Inputs:
//   r_m,r_c,r_p: inner, center, and outer radii
//   theta_m,theta_c,theta_p: upper, center, and lower polar angles
// Outputs:
//   pbeta: value set to plasma beta at cell center
//   returned value: true if pbeta points to meaningful number (inside torus)
// Notes:
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)

static bool CalculateBeta(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
                          Real theta_p, Real phi_m, Real phi_c, Real phi_p, Real *pbeta) {
  // Assemble arrays of points
  Real r_vals[7], theta_vals[7], phi_vals[7];
  r_vals[0] = r_c; theta_vals[0] = theta_c; phi_vals[0] = phi_c;
  r_vals[1] = r_m; theta_vals[1] = theta_c; phi_vals[1] = phi_c;
  r_vals[2] = r_p; theta_vals[2] = theta_c; phi_vals[2] = phi_c;
  r_vals[3] = r_c; theta_vals[3] = theta_m; phi_vals[3] = phi_c;
  r_vals[4] = r_c; theta_vals[4] = theta_p; phi_vals[4] = phi_c;
  r_vals[5] = r_c; theta_vals[5] = theta_c; phi_vals[5] = phi_m;
  r_vals[6] = r_c; theta_vals[6] = theta_c; phi_vals[6] = phi_p;

  // Account for tilt
  Real sin_theta_vals[7], cos_theta_vals[7];
  Real sin_phi_vals[7], cos_phi_vals[7];
  Real sin_vartheta_vals[7], cos_vartheta_vals[7];
  Real sin_varphi_vals[7], cos_varphi_vals[7];
  for (int p = 0; p < 7; ++p) {
    sin_theta_vals[p] = std::sin(theta_vals[p]);
    cos_theta_vals[p] = std::cos(theta_vals[p]);
    sin_phi_vals[p] = std::sin(phi_vals[p]);
    cos_phi_vals[p] = std::cos(phi_vals[p]);
    Real varphi;
    if (psi != 0.0) {
      Real x = sin_theta_vals[p] * cos_phi_vals[p];
      Real y = sin_theta_vals[p] * sin_phi_vals[p];
      Real z = cos_theta_vals[p];
      Real varx = cos_psi * x - sin_psi * z;
      Real vary = y;
      Real varz = sin_psi * x + cos_psi * z;
      sin_vartheta_vals[p] = std::sqrt(SQR(varx) + SQR(vary));
      if (field_config == vertical) {
        break;
      }
      cos_vartheta_vals[p] = varz;
      varphi = std::atan2(vary, varx);
    } else {
      sin_vartheta_vals[p] = std::abs(sin_theta_vals[p]);
      if (field_config == vertical) {
        break;
      }
      cos_vartheta_vals[p] = cos_theta_vals[p];
      varphi = (sin_theta_vals[p] < 0.0) ? phi_vals[p]-PI : phi_vals[p];
    }
    sin_varphi_vals[p] = std::sin(varphi);
    cos_varphi_vals[p] = std::cos(varphi);
  }

  // Determine if we are in the torus (FM 3.6)
  if (r_m < r_edge) {
    return false;
  }
  Real log_h_vals[7];
  for (int p = 0; p < 7; ++p) {
    log_h_vals[p] = LogHAux(r_vals[p], sin_vartheta_vals[p]) - log_h_edge;
    if (log_h_vals[p] < 0.0) {
      return false;
    }
    if (field_config == vertical) {
      break;
    }
  }

  // Calculate vector potential values in torus coordinates
  Real a_varphi_vals[7];
  Real pgas;
  for (int p = 0; p < 7; ++p) {
    Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h_vals[p])-1.0);
    Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
    if (p == 0 and rho < sample_cutoff) {
      return false;
    }
    if (p == 0) {
      pgas = pgas_over_rho * rho;
    }
    if (field_config == vertical) {
      break;
    }
    Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
    a_varphi_vals[p] =
        std::pow(r_vals[p], potential_r_pow) * std::pow(rho_cutoff, potential_rho_pow);
    if (a_varphi_vals[p] == 0.0) {
      return false;
    }
  }

  // Account for tilt
  Real a_theta_vals[7], a_phi_vals[7];
  if (field_config != vertical) {
    for (int p = 0; p < 7; ++p) {
      if (psi != 0.0) {
        Real dvarphi_dtheta = -sin_psi * sin_phi_vals[p] / SQR(sin_vartheta_vals[p]);
        Real dvarphi_dphi = sin_theta_vals[p] / SQR(sin_vartheta_vals[p])
            * (cos_psi * sin_theta_vals[p]
            - sin_psi * cos_theta_vals[p] * cos_phi_vals[p]);
        a_theta_vals[p] = dvarphi_dtheta * a_varphi_vals[p];
        a_phi_vals[p] = dvarphi_dphi * a_varphi_vals[p];
      } else {
        a_theta_vals[p] = 0.0;
        a_phi_vals[p] = a_varphi_vals[p];
      }
    }
  }

  // Calculate cell-centered 3-magnetic field
  Real det = (SQR(r_c) + SQR(a) * SQR(cos_theta_vals[0])) * std::abs(sin_theta_vals[0]);
  Real bb1, bb2, bb3;
  if (field_config != vertical) {
    bb1 = 1.0/det * ((a_phi_vals[4]-a_phi_vals[3]) / (theta_p-theta_m)
        - (a_theta_vals[6]-a_theta_vals[5]) / (phi_p-phi_m));
    bb2 = -1.0/det * (a_phi_vals[2]-a_phi_vals[1]) / (r_p-r_m);
    bb3 = 1.0/det * (a_theta_vals[2]-a_theta_vals[1]) / (r_p-r_m);
  } else {
    Real rr = r_c * std::sin(theta_c);
    Real z = r_c * std::cos(theta_c);
    bb1 = rr * z / det;
    bb2 = -SQR(rr) / (r_c * det);
    bb3 = 0.0;
  }

  // Calculate beta
  Real pmag = CalculateMagneticPressure(bb1, bb2, bb3, r_c, theta_c, phi_c);
  *pbeta = pgas/pmag;
  return true;
}

//----------------------------------------------------------------------------------------
// Function for calculating beta given vector potential
// Inputs:
//   r_m,r_c,r_p: inner, center, and outer radii
//   theta_m,theta_c,theta_p: upper, center, and lower polar angles
//   a_cm,a_cp,a_mc,a_pc: A_phi offset by theta (down,up) and r (down,up)
// Outputs:
//   pbeta: value set to plasma beta at cell center
//   returned value: true if pbeta points to meaningful number (inside torus)
// Notes:
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)

static bool CalculateBetaFromA(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
              Real theta_p, Real a_cm, Real a_cp, Real a_mc, Real a_pc, Real *pbeta) {
  // Calculate trigonometric functions of theta
  Real sin_theta_c = std::sin(theta_c);
  Real cos_theta_c = std::cos(theta_c);

  // Determine if we are in the torus (FM 3.6)
  if (r_m < r_edge) {
    return false;
  }
  Real log_h = LogHAux(r_c, sin_theta_c) - log_h_edge;
  if (log_h < 0.0) {
    return false;
  }

  // Calculate primitives
  Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
  Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
  Real pgas = pgas_over_rho * rho;

  // Check A_phi
  if (a_cm == 0.0 or a_cp == 0.0 or a_mc == 0.0 or a_pc == 0.0) {
    return false;
  }

  // Calculate 3-magnetic field
  Real det = (SQR(r_c) + SQR(a) * SQR(cos_theta_c)) * std::abs(sin_theta_c);
  Real bb1 = 1.0/det * (a_cp-a_cm) / (theta_p-theta_m);
  Real bb2 = -1.0/det * (a_pc-a_mc) / (r_p-r_m);
  Real bb3 = 0.0;

  // Calculate beta
  Real pmag = CalculateMagneticPressure(bb1, bb2, bb3, r_c, theta_c, 0.0);
  *pbeta = pgas/pmag;
  return true;
}

//----------------------------------------------------------------------------------------
// Function to calculate 1/2 * b^lambda b_lambda
// Inputs:
//   bb1,bb2,bb3: components of 3-magnetic field in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates
// Outputs:
//   returned value: magnetic pressure

static Real CalculateMagneticPressure(Real bb1, Real bb2, Real bb3, Real r, Real theta,
                                      Real phi) {
  // Calculate Boyer-Lindquist metric
  Real sin_theta = std::sin(theta);
  Real cos_theta = std::cos(theta);
  Real delta = SQR(r) - 2.0*m*r + SQR(a);
  Real sigma = SQR(r) + SQR(a) * SQR(cos_theta);
  Real g_00 = -(1.0 - 2.0*m*r/sigma);
  Real g_01 = 0.0;
  Real g_02 = 0.0;
  Real g_03 = -2.0*m*a*r/sigma * SQR(sin_theta);
  Real g_11 = sigma/delta;
  Real g_12 = 0.0;
  Real g_13 = 0.0;
  Real g_22 = sigma;
  Real g_23 = 0.0;
  Real g_33 = (SQR(r) + SQR(a) + 2.0*m*SQR(a)*r/sigma * SQR(sin_theta)) * SQR(sin_theta);
  Real g_10 = g_01;
  Real g_20 = g_02;
  Real g_21 = g_12;
  Real g_30 = g_03;
  Real g_31 = g_13;
  Real g_32 = g_23;

  // Calculate 4-velocity
  Real u0, u1, u2, u3;
  CalculateVelocityInTiltedTorus(r, theta, phi, &u0, &u1, &u2, &u3);

  // Calculate 4-magnetic field
  Real b0 = bb1 * (g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3)
          + bb2 * (g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3)
          + bb3 * (g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3);
  Real b1 = 1.0/u0 * (bb1 + b0 * u1);
  Real b2 = 1.0/u0 * (bb2 + b0 * u2);
  Real b3 = 1.0/u0 * (bb3 + b0 * u3);

  // Calculate magnetic pressure
  Real b_sq = g_00*b0*b0 + g_01*b0*b1 + g_02*b0*b2 + g_03*b0*b3
            + g_10*b1*b0 + g_11*b1*b1 + g_12*b1*b2 + g_13*b1*b3
            + g_20*b2*b0 + g_21*b2*b1 + g_22*b2*b2 + g_23*b2*b3
            + g_30*b3*b0 + g_31*b3*b1 + g_32*b3*b2 + g_33*b3*b3;
  return 0.5*b_sq;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates
// Notes:
//   conversion is trivial in all currently implemented coordinate systems

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi) {

    Real x = x1;
    Real y = x2;
    Real z = x3;
    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);

    *pr = r;
    *ptheta = std::acos(z/r);
    *pphi = std::atan2( (r*y-a*x)/(SQR(r)+SQR(a) ), (a*y+r*x)/(SQR(r) + SQR(a) )  );
  return;
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
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

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
                     Real x2, Real x3, Real *pa1, Real *pa2, Real *pa3) {

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



void get_bh_position(Real t, Real *xbh, Real *ybh, Real *zbh){

  *xbh = 0.0;
  *ybh = 0.0;
  *zbh = v_bh2 * (t) - 80.0;

}
void get_prime_coords(Real x, Real y, Real z, Real t, Real *xprime, Real *yprime, Real *zprime, Real *rprime, Real *Rprime){

  Real xbh,ybh,zbh;
  get_bh_position(t,&xbh,&ybh,&zbh);

  *xprime = x - xbh;
  *yprime = y - ybh;
  *zprime = z - zbh;


  if (std::fabs(*zprime)<SMALL) *zprime= SMALL;
  *Rprime = std::sqrt(SQR(*xprime) + SQR(*yprime) + SQR(*zprime));
  *rprime = SQR(*Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(*Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(*zprime) );
  *rprime = std::sqrt(*rprime/2.0);

}



void cks_metric(Real x1, Real x2, Real x3,AthenaArray<Real> &g){
    // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a_spin = a; //-a;

  if (std::fabs(z)<SMALL) z= SMALL;

  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x=  0.1;
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
  g(I00) = eta[0] + f * l_lower[0]*l_lower[0];
  g(I01) = f * l_lower[0]*l_lower[1];
  g(I02) = f * l_lower[0]*l_lower[2];
  g(I03) = f * l_lower[0]*l_lower[3];
  g(I11) = eta[1] + f * l_lower[1]*l_lower[1];
  g(I12) = f * l_lower[1]*l_lower[2];
  g(I13) = f * l_lower[1]*l_lower[3];
  g(I22) = eta[2] + f * l_lower[2]*l_lower[2];
  g(I23) = f * l_lower[2]*l_lower[3];
  g(I33) = eta[3] + f * l_lower[3]*l_lower[3];


}

void cks_inverse_metric(Real x1, Real x2, Real x3,AthenaArray<Real> &g_inv){
    // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a_spin = a; //-a;

  if (std::fabs(z)<SMALL) z= SMALL;

  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x=  0.1;
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
    // // Set contravariant components
  g_inv(I00) = eta[0] - f * l_upper[0]*l_upper[0] ;
  g_inv(I01) =        - f * l_upper[0]*l_upper[1] ;
  g_inv(I02) =        - f * l_upper[0]*l_upper[2] ;
  g_inv(I03) =        - f * l_upper[0]*l_upper[3] ;
  g_inv(I11) = eta[1] - f * l_upper[1]*l_upper[1] ;
  g_inv(I12) =        - f * l_upper[1]*l_upper[2] ;
  g_inv(I13) =        - f * l_upper[1]*l_upper[3] ;
  g_inv(I22) = eta[2] - f * l_upper[2]*l_upper[2] ;
  g_inv(I23) =        - f * l_upper[2]*l_upper[3] ;
  g_inv(I33) = eta[3] - f * l_upper[3]*l_upper[3] ;


}
void delta_cks_metric(ParameterInput *pin,Real t, Real x1, Real x2, Real x3,AthenaArray<Real> &delta_g){
  q = pin->GetOrAddReal("problem", "q", 0.1);
  aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check

  Real x = x1;
  Real y = x2;
  Real z = x3;


  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);


/// prevent metric from getting nan sqrt(-gdet)
  Real thprime  = std::acos(zprime/rprime);
  Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
                              (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

  Real rhprime = ( q + std::sqrt(q*q-SQR(aprime)) );
  if (rprime<rhprime/2.0) {
    rprime = rhprime/2.0;
    xprime = rprime * std::cos(phiprime)*std::sin(thprime) - aprime * std::sin(phiprime)*std::sin(thprime);
    yprime = rprime * std::sin(phiprime)*std::sin(thprime) + aprime * std::cos(phiprime)*std::sin(thprime);
    zprime = rprime * std::cos(thprime);
  }



  //if (r<0.01) r = 0.01;


  Real l_lowerprime[4],l_upperprime[4];

  Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));
  l_upperprime[0] = -1.0;
  l_upperprime[1] = (rprime*xprime + aprime*yprime)/( SQR(rprime) + SQR(aprime) );
  l_upperprime[2] = (rprime*yprime - aprime*xprime)/( SQR(rprime) + SQR(aprime) );
  l_upperprime[3] = zprime/rprime;

  l_lowerprime[0] = 1.0;
  l_lowerprime[1] = l_upperprime[1];
  l_lowerprime[2] = l_upperprime[2];
  l_lowerprime[3] = l_upperprime[3];






  // Set covariant components
  delta_g(I00) = fprime * l_lowerprime[0]*l_lowerprime[0];
  delta_g(I01) = fprime * l_lowerprime[0]*l_lowerprime[1];
  delta_g(I02) = fprime * l_lowerprime[0]*l_lowerprime[2];
  delta_g(I03) = fprime * l_lowerprime[0]*l_lowerprime[3];
  delta_g(I11) = fprime * l_lowerprime[1]*l_lowerprime[1];
  delta_g(I12) = fprime * l_lowerprime[1]*l_lowerprime[2];
  delta_g(I13) = fprime * l_lowerprime[1]*l_lowerprime[3];
  delta_g(I22) = fprime * l_lowerprime[2]*l_lowerprime[2];
  delta_g(I23) = fprime * l_lowerprime[2]*l_lowerprime[3];
  delta_g(I33) = fprime * l_lowerprime[3]*l_lowerprime[3];

}
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


#define DEL 1e-7
void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{

  a = pin->GetReal("coord", "a");
  m = pin->GetReal("coord", "m");

  //////////////Perturber Black Hole//////////////////

  q = pin->GetOrAddReal("problem", "q", 1.0);
  aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check
  v_bh2 = pin->GetOrAddReal("problem", "vbh", 0.05);

  Binary_BH_Metric(t,x1,x2,x3,g,g_inv,dg_dx1,dg_dx2,dg_dx3,dg_dt);

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

  a = pin->GetReal("coord", "a");
  Real a_spin = a;

  if ((std::fabs(z)<SMALL) && ( z>=0 )) z=  SMALL;
  if ((std::fabs(z)<SMALL) && ( z<0  )) z= -SMALL;

  // if ((std::fabs(x)<SMALL) && (x>=0)) x= SMALL;
  // if ((std::fabs(x)<SMALL) && (x<0)) x= -SMALL;

  // if ((std::fabs(y)<SMALL) && (y>=0)) y= SMALL;
  // if ((std::fabs(y)<SMALL) && (y<0)) y= -SMALL;  

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

#define DEL 1e-7
void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{

  // if  (Globals::my_rank == 0) fprintf(stderr,"Metric time in pgen file (GLOBAL RANK): %g \n", t);
  // else fprintf(stderr,"Metric time in pgen file (RANK %d): %g \n", Globals::my_rank,t);
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a_spin =a;

  Real eta[4];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;



  //////////////Perturber Black Hole//////////////////


  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);


  Real dx_bh2_dt = 0.0;
  Real dy_bh2_dt = 0.0;
  Real dz_bh2_dt = v_bh2;




/// prevent metric from getting nan sqrt(-gdet)
  Real thprime  = std::acos(zprime/rprime);
  Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
                              (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

  Real rhprime = ( q + std::sqrt(SQR(q)-SQR(aprime)) );
  if (rprime < rhprime*0.8) {
    rprime = rhprime*0.8;
    xprime = rprime * std::cos(phiprime)*std::sin(thprime) - aprime * std::sin(phiprime)*std::sin(thprime);
    yprime = rprime * std::sin(phiprime)*std::sin(thprime) + aprime * std::cos(phiprime)*std::sin(thprime);
    zprime = rprime * std::cos(thprime);
  }



  //if (r<0.01) r = 0.01;


  Real l_lowerprime[4],l_upperprime[4];

  Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));
  l_upperprime[0] = -1.0;
  l_upperprime[1] = (rprime*xprime + aprime*yprime)/( SQR(rprime) + SQR(aprime) );
  l_upperprime[2] = (rprime*yprime - aprime*xprime)/( SQR(rprime) + SQR(aprime) );
  l_upperprime[3] = zprime/rprime;

  l_lowerprime[0] = 1.0;
  l_lowerprime[1] = l_upperprime[1];
  l_lowerprime[2] = l_upperprime[2];
  l_lowerprime[3] = l_upperprime[3];






  // Set covariant components
  // g(I00) = eta[0] + fprime * l_lowerprime[0]*l_lowerprime[0] + v_bh2 * fprime * l_lowerprime[0]*l_lowerprime[3];
  // g(I01) =          fprime * l_lowerprime[0]*l_lowerprime[1] + v_bh2 * fprime * l_lowerprime[1]*l_lowerprime[3];
  // g(I02) =          fprime * l_lowerprime[0]*l_lowerprime[2] + v_bh2 * fprime * l_lowerprime[2]*l_lowerprime[3];
  // g(I03) =          fprime * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * l_lowerprime[3]*l_lowerprime[3] + v_bh2;
  g(I00) = eta[0] + fprime * l_lowerprime[0]*l_lowerprime[0]; // - 2.0*v_bh2 * fprime * l_lowerprime[0]*l_lowerprime[3]  
                  //+ SQR(v_bh2)*fprime*l_lowerprime[3]*l_lowerprime[3] + SQR(v_bh2) ;
  g(I01) =          fprime * l_lowerprime[0]*l_lowerprime[1]; // - v_bh2 * fprime * l_lowerprime[1]*l_lowerprime[3];
  g(I02) =          fprime * l_lowerprime[0]*l_lowerprime[2]; // - v_bh2 * fprime * l_lowerprime[2]*l_lowerprime[3];
  g(I03) =          fprime * l_lowerprime[0]*l_lowerprime[3]; // - v_bh2 * fprime * l_lowerprime[3]*l_lowerprime[3] - v_bh2;
  g(I11) = eta[1] + fprime * l_lowerprime[1]*l_lowerprime[1];
  g(I12) =          fprime * l_lowerprime[1]*l_lowerprime[2];
  g(I13) =          fprime * l_lowerprime[1]*l_lowerprime[3];
  g(I22) = eta[2] + fprime * l_lowerprime[2]*l_lowerprime[2];
  g(I23) =          fprime * l_lowerprime[2]*l_lowerprime[3];
  g(I33) = eta[3] + fprime * l_lowerprime[3]*l_lowerprime[3];

  // Real det_test = Determinant(g);

  // if (std::isnan( std::sqrt(-det_test))) {
  //   fprintf(stderr,"NAN determinant in metric!! Det: %g \n xyz: %g %g %g \n r: %g \n",det_test,x,y,z,r);
  //   exit(0);
  // }


  bool invertible = gluInvertMatrix(g,g_inv);

  if (invertible==false) {
    fprintf(stderr,"Non-invertible matrix at xyz: %g %g %g\n", x,y,z);
  }


  //expressioons for a = 0

  // f = 2.0/R;
  // l_lower[1] = x/R;
  // l_lower[2] = y/R;
  // l_lower[3] = z/R;
  // df_dx1 = -2.0 * x/SQR(R)/R;
  // df_dx2 = -2.0 * y/SQR(R)/R;
  // df_dx3 = -2.0 * z/SQR(R)/R;

  // dl1_dx1 = -SQR(x)/SQR(R)/R + 1.0/R;
  // dl1_dx2 = -x*y/SQR(R)/R; 
  // dl1_dx3 = -x*z/SQR(R)/R;

  // dl2_dx1 = -x*y/SQR(R)/R;
  // dl2_dx2 = -SQR(y)/SQR(R)/R + 1.0/R;
  // dl2_dx3 = -y*z/SQR(R)/R;

  // dl3_dx1 = -x*z/SQR(R)/R;
  // dl3_dx2 = -y*z/SQR(R)/R;
  // dl3_dx3 = -SQR(z)/SQR(R)/R;



  // // Set x-derivatives of covariant components
  dg_dx1(I00) = 0.0;
  dg_dx1(I01) = 0.0;
  dg_dx1(I02) = 0.0;
  dg_dx1(I03) = 0.0;
  dg_dx1(I11) = 0.0;
  dg_dx1(I12) = 0.0;
  dg_dx1(I13) = 0.0;
  dg_dx1(I22) = 0.0;
  dg_dx1(I23) = 0.0;
  dg_dx1(I33) = 0.0;

  // Set y-derivatives of covariant components
  dg_dx2(I00) = 0.0;
  dg_dx2(I01) = 0.0;
  dg_dx2(I02) = 0.0;
  dg_dx2(I03) = 0.0;
  dg_dx2(I11) = 0.0;
  dg_dx2(I12) = 0.0;
  dg_dx2(I13) = 0.0;
  dg_dx2(I22) = 0.0;
  dg_dx2(I23) = 0.0;
  dg_dx2(I33) = 0.0;

  // Set z-derivatives of covariant components
  dg_dx3(I00) = 0.0;
  dg_dx3(I01) = 0.0;
  dg_dx3(I02) = 0.0;
  dg_dx3(I03) = 0.0;
  dg_dx3(I11) = 0.0;
  dg_dx3(I12) = 0.0;
  dg_dx3(I13) = 0.0;
  dg_dx3(I22) = 0.0;
  dg_dx3(I23) = 0.0;
  dg_dx3(I33) = 0.0;



/////Secondary Black hole/////

  Real sqrt_term =  2.0*SQR(rprime)-SQR(Rprime) + SQR(aprime);
  Real rsq_p_asq = SQR(rprime) + SQR(aprime);

  Real fprime_over_q = 2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));


  Real dfprime_dx1 = q * SQR(fprime_over_q)*xprime/(2.0*std::pow(rprime,3)) * 
                      ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
  //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
  Real dfprime_dx2 = q * SQR(fprime_over_q)*yprime/(2.0*std::pow(rprime,3)) * 
                      ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
  Real dfprime_dx3 = q * SQR(fprime_over_q)*zprime/(2.0*std::pow(rprime,5)) * 
                      ( ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(aprime*rprime)) ;
  //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
  Real dl1prime_dx1 = xprime*rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + rprime/( rsq_p_asq );
  // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
  Real dl1prime_dx2 = yprime*rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ aprime/( rsq_p_asq );
  Real dl1prime_dx3 = zprime/rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( (rsq_p_asq) * ( sqrt_term ) ) ;
  Real dl2prime_dx1 = xprime*rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - aprime/( rsq_p_asq );
  Real dl2prime_dx2 = yprime*rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + rprime/( rsq_p_asq );
  Real dl2prime_dx3 = zprime/rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( (rsq_p_asq) * ( sqrt_term ) );
  Real dl3prime_dx1 = - xprime*zprime/(rprime) /( sqrt_term );
  Real dl3prime_dx2 = - yprime*zprime/(rprime) /( sqrt_term );
  Real dl3prime_dx3 = - SQR(zprime)/(SQR(rprime)*rprime) * ( rsq_p_asq )/( sqrt_term ) + 1.0/rprime;

  Real dl0prime_dx1 = 0.0;
  Real dl0prime_dx2 = 0.0;
  Real dl0prime_dx3 = 0.0;

  AthenaArray<Real> dgprime_dx1, dgprime_dx2, dgprime_dx3;

  dgprime_dx1.NewAthenaArray(NMETRIC);
  dgprime_dx2.NewAthenaArray(NMETRIC);
  dgprime_dx3.NewAthenaArray(NMETRIC);

  // // Set x-derivatives of covariant components
  // dgprime_dx1(I00) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx1 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx1
  //                    + v_bh2 * dfprime_dx1 * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * dl0prime_dx1*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[0]*dl3prime_dx1;
  // dgprime_dx1(I01) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx1;
  //                    + v_bh2 * dfprime_dx1 * l_lowerprime[1]*l_lowerprime[3] + v_bh2 * fprime * dl1prime_dx1*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[1]*dl3prime_dx1;
  // dgprime_dx1(I02) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx1
  //                    + v_bh2 * dfprime_dx1 * l_lowerprime[2]*l_lowerprime[3] + v_bh2 * fprime * dl2prime_dx1*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[2]*dl3prime_dx1;
  // dgprime_dx1(I03) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx1
  //                    + v_bh2 * dfprime_dx1 * l_lowerprime[3]*l_lowerprime[3] + v_bh2 * fprime * dl3prime_dx1*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[3]*dl3prime_dx1;  

  dgprime_dx1(I00) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx1 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx1;
                    // - 2.0 * v_bh2 * (dfprime_dx1 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1*l_lowerprime[3]
                    // +                fprime * l_lowerprime[0]*dl3prime_dx1) 
                    // +  SQR(v_bh2) * (dfprime_dx1 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx1*l_lowerprime[3] 
                    // +                fprime * l_lowerprime[3]*dl3prime_dx1);
  dgprime_dx1(I01) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx1;
                    // - v_bh2 * (dfprime_dx1 * l_lowerprime[1]*l_lowerprime[3] 
                    // +          fprime * dl1prime_dx1*l_lowerprime[3]
                    // +          fprime * l_lowerprime[1]*dl3prime_dx1);
  dgprime_dx1(I02) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx1;
                    // - v_bh2 * (dfprime_dx1 * l_lowerprime[2]*l_lowerprime[3] 
                    // +          fprime * dl2prime_dx1*l_lowerprime[3]
                    // +          fprime * l_lowerprime[2]*dl3prime_dx1);
  dgprime_dx1(I03) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx1;
                    // - v_bh2 * (dfprime_dx1 * l_lowerprime[3]*l_lowerprime[3] 
                    // +          fprime * dl3prime_dx1*l_lowerprime[3]
                    // +          fprime * l_lowerprime[3]*dl3prime_dx1);  
  dgprime_dx1(I11) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx1;
  dgprime_dx1(I12) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx1;
  dgprime_dx1(I13) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx1;
  dgprime_dx1(I22) = dfprime_dx1*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx1;
  dgprime_dx1(I23) = dfprime_dx1*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx1;
  dgprime_dx1(I33) = dfprime_dx1*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx1;

  // Set y-derivatives of covariant components
  // dgprime_dx2(I00) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx2 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx2
  //                    + v_bh2 * dfprime_dx2 * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * dl0prime_dx2*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[0]*dl3prime_dx2;
  // dgprime_dx2(I01) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx2
  //                    + v_bh2 * dfprime_dx2 * l_lowerprime[1]*l_lowerprime[3] + v_bh2 * fprime * dl1prime_dx2*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[1]*dl3prime_dx2;
  // dgprime_dx2(I02) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx2
  //                    + v_bh2 * dfprime_dx2 * l_lowerprime[2]*l_lowerprime[3] + v_bh2 * fprime * dl2prime_dx2*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[2]*dl3prime_dx2;
  // dgprime_dx2(I03) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx2
  //                    + v_bh2 * dfprime_dx2 * l_lowerprime[3]*l_lowerprime[3] + v_bh2 * fprime * dl3prime_dx2*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[3]*dl3prime_dx2;  
  dgprime_dx2(I00) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx2 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx2;
                    // - 2.0 * v_bh2 * (dfprime_dx2 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2*l_lowerprime[3]
                    // +                fprime * l_lowerprime[0]*dl3prime_dx2) 
                    // +  SQR(v_bh2) * (dfprime_dx2 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx2*l_lowerprime[3] 
                    // +                fprime * l_lowerprime[3]*dl3prime_dx2);
  dgprime_dx2(I01) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx2;
                    // - v_bh2 * (dfprime_dx2 * l_lowerprime[1]*l_lowerprime[3] 
                    // +          fprime * dl1prime_dx2*l_lowerprime[3]
                    // +          fprime * l_lowerprime[1]*dl3prime_dx2);
  dgprime_dx2(I02) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx2;
                    // - v_bh2 * (dfprime_dx2 * l_lowerprime[2]*l_lowerprime[3] 
                    // +          fprime * dl2prime_dx2*l_lowerprime[3]
                    // +          fprime * l_lowerprime[2]*dl3prime_dx2);
  dgprime_dx2(I03) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx2;
                    // - v_bh2 * (dfprime_dx2 * l_lowerprime[3]*l_lowerprime[3] 
                    // +          fprime * dl3prime_dx2*l_lowerprime[3]
                    // +          fprime * l_lowerprime[3]*dl3prime_dx2);  
  dgprime_dx2(I11) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx2;
  dgprime_dx2(I12) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx2;
  dgprime_dx2(I13) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx2;
  dgprime_dx2(I22) = dfprime_dx2*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx2;
  dgprime_dx2(I23) = dfprime_dx2*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx2;
  dgprime_dx2(I33) = dfprime_dx2*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx2;

  // Set z-derivatives of covariant components
  // dgprime_dx3(I00) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx3 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx3
  //                    + v_bh2 * dfprime_dx3 * l_lowerprime[0]*l_lowerprime[3] + v_bh2 * fprime * dl0prime_dx3*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[0]*dl3prime_dx3;
  // dgprime_dx3(I01) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx3
  //                     + v_bh2 * dfprime_dx3 * l_lowerprime[1]*l_lowerprime[3] + v_bh2 * fprime * dl1prime_dx3*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[1]*dl3prime_dx3;
  // dgprime_dx3(I02) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx3
  //                      + v_bh2 * dfprime_dx3 * l_lowerprime[2]*l_lowerprime[3] + v_bh2 * fprime * dl2prime_dx3*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[2]*dl3prime_dx3;;
  // dgprime_dx3(I03) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx3
  //                    + v_bh2 * dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] + v_bh2 * fprime * dl3prime_dx3*l_lowerprime[3]
  //                    + v_bh2 * fprime * l_lowerprime[3]*dl3prime_dx3; ;

  dgprime_dx3(I00) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx3 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx3;
                    // - 2.0 * v_bh2 * (dfprime_dx3 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3*l_lowerprime[3]
                    // +                fprime * l_lowerprime[0]*dl3prime_dx3) 
                    // +  SQR(v_bh2) * (dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx3*l_lowerprime[3] 
                    // +                fprime * l_lowerprime[3]*dl3prime_dx3);
  dgprime_dx3(I01) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx3;
                    // - v_bh2 * (dfprime_dx3 * l_lowerprime[1]*l_lowerprime[3] 
                    // +          fprime * dl1prime_dx3*l_lowerprime[3]
                    // +          fprime * l_lowerprime[1]*dl3prime_dx3);
  dgprime_dx3(I02) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx3;
                    // - v_bh2 * (dfprime_dx3 * l_lowerprime[2]*l_lowerprime[3] 
                    // +          fprime * dl2prime_dx3*l_lowerprime[3]
                    // +          fprime * l_lowerprime[2]*dl3prime_dx3);
  dgprime_dx3(I03) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx3;
                    // - v_bh2 * (dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] 
                    // +          fprime * dl3prime_dx3*l_lowerprime[3]
                    // +          fprime * l_lowerprime[3]*dl3prime_dx3);  
  dgprime_dx3(I11) = dfprime_dx3*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx3;
  dgprime_dx3(I12) = dfprime_dx3*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx3;
  dgprime_dx3(I13) = dfprime_dx3*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx3;
  dgprime_dx3(I22) = dfprime_dx3*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx3;
  dgprime_dx3(I23) = dfprime_dx3*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx3;
  dgprime_dx3(I33) = dfprime_dx3*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx3;





  // // Set x-derivatives of covariant components
  dg_dx1(I00) += dgprime_dx1(I00);
  dg_dx1(I01) += dgprime_dx1(I01);
  dg_dx1(I02) += dgprime_dx1(I02);
  dg_dx1(I03) += dgprime_dx1(I03);
  dg_dx1(I11) += dgprime_dx1(I11);
  dg_dx1(I12) += dgprime_dx1(I12);
  dg_dx1(I13) += dgprime_dx1(I13);
  dg_dx1(I22) += dgprime_dx1(I22);
  dg_dx1(I23) += dgprime_dx1(I23);
  dg_dx1(I33) += dgprime_dx1(I33);

  // Set y-derivatives of covariant components
  dg_dx2(I00) += dgprime_dx2(I00);
  dg_dx2(I01) += dgprime_dx2(I01);
  dg_dx2(I02) += dgprime_dx2(I02);
  dg_dx2(I03) += dgprime_dx2(I03);
  dg_dx2(I11) += dgprime_dx2(I11);
  dg_dx2(I12) += dgprime_dx2(I12);
  dg_dx2(I13) += dgprime_dx2(I13);
  dg_dx2(I22) += dgprime_dx2(I22);
  dg_dx2(I23) += dgprime_dx2(I23);
  dg_dx2(I33) += dgprime_dx2(I33);

  // Set z-derivatives of covariant components
  dg_dx3(I00) += dgprime_dx3(I00);
  dg_dx3(I01) += dgprime_dx3(I01);
  dg_dx3(I02) += dgprime_dx3(I02);
  dg_dx3(I03) += dgprime_dx3(I03);
  dg_dx3(I11) += dgprime_dx3(I11);
  dg_dx3(I12) += dgprime_dx3(I12);
  dg_dx3(I13) += dgprime_dx3(I13);
  dg_dx3(I22) += dgprime_dx3(I22);
  dg_dx3(I23) += dgprime_dx3(I23);
  dg_dx3(I33) += dgprime_dx3(I33);





  // Set t-derivatives of covariant components
  dg_dt(I00) = -1.0 * (dx_bh2_dt * dgprime_dx1(I00) + dy_bh2_dt * dgprime_dx2(I00) + dz_bh2_dt * dgprime_dx3(I00) );
  dg_dt(I01) = -1.0 * (dx_bh2_dt * dgprime_dx1(I01) + dy_bh2_dt * dgprime_dx2(I01) + dz_bh2_dt * dgprime_dx3(I01) );
  dg_dt(I02) = -1.0 * (dx_bh2_dt * dgprime_dx1(I02) + dy_bh2_dt * dgprime_dx2(I02) + dz_bh2_dt * dgprime_dx3(I02) );
  dg_dt(I03) = -1.0 * (dx_bh2_dt * dgprime_dx1(I03) + dy_bh2_dt * dgprime_dx2(I03) + dz_bh2_dt * dgprime_dx3(I03) );
  dg_dt(I11) = -1.0 * (dx_bh2_dt * dgprime_dx1(I11) + dy_bh2_dt * dgprime_dx2(I11) + dz_bh2_dt * dgprime_dx3(I11) );
  dg_dt(I12) = -1.0 * (dx_bh2_dt * dgprime_dx1(I12) + dy_bh2_dt * dgprime_dx2(I12) + dz_bh2_dt * dgprime_dx3(I12) );
  dg_dt(I13) = -1.0 * (dx_bh2_dt * dgprime_dx1(I13) + dy_bh2_dt * dgprime_dx2(I13) + dz_bh2_dt * dgprime_dx3(I13) );
  dg_dt(I22) = -1.0 * (dx_bh2_dt * dgprime_dx1(I22) + dy_bh2_dt * dgprime_dx2(I22) + dz_bh2_dt * dgprime_dx3(I22) );
  dg_dt(I23) = -1.0 * (dx_bh2_dt * dgprime_dx1(I23) + dy_bh2_dt * dgprime_dx2(I23) + dz_bh2_dt * dgprime_dx3(I23) );
  dg_dt(I33) = -1.0 * (dx_bh2_dt * dgprime_dx1(I33) + dy_bh2_dt * dgprime_dx2(I33) + dz_bh2_dt * dgprime_dx3(I33) );


  dgprime_dx1.DeleteAthenaArray();
  dgprime_dx2.DeleteAthenaArray();
  dgprime_dx3.DeleteAthenaArray();


  // AthenaArray<Real> gp,gm;
  // AthenaArray<Real> delta_gp,delta_gm;



  // gp.NewAthenaArray(NMETRIC);
  // gm.NewAthenaArray(NMETRIC);
  // delta_gp.NewAthenaArray(NMETRIC);
  // delta_gm.NewAthenaArray(NMETRIC);



  // cks_metric(x1,x2,x3,g);
  // delta_cks_metric(pin,t,x1,x2,x3,delta_gp);

  // for (int n = 0; n < NMETRIC; ++n) {
  //    g(n) += delta_gp(n);
  // }


  // cks_inverse_metric(x1,x2,x3,g_inv);
  // delta_cks_metric_inverse(pin,t,x1,x2,x3,delta_gp);

  // for (int n = 0; n < NMETRIC; ++n) {
  //    g_inv(n) += delta_gp(n);
  // }

  // Real x1p = x1 + DEL * rprime;
  // Real x1m = x1 - DEL * rprime;

  // cks_metric(x1p,x2,x3,gp);
  // cks_metric(x1m,x2,x3,gm);
  // delta_cks_metric(pin,t,x1p,x2,x3,delta_gp);
  // delta_cks_metric(pin,t,x1m,x2,x3,delta_gm);

  // for (int n = 0; n < NMETRIC; ++n) {
  //    gp(n) += delta_gp(n);
  //    gm(n) += delta_gm(n);
  // }

  //   // // Set x-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx1(n) = (gp(n)-gm(n))/(x1p-x1m);
  // }

  // Real x2p = x2 + DEL * rprime;
  // Real x2m = x2 - DEL * rprime;

  // cks_metric(x1,x2p,x3,gp);
  // cks_metric(x1,x2m,x3,gm);
  // delta_cks_metric(pin,t,x1,x2p,x3,delta_gp);
  // delta_cks_metric(pin,t,x1,x2m,x3,delta_gm);
  // for (int n = 0; n < NMETRIC; ++n) {
  //    gp(n) += delta_gp(n);
  //    gm(n) += delta_gm(n);
  // }

  //   // // Set y-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx2(n) = (gp(n)-gm(n))/(x2p-x2m);
  // }
  
  // Real x3p = x3 + DEL * rprime;
  // Real x3m = x3 - DEL * rprime;

  // cks_metric(x1,x2,x3p,gp);
  // cks_metric(x1,x2,x3m,gm);
  // delta_cks_metric(pin,t,x1,x2,x3p,delta_gp);
  // delta_cks_metric(pin,t,x1,x2,x3m,delta_gm);
  // for (int n = 0; n < NMETRIC; ++n) {
  //    gp(n) += delta_gp(n);
  //    gm(n) += delta_gm(n);
  // }

  //   // // Set z-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx3(n) = (gp(n)-gm(n))/(x3p-x3m);
  // }

  // Real tp = t + DEL ;
  // Real tm = t - DEL ;

  // cks_metric(x1,x2,x3,gp);
  // cks_metric(x1,x2,x3,gm);
  // delta_cks_metric(pin,tp,x1,x2,x3,delta_gp);
  // delta_cks_metric(pin,tm,x1,x2,x3,delta_gm);

  // for (int n = 0; n < NMETRIC; ++n) {
  //    gp(n) += delta_gp(n);
  //    gm(n) += delta_gm(n);
  // }

  //   // // Set t-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dt(n) = (gp(n)-gm(n))/(tp-tm);
  // }

  // gp.DeleteAthenaArray();
  // gm.DeleteAthenaArray();
  // delta_gm.DeleteAthenaArray();
  // delta_gp.DeleteAthenaArray();
  return;
}