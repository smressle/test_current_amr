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

int RefinementCondition(MeshBlock *pmb);
void  Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);
bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv);


void get_prime_coords(Real x, Real y, Real z, Real t, Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);
void get_bh_position(Real t, Real *xbh, Real *ybh, Real *zbh);
void get_bh_velocity(Real t, Real *vxbh, Real *vybh, Real *vzbh);
void get_bh_acceleration(Real t, Real *axbh, Real *aybh, Real *azbh);

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ);

void single_bh_metric(Real x1, Real x2, Real x3, ParameterInput *pin,AthenaArray<Real> &g);

void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
  AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt, bool take_derivatives);


void BoostVector(Real t, Real a0, Real a1, Real a2, Real a3, Real *pa0, Real *pa1, Real *pa2, Real *pa3);

Real DivergenceB(MeshBlock *pmb, int iout);

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
static Real extra_field_norm;                      // factor to multiply field by 
static Real beta_min;                              // min ratio of gas to mag pressure
static Real x1_min, x1_max, x2_min, x2_max;        // 2D limits in chosen coordinates
static Real x3_min, x3_max;                        // 3D limits in chosen coordinates
static Real r_min, r_max, theta_min, theta_max;    // limits in r,theta for 2D samples
static Real phi_min, phi_max;                      // limits in phi for 3D samples
static Real pert_amp, pert_kr, pert_kz;            // parameters for initial perturbations
static Real dfloor,pfloor;                         // density and pressure floors
static Real rh;                                    // horizon radius
static Real n_pow;

static Real aprime,q;          // black hole mass and spin
static Real r_inner_boundary,r_inner_boundary_2;
static Real rh2;
static Real r_bh2;
static Real Omega_bh2;
static Real eccentricity, tau, mean_angular_motion;
static Real t0; //time at which second BH is at polar axis
static Real orbit_inclination;
static Real H_over_r_target;

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

/******************************************/
/*        Rotation Functions              */
/******************************************/


// void pre_compute_rotation_matrix(const Real alpha,const Real beta,const Real gamma) {
    
//     double X_rot[3][3];
//     double Z_rot[3][3];
//     double Z_rot2[3][3];
//     double tmp[3][3],rot[3][3];
//     int i,j,k;
    
    
//     Z_rot2[0][0] = std::cos(gamma);
//     Z_rot2[0][1] = -std::sin(gamma);
//     Z_rot2[0][2] = 0.;
//     Z_rot2[1][0] = std::sin(gamma);
//     Z_rot2[1][1] = std::cos(gamma);
//     Z_rot2[1][2] = 0.;
//     Z_rot2[2][0] = 0.;
//     Z_rot2[2][1] = 0.;
//     Z_rot2[2][2] = 1.;
    
//     X_rot[0][0] = 1.;
//     X_rot[0][1] = 0.;
//     X_rot[0][2] = 0.;
//     X_rot[1][0] = 0.;
//     X_rot[1][1] = std::cos(beta);
//     X_rot[1][2] = -std::sin(beta);
//     X_rot[2][0] = 0.;
//     X_rot[2][1] = std::sin(beta);
//     X_rot[2][2] = std::cos(beta);
    
//     Z_rot[0][0] = std::cos(alpha);
//     Z_rot[0][1] = -std::sin(alpha);
//     Z_rot[0][2] = 0.;
//     Z_rot[1][0] = std::sin(alpha);
//     Z_rot[1][1] = std::cos(alpha);
//     Z_rot[1][2] = 0.;
//     Z_rot[2][0] = 0.;
//     Z_rot[2][1] = 0.;
//     Z_rot[2][2] = 1.;
    
    
//     for (i=0; i<3; i++){
//         for (j=0; j<3; j++) {
//             rot[i][j] = 0.;
//             tmp[i][j] = 0.;
//         }
//     }
    
//     for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++) tmp[i][j] += X_rot[i][k] * Z_rot[k][j] ;
//     for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++) rot[i][j] += Z_rot2[i][k] * tmp[k][j] ;
    
    
//     for (i=0; i<3; i++){
//         for (j=0; j<3; j++) {
//             rotation_matrix[i][j] = rot[i][j] ;
//         }
//     }


    
// }
// void rotate_orbit(const Real alpha, const Real beta, const Real gamma, const Real x1_prime, const Real x2_prime, Real * x1, Real * x2, Real * x3)
// {
//   Real alpha,beta,gamma;

//   double X_rot[3][3];
//   double Z_rot[3][3];
//   double Z_rot2[3][3];
//   double tmp[3][3],rot[3][3];
//   double x_prime[3], x_result[3];
//   int i,j,k;

//   x_prime[0] = x1_prime;
//   x_prime[1] = x2_prime;
//   x_prime[2] = 0.;



//   for (i=0; i<3; i++) x_result[i] = 0.;

  
//   for (i=0; i<3; i++) for (j=0; j<3; j++) x_result[i] += rotation_matrix[j][i]*x_prime[j] ;   /*Note this is inverse rotation so rot[j,i] instead of rot[i,j] */


//     *x1 = x_result[0];
//     *x2 = x_result[1];
//     *x3 = x_result[2];


// }
/*
Solve Kepler's equation for a given star in the plane of the orbit and then rotate
to the lab frame
*/
// void get_bh2_position_from_Kepler(const Real t)
// {

//   Real mean_anomaly = mean_angular_motion * (t - tau);
//   Real a = std::pow(gm_/SQR(mean_angular_motion),1./3.);    //mean_angular_motion = np.sqrt(mu/(a*a*a));
//     Real b;
//     if (eccentricity <1){
//         b =a * sqrt(1. - SQR(eccentricity) );
//         mean_anomaly = fmod(mean_anomaly, 2*PI);
//         if (mean_anomaly >  PI) mean_anomaly = mean_anomaly- 2.0*PI;
//         if (mean_anomaly < -PI) mean_anomaly = mean_anomaly + 2.0*PI;
//     }
//     else{
//         b = a * sqrt(SQR(eccentricity) -1. );
//     }


//     //Construct the initial guess.
//     Real E;
//     if (eccentricity <1){
//       Real sgn = 1.0;
//       if (std::sin(mean_anomaly) < 0.0) sgn = -1.0;
//       E = mean_anomaly + sgn*(0.85)*eccentricity;
//      }
//     else{
//       Real sgn = 1.0;
//       if (std::sinh(-mean_anomaly) < 0.0) sgn = -1.0;
//       E = mean_anomaly;
//     }

//     //Solve kepler's equation iteratively to improve the solution E.
//     Real error = 1.0;
//     Real max_error = 1e-6;
//     int i_max = 100;
//     int i;

//     if (eccentricity <1){
//       for(i = 0; i < i_max; i++){
//         Real es = eccentricity*std::sin(E);
//         Real ec = eccentricity*std::cos(E);
//         Real f = E - es - mean_anomaly;
//         error = fabs(f);
//         if (error < max_error) break;
//         Real df = 1.0 - ec;
//         Real ddf = es;
//         Real dddf = ec;
//         Real d1 = -f/df;
//         Real d2 = -f/(df + d1*ddf/2.0);
//         Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
//         E = E + d3;
//       }
//     }
//     else{
//       for(i = 0; i < i_max; i++){
//         Real es = eccentricity*std::sinh(E);
//         Real ec = eccentricity*std::cosh(E);
//         Real f = E - es + mean_anomaly;
//         error = fabs(f);
//         if (error < max_error) break;
//         Real df = 1.0 - ec;
//         Real ddf = -es;
//         Real dddf = -ec;
//         Real d1 = -f/df;
//         Real d2 = -f/(df + d1*ddf/2.0);
//         Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
//         E = E + d3;
//       }
//     }

//      //Warn if solution did not converge.
//      if (error > max_error)
//        std::cout << "***Warning*** Orbit::keplers_eqn() failed to converge***\n";

//      Real x1_prime,x2_prime,v1_prime,v2_prime;
//     if (eccentricity<1){
//       x1_prime= a * (std::cos(E) - eccentricity) ;
//       x2_prime= b * std::sin(E) ;
      
//       /* Time Derivative of E */
//       Real Edot = mean_angular_motion/ (1.-eccentricity * std::cos(E));
      
//       v1_prime = - a * std::sin(E) * Edot;
//       v2_prime =   b * std::cos(E) * Edot;
//     }
//     else{
//       x1_prime = a * ( eccentricity - std::cosh(E) );
//       x2_prime = b * std::sinh(E);

//       /* Time Derivative of E */  
//       Real Edot = -mean_angular_motion/ (1. - eccentricity * std::cosh(E));

//       v1_prime = a * (-std::sinh(E)*Edot);
//       v2_prime = b * std::cosh(E) * Edot;
//     }

//     // Real x1,x2,x3;

//     // rotate_orbit(star,i_star, x1_prime, x2_prime,&x1,&x2,&x3 );
    
//     // star[i_star].x1 = x1;
//     // star[i_star].x2 = x2;
//     // star[i_star].x3 = x3;
    
//     // Real v1,v2,v3;
//     // rotate_orbit(star,i_star,v1_prime,v2_prime,&v1, &v2, &v3);
    
    
//     // star[i_star].v1 = v1;
//     // star[i_star].v2 = v2;
//     // star[i_star].v3 = v3;


  
// }
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

    potential_cutoff = pin->GetReal("problem", "potential_cutoff");
    potential_r_pow = pin->GetReal("problem", "potential_r_pow");
    potential_rho_pow = pin->GetReal("problem", "potential_rho_pow");
    potential_sinth_pow = pin->GetOrAddReal("problem", "potential_sinth_pow",0.0);
    potential_costh_pow = pin->GetOrAddReal("problem", "potential_costh_pow",0.0);

    extra_field_norm = pin->GetOrAddReal("problem", "extra_field_norm",1.0);



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

  AllocateUserHistoryOutput(1);

  EnrollUserHistoryOutput(0, DivergenceB, "divB");


  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);

  EnrollUserExplicitSourceFunction(NobleCooling);


  //init_orbit_tables();
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
  a = pcoord->GetSpin();
  q = pin->GetOrAddReal("problem", "q", 0.1);
  aprime = q * pin->GetOrAddReal("problem", "a_bh2", 0.0);
  r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);

  t0 = pin->GetOrAddReal("problem","t0", 1e4);

  orbit_inclination = pin->GetOrAddReal("problem","orbit_inclination",0.0);


  Real v_bh2 = 1.0/std::sqrt(r_bh2);
  // Omega_bh2 = 0.0; //
  Omega_bh2 = v_bh2/r_bh2;

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
  //Real bh2_focus_radius = 3.125*0.1;

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
  gamma_adi = peos->GetGamma();
  Real gam = gamma_adi;



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
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r,
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
          TransformVector(u0_bl, 0.0, u2_bl, u3_bl, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
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
      GetBoyerLindquistCoordinates(x1_val, x2_val, x3_val, r_vals+p, theta_vals+p,
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
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                a_phi_edges(k,j,i) = std::pow(r, potential_r_pow)
                    * std::pow(rho_cutoff, potential_rho_pow)
                    * std::pow(std::sin(theta),potential_sinth_pow)
                    * std::pow(std::cos(theta),potential_costh_pow);
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
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k,j,i) == true) {
                Real rho = phydro->w(IDN,k,j,i);
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                a_phi_cells(k,j,i) = std::pow(r, potential_r_pow)
                    * std::pow(rho_cutoff, potential_rho_pow)
                    * std::pow(std::sin(theta),potential_sinth_pow)
                    * std::pow(std::cos(theta),potential_costh_pow);
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
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
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
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),
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
          << "field_config must be \"normal\" or \"MAD\"" << std::endl;
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
            TransformAphi(a_phi_edges(k,j+1,i),pcoord->x1f(i), pcoord->x2f(j+1),pcoord->x3v(k),
                &tmp,&tmp,&Az_2);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2f(j),pcoord->x3v(k),
                &tmp,&tmp,&Az_1);
                  

            pfield->b.x1f(k,j,i) = 1.0/std::sqrt(-det) * (Az_2-Az_1) / (pcoord->dx2f(j) );

            //d Ay/dz
            Real  Ay_2,Ay_1;
            TransformAphi(a_phi_edges(k+1,j,i),pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k+1),
                &tmp,&Ay_2,&tmp);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k),
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
            TransformAphi(a_phi_edges(k+1,j,i),pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k+1),
                &Ax_2,&tmp,&tmp);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k),
                &Ax_1,&tmp,&tmp);
                  

            pfield->b.x2f(k,j,i) = 1.0/std::sqrt(-det) * (Ax_2-Ax_1) / (pcoord->dx3f(k) );

            //d Az/dx
            Real Az_2,Az_1;
            TransformAphi(a_phi_edges(k,j,i+1),pcoord->x1f(i+1), pcoord->x2f(j),pcoord->x3v(k),
                &tmp,&tmp,&Az_2);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2f(j),pcoord->x3v(k),
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
            TransformAphi(a_phi_edges(k,j,i+1),pcoord->x1f(i+1), pcoord->x2v(j),pcoord->x3f(k),
                &tmp,&Ay_2,&tmp);
            TransformAphi(a_phi_edges(k,j,i),  pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k),
                &tmp,&Ay_1,&tmp);
                  

            pfield->b.x3f(k,j,i) = 1.0/std::sqrt(-det) * (Ay_2-Ay_1) / (pcoord->dx1f(i) );

            //d Ax/dy
            Real Ax_2,Ax_1;
            TransformAphi(a_phi_edges(k,j+1,i),pcoord->x1v(i), pcoord->x2f(j+1),pcoord->x3f(k),
                &Ax_2,&tmp,&tmp);
            TransformAphi(a_phi_edges(k,j,i),  pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k),
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
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r,
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

          // if (rprime < rh2){
              

          //     //set uu assuming u is zero
          //     Real gamma = 1.0;
          //     Real alpha = std::sqrt(-1.0/gi(I00,i));
          //     Real u0 = gamma/alpha;
          //     Real uu1 = - gi(I01,i)/gi(I00,i) * u0;
          //     Real uu2 = - gi(I02,i)/gi(I00,i) * u0;
          //     Real uu3 = - gi(I03,i)/gi(I00,i) * u0;
              
          //     prim(IDN,k,j,i) = dfloor;
          //     prim(IVX,k,j,i) = 0.;
          //     prim(IVY,k,j,i) = 0.;
          //     prim(IVZ,k,j,i) = 0.;
          //     prim(IPR,k,j,i) = pfloor;
            
              
              
          // }


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


              Real dx_bh2_dt,dy_bh2_dt,dz_bh2_dt;



              get_bh_velocity(t,&dx_bh2_dt,&dy_bh2_dt,&dz_bh2_dt);



              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(t,u0,u1,u2,u3, &u0prime,&u1prime,&u2prime,&u3prime);
              // Real u0prime = (u0 + dx_bh2_dt * u1 + dy_bh2_dt * u2 + dz_bh2_dt * u3);
              // Real u1prime = (u1 + dx_bh2_dt * u0);
              // Real u2prime = (u2 + dy_bh2_dt * u0);
              // Real u3prime = (u3 + dz_bh2_dt * u0);



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



void NobleCooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){



  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];


  // // Go through all cells
  // for (int k = ks; k <= ke; ++k) {
  //   for (int j = js; j <= je; ++j) {
  //     pcoord->CellMetric(k, j, is, ie, g, gi);

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real radius, theta,phi;
        GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &radius,
                                         &theta, &phi);
        Real ug = prim(IPR,k,j,i)/(gamma_adi-1.0);
        // Real kappa = prim(IPR,k,j,i)/ std::pow(prim(IDN,k,j,i),gamma_adi);

        // Real radius = std::sqrt( SQR( pmb->pcoord->x1v(i) ) + SQR( pmb->pcoord->x2v(j) ) + SQR( pmb->pcoord->x3v(k) ) );
        // Real v_kep = std::sqrt((1.0 + q)/radius);

        // See Teixeira+ 2014 https://iopscience.iop.org/article/10.1088/0004-637X/796/2/103

        Real Omega = 1.0/( std::pow(radius,1.5) + a);
        // Real t_cool = 2.0 * PI * radius/v_kep;  //orbital time 

        // Real H_over_r_target = 0.02;
        Real Target_Temperature = PI/2.0 * SQR( H_over_r_target * radius * Omega);

        Real Y = prim(IPR,k,j,i)/prim(IDN,k,j,i)/Target_Temperature;
        // Real kappa_0 = 0.01;
        // Real delta_kappa = kappa-kappa_0;

        // if (Y>2.0) Y = 2.0;
        Real L_cool = Omega * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );

        if (L_cool > ug/dt * 0.01) L_cool = ug/dt * 0.01;

        // Real L_cool = ug/t_cool * std::sqrt( delta_kappa/kappa_0 + std::fabs(delta_kappa/kappa_0)  );

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


       // Real bsq = 0.0;
       // if (MAGNETIC_FIELDS_ENABLED){
       //      // Calculate 4-magnetic field
       //    Real bb1 = bcc(IB1,k,j,i);
       //    Real bb2 = bcc(IB2,k,j,i);
       //    Real bb3 = bcc(IB3,k,j,i);
       //    Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
       //            + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
       //            + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
       //            + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
       //    Real b1 = (bb1 + b0 * u1) / u0;
       //    Real b2 = (bb2 + b0 * u2) / u0;
       //    Real b3 = (bb3 + b0 * u3) / u0;
       //    Real b_0, b_1, b_2, b_3;
       //    pmb->pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

       //    // Calculate magnetic pressure
       //    bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

       //  }

        // Real Be = - ( prim(IDN,k,j,i) + ug + prim(IPR,k,j,i) + bsq) * u_0 -1.0; 


        // Do not include bsq in enthalpy
        Real Be = - ( 1.0 + ug/prim(IDN,k,j,i) + prim(IPR,k,j,i)/prim(IDN,k,j,i) ) * u_0 -1.0; 



        if (Be>0) L_cool = 0.0;

        // Calculate Boyer-Lindquist coordinates of cell
        rh = ( m + std::sqrt(SQR(m)-SQR(a)) );
        if (radius < rh) L_cool = 0.0;

        Real xprime,yprime,zprime,rprime,Rprime;
        get_prime_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), time, &xprime,&yprime, &zprime, &rprime,&Rprime);
        Real rhprime = ( q + std::sqrt(SQR(q)-SQR(aprime)) );

        if (rprime < rhprime) L_cool = 0.0;


        cons(IEN,k,j,i) += -dt * L_cool * u_0; 
        cons(IM1,k,j,i) += -dt * L_cool * u_1;
        cons(IM2,k,j,i) += -dt * L_cool * u_2;
        cons(IM3,k,j,i) += -dt * L_cool * u_3;

      }
    }
  }


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
        user_out_var(0,k,j,i) = gamma;

        // Calculate 4-velocity
        Real alpha = std::sqrt(-1.0/gi(I00,i));
        Real u0 = gamma/alpha;
        Real u1 = uu1 - alpha * gamma * gi(I01,i);
        Real u2 = uu2 - alpha * gamma * gi(I02,i);
        Real u3 = uu3 - alpha * gamma * gi(I03,i);
        Real u_0, u_1, u_2, u_3;

        user_out_var(1,k,j,i) = u0;
        user_out_var(2,k,j,i) = u1;
        user_out_var(3,k,j,i) = u2;
        user_out_var(4,k,j,i) = u3;
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

  Real xbh_ = 0.0;
  Real ybh_ = r_bh2 * std::sin(Omega_bh2 * (t-t0));
  Real zbh_ = r_bh2 * std::cos(Omega_bh2 * (t-t0));

  // *xbh = 0.0;
  // *ybh = r_bh2 * std::sin(Omega_bh2 * (t-t0));
  // *zbh = r_bh2 * std::cos(Omega_bh2 * (t-t0));

  *xbh = std::sin(orbit_inclination) * zbh_;
  *ybh = ybh_;
  *zbh = std::cos(orbit_inclination) * zbh_;


  // *zbh = 0.0;
  // *xbh = r_bh2 * std::sin(Omega_bh2 * (t-t0));
  // *ybh = r_bh2 * std::cos(Omega_bh2 * (t-t0));


  // *zbh = 0.0;
  // *xbh = r_bh2 * std::cos(Omega_bh2 * (t-t0));
  // *ybh = r_bh2 * std::sin(Omega_bh2 * (t-t0));

}
void get_bh_velocity(Real t, Real *vxbh, Real *vybh, Real *vzbh){

  Real vxbh_ = 0.0;
  Real vybh_ =  Omega_bh2 * r_bh2 * std::cos(Omega_bh2 * (t-t0));
  Real vzbh_ = -Omega_bh2 * r_bh2 * std::sin(Omega_bh2 * (t-t0));

  // *vxbh = 0.0;
  // *vybh = Omega_bh2 * r_bh2 * std::cos(Omega_bh2 * (t-t0));
  // *vzbh = -Omega_bh2 * r_bh2 * std::sin(Omega_bh2 * (t-t0));


  *vxbh = std::sin(orbit_inclination) * vzbh_;
  *vybh = vybh_;
  *vzbh = std::cos(orbit_inclination) * vzbh_;
  // *zbh = 0.0;
  // *xbh = r_bh2 * std::sin(Omega_bh2 * (t-t0));
  // *ybh = r_bh2 * std::cos(Omega_bh2 * (t-t0));


  // *zbh = 0.0;
  // *xbh = r_bh2 * std::cos(Omega_bh2 * (t-t0));
  // *ybh = r_bh2 * std::sin(Omega_bh2 * (t-t0));

}
void get_bh_acceleration(Real t, Real *axbh, Real *aybh, Real *azbh){


  Real axbh_ = 0.0;
  Real aybh_ = -SQR(Omega_bh2) * r_bh2 * std::sin(Omega_bh2 * (t-t0));
  Real azbh_ = -SQR(Omega_bh2) * r_bh2 * std::cos(Omega_bh2 * (t-t0));

  // *axbh = 0.0;
  // *aybh = -SQR(Omega_bh2) * r_bh2 * std::sin(Omega_bh2 * (t-t0));
  // *azbh = -SQR(Omega_bh2) * r_bh2 * std::cos(Omega_bh2 * (t-t0));

  *axbh = std::sin(orbit_inclination) * azbh_;
  *aybh = aybh_;
  *azbh = std::cos(orbit_inclination) * azbh_;

}

void get_prime_coords(Real x, Real y, Real z, Real t, Real *xprime, Real *yprime, Real *zprime, Real *rprime, Real *Rprime){

  Real xbh,ybh,zbh;
  Real vxbh,vybh,vzbh;
  get_bh_position(t,&xbh,&ybh,&zbh);
  get_bh_velocity(t,&vxbh,&vybh,&vzbh);



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

  if (std::fabs(*zprime)<SMALL) *zprime= SMALL;
  *Rprime = std::sqrt(SQR(*xprime) + SQR(*yprime) + SQR(*zprime));
  *rprime = SQR(*Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(*Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(*zprime) );
  *rprime = std::sqrt(*rprime/2.0);

  return;

}

//From BHframe to lab frame

void BoostVector(Real t,Real a0, Real a1, Real a2, Real a3, Real *pa0, Real *pa1, Real *pa2, Real *pa3){


  Real xbh,ybh,zbh;
  Real vxbh,vybh,vzbh;
  get_bh_position(t,&xbh,&ybh,&zbh);
  get_bh_velocity(t,&vxbh,&vybh,&vzbh);



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

void cks_metric(Real x1, Real x2, Real x3,AthenaArray<Real> &g){
    // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a_spin = a; //-a;

  if (std::fabs(z)<SMALL) z= SMALL;

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
// void delta_cks_metric(ParameterInput *pin,Real t, Real x1, Real x2, Real x3,AthenaArray<Real> &delta_g){
//   Real q = pin->GetOrAddReal("problem", "q", 0.1);
//   Real aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check


//  // Real t = 10000;
//     // Position of black hole

//   Real x = x1;
//   Real y = x2;
//   Real z = x3;

//   if (std::fabs(z)<SMALL) z= SMALL;

//   if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
//     x=  0.1;
//     y = 0.1;
//     z = 0.1;
//   }

//   Real r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);
//   Real v_bh2 = 1.0/std::sqrt(r_bh2);
//   Real Omega_bh2 = v_bh2/r_bh2;
//   Real x_bh2 = 0.0;
//   Real y_bh2 = r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
//   Real z_bh2 = r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);

//   Real xprime = x - x_bh2;
//   Real yprime = y - y_bh2;
//   Real zprime = z - z_bh2;


// //velocity of the second black hole.  For non-circular orbit need to compute velocities in cartesian coordinates

//   Real dx_bh2_dt = 0.0;
//   Real dy_bh2_dt =  2.0*PI*Omega_bh2 * r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);
//   Real dz_bh2_dt = -2.0*PI*Omega_bh2 * r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
//   if (std::fabs(zprime)<SMALL) zprime= SMALL;
//   Real Rprime = std::sqrt(SQR(xprime) + SQR(yprime) + SQR(zprime));
//   Real rprime = SQR(Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(zprime) );
//   rprime = std::sqrt(rprime/2.0);



// /// prevent metric from getting nan sqrt(-gdet)
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


void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{


  a = pin->GetReal("coord", "a");
  m = pin->GetReal("coord", "m");

  //////////////Perturber Black Hole//////////////////

  q = pin->GetOrAddReal("problem", "q", 1.0);
  aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check
  r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);
  t0 = pin->GetOrAddReal("problem","t0", 1e4);

  Binary_BH_Metric(t,x1,x2,x3,g,g_inv,dg_dx1,dg_dx2,dg_dx3,dg_dt,true);

  return;

}

#define DEL 1e-7
void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt,bool take_derivatives)
{


  // if  (Globals::my_rank == 0) fprintf(stderr,"Metric time in pgen file (GLOBAL RANK): %g \n", t);
  // else fprintf(stderr,"Metric time in pgen file (RANK %d): %g \n", Globals::my_rank,t);
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a_spin = a;

  // Real ax,ay,az;
  // Real a_cross_x[3];


  // a_cross_x[0] = ay * z - az * y;
  // a_cross_x[1] = az * x - ax * z;
  // a_cross_x[2] = ax * z - ay * x;

  // Real a_dot_x = ax * x + ay * y + az * z;

  // rsq_p_as = SQR(r) + SQR(a);

  // l_upper[1] = (r * x - a_cross_x[0] + a_dot_x * ax/r)/(rsq_p_asq);
  // l_upper[2] = (r * y - a_cross_x[1] + a_dot_x * ay/r)/(rsq_p_asq);
  // l_upper[3] = (r * z - a_cross_x[2] + a_dot_x * az/r)/(rsq_p_asq);

  if ((std::fabs(z)<SMALL) && (z>=0)) z =  SMALL;
  if ((std::fabs(z)<SMALL) && (z <0)) z = -SMALL;


  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x = 0.1;
    y = 0.1;
    z = 0.1;
  }

  Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = std::sqrt(r/2.0);


/// prevent metric from gettin nan sqrt(-gdet)
  Real th  = std::acos(z/r);
  Real phi = std::atan2( (r*y-a*x)/(SQR(r) + SQR(a) ), 
                              (a*y+r*x)/(SQR(r) + SQR(a) )  );
  rh =  ( m + std::sqrt(SQR(m)-SQR(a)) );
  if (r<rh/2.0) {
    r = rh/2.0;
    x = r * std::cos(phi)*std::sin(th) - a * std::sin(phi)*std::sin(th);
    y = r * std::sin(phi)*std::sin(th) + a * std::cos(phi)*std::sin(th);
    z = r * std::cos(th);
  }



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



  //////////////Perturber Black Hole//////////////////


  Real v_bh2 = 1.0/std::sqrt(r_bh2);
  Omega_bh2 = v_bh2/r_bh2;

  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);


  Real dx_bh2_dt,dy_bh2_dt,dz_bh2_dt;
  get_bh_velocity(t,&dx_bh2_dt,&dy_bh2_dt,&dz_bh2_dt);

  Real ay_bh2,az_bh2,ax_bh2;
  get_bh_acceleration(t,&ax_bh2,&ay_bh2,&az_bh2);

  // Real ax_bh2 = 0.0; 
  // Real ay_bh2 = -SQR(Omega_bh2) * r_bh2 * std::sin(Omega_bh2 * (t-t0));
  // Real az_bh2 = -SQR(Omega_bh2) * r_bh2 * std::cos(Omega_bh2 * (t-t0));

  // Real dz_bh2_dt = 0.0;
  // Real dx_bh2_dt =  Omega_bh2 * r_bh2 * std::cos(Omega_bh2 * (t-t0));
  // Real dy_bh2_dt = -Omega_bh2 * r_bh2 * std::sin(Omega_bh2 * (t-t0));


  // Real dz_bh2_dt = 0.0;
  // Real dx_bh2_dt = -Omega_bh2 * r_bh2 * std::sin(Omega_bh2 * (t-t0));
  // Real dy_bh2_dt = Omega_bh2 * r_bh2 * std::cos(Omega_bh2 * (t-t0));


/// prevent metric from gettin nan sqrt(-gdet)
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


  //First calculated all quantities in BH rest (primed) frame

  Real l_lowerprime[4],l_upperprime[4];
  Real l_lowerprime_transformed[4];
  AthenaArray<Real> Lambda,dLambda_dt;

  Lambda.NewAthenaArray(NMETRIC);
  dLambda_dt.NewAthenaArray(NMETRIC);

  Real fprime = q *  2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));
  l_upperprime[0] = -1.0;
  l_upperprime[1] = (rprime*xprime + aprime*yprime)/( SQR(rprime) + SQR(aprime) );
  l_upperprime[2] = (rprime*yprime - aprime*xprime)/( SQR(rprime) + SQR(aprime) );
  l_upperprime[3] = zprime/rprime;

  l_lowerprime[0] = 1.0;
  l_lowerprime[1] = l_upperprime[1];
  l_lowerprime[2] = l_upperprime[2];
  l_lowerprime[3] = l_upperprime[3];

  //Terms for the boost //

  Real vsq = SQR(dx_bh2_dt) + SQR(dy_bh2_dt) + SQR(dz_bh2_dt);
  Real beta_mag = std::sqrt(vsq);
  Real Lorentz = std::sqrt(1.0/(1.0 - vsq));
  ///Real Lorentz = 1.0;
  Real nx = dx_bh2_dt/beta_mag;
  Real ny = dy_bh2_dt/beta_mag;
  Real nz = dz_bh2_dt/beta_mag;


  Real dLorentz_dt = std::pow(Lorentz,3.0) * (dx_bh2_dt*ax_bh2 + dy_bh2_dt*ay_bh2 + dz_bh2_dt*az_bh2);

  // dbeta_dt = nx*ax + ny*ay + nz*az

  Real dnx_dt = ax_bh2/beta_mag - nx/beta_mag * (nx*ax_bh2+ny*ay_bh2+nz*az_bh2);
  Real dny_dt = ay_bh2/beta_mag - ny/beta_mag * (nx*ax_bh2+ny*ay_bh2+nz*az_bh2);
  Real dnz_dt = az_bh2/beta_mag - nz/beta_mag * (nx*ax_bh2+ny*ay_bh2+nz*az_bh2);


  // This is the inverse transformation since l_mu is lowered.  This 
  // takes a lowered vector from BH frame to lab frame.   
  Lambda(I00) =  Lorentz;
  Lambda(I01) = -Lorentz * dx_bh2_dt;
  Lambda(I02) = -Lorentz * dy_bh2_dt;
  Lambda(I03) = -Lorentz * dz_bh2_dt;
  Lambda(I11) = ( 1.0 + (Lorentz - 1.0) * nx * nx );
  Lambda(I12) = (       (Lorentz - 1.0) * nx * ny ); 
  Lambda(I13) = (       (Lorentz - 1.0) * nx * nz );
  Lambda(I22) = ( 1.0 + (Lorentz - 1.0) * ny * ny ); 
  Lambda(I23) = (       (Lorentz - 1.0) * ny * nz );
  Lambda(I33) = ( 1.0 + (Lorentz - 1.0) * nz * nz );

  // Derivative with respect to time of Lambda. Used for taking derivative of metric

  dLambda_dt(I00) = dLorentz_dt;
  dLambda_dt(I01) = -dx_bh2_dt*dLorentz_dt - Lorentz*ax_bh2;
  dLambda_dt(I02) = -dy_bh2_dt*dLorentz_dt - Lorentz*ay_bh2;
  dLambda_dt(I03) = -dz_bh2_dt*dLorentz_dt - Lorentz*az_bh2;
  dLambda_dt(I11) = dLorentz_dt * nx * nx + (1.0 + (Lorentz - 1.0)) * ( dnx_dt * nx + nx * dnx_dt);
  dLambda_dt(I12) = dLorentz_dt * nx * ny + (      (Lorentz - 1.0)) * ( dnx_dt * ny + nx * dny_dt);
  dLambda_dt(I13) = dLorentz_dt * nx * nz + (      (Lorentz - 1.0)) * ( dnx_dt * nz + nx * dnz_dt);
  dLambda_dt(I22) = dLorentz_dt * ny * ny + (1.0 + (Lorentz - 1.0)) * ( dny_dt * ny + ny * dny_dt);
  dLambda_dt(I23) = dLorentz_dt * ny * nz + (      (Lorentz - 1.0)) * ( dny_dt * nz + ny * dnz_dt);
  dLambda_dt(I33) = dLorentz_dt * nz * nz + (1.0 + (Lorentz - 1.0)) * ( dnz_dt * nz + nz * dnz_dt);



  // Real l0 = l_lowerprime[0];
  // Real l1 = l_lowerprime[1];
  // Real l2 = l_lowerprime[2];
  // Real l3 = l_lowerprime[3];

  // l_lowerprime[0] = Lorentz * (l0 - dx_bh2_dt * l1 - dy_bh2_dt * l2 - dz_bh2_dt * l3);
  // l_lowerprime[3] = Lorentz * (l3 - v_bh2 * l0);

  //These assuem gamma = 1.  Much more complicated if not

  // l_lowerprime[0] = (l0 - dx_bh2_dt * l1 - dy_bh2_dt * l2 - dz_bh2_dt * l3);
  // l_lowerprime[1] = (l1 - dx_bh2_dt * l0);
  // l_lowerprime[2] = (l2 - dy_bh2_dt * l0);
  // l_lowerprime[3] = (l3 - dz_bh2_dt * l0);



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

  // Real det_test = Determinant(g);

  // if (std::isnan( std::sqrt(-det_test))) {
  //   fprintf(stderr,"NAN determinant in metric!! Det: %g \n xyz: %g %g %g \n r: %g \n",det_test,x,y,z,r);
  //   exit(0);
  // }

    // Add Boost terms by transforming from primed frame to central BH frame (for second part of metric only)

  // g(I00) += -2.0 * ( dx_bh2_dt * fprime * l_lowerprime[0]*l_lowerprime[1] 
  //                  + dy_bh2_dt * fprime * l_lowerprime[0]*l_lowerprime[2] 
  //                  + dz_bh2_dt * fprime * l_lowerprime[0]*l_lowerprime[3])
  //           + SQR(dx_bh2_dt) * fprime * l_lowerprime[1]*l_lowerprime[1]
  //           + SQR(dy_bh2_dt) * fprime * l_lowerprime[2]*l_lowerprime[2] 
  //           + SQR(dz_bh2_dt) * fprime * l_lowerprime[3]*l_lowerprime[3];
  // g(I01) += - dx_bh2_dt * fprime * l_lowerprime[1]*l_lowerprime[1];
  // g(I02) += - dy_bh2_dt * fprime * l_lowerprime[2]*l_lowerprime[2];
  // g(I03) += - dz_bh2_dt * fprime * l_lowerprime[3]*l_lowerprime[3];


  bool invertible = gluInvertMatrix(g,g_inv);

  if (invertible==false) {
    fprintf(stderr,"Non-invertible matrix at xyz: %g %g %g\n", x,y,z);
  }



  if (take_derivatives){



      //Compute derivatives of primary BH

      Real sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a);
      Real rsq_p_asq = SQR(r) + SQR(a);

      Real df_dx1 = SQR(f)*x/(2.0*std::pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
      //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
      Real df_dx2 = SQR(f)*y/(2.0*std::pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
      Real df_dx3 = SQR(f)*z/(2.0*std::pow(r,5)) * ( ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(a*r)) ;
      //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
      Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
      // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
      Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ a_spin/( rsq_p_asq );
      Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( (rsq_p_asq) * ( sqrt_term ) ) ;
      Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - a_spin/( rsq_p_asq );
      Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
      Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( (rsq_p_asq) * ( sqrt_term ) );
      Real dl3_dx1 = - x*z/(r) /( sqrt_term );
      Real dl3_dx2 = - y*z/(r) /( sqrt_term );
      Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( rsq_p_asq )/( sqrt_term ) + 1.0/r;

      Real dl0_dx1 = 0.0;
      Real dl0_dx2 = 0.0;
      Real dl0_dx3 = 0.0;

      if (std::isnan(f) || std::isnan(r) || std::isnan(sqrt_term) || std::isnan (df_dx1) || std::isnan(df_dx2)){
        fprintf(stderr,"ISNAN in metric\n x y y: %g %g %g r: %g \n",x,y,z,r);
        exit(0);
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
      dg_dx1(I00) = df_dx1*l_lower[0]*l_lower[0] + f * dl0_dx1 * l_lower[0] + f * l_lower[0] * dl0_dx1;
      dg_dx1(I01) = df_dx1*l_lower[0]*l_lower[1] + f * dl0_dx1 * l_lower[1] + f * l_lower[0] * dl1_dx1;
      dg_dx1(I02) = df_dx1*l_lower[0]*l_lower[2] + f * dl0_dx1 * l_lower[2] + f * l_lower[0] * dl2_dx1;
      dg_dx1(I03) = df_dx1*l_lower[0]*l_lower[3] + f * dl0_dx1 * l_lower[3] + f * l_lower[0] * dl3_dx1;
      dg_dx1(I11) = df_dx1*l_lower[1]*l_lower[1] + f * dl1_dx1 * l_lower[1] + f * l_lower[1] * dl1_dx1;
      dg_dx1(I12) = df_dx1*l_lower[1]*l_lower[2] + f * dl1_dx1 * l_lower[2] + f * l_lower[1] * dl2_dx1;
      dg_dx1(I13) = df_dx1*l_lower[1]*l_lower[3] + f * dl1_dx1 * l_lower[3] + f * l_lower[1] * dl3_dx1;
      dg_dx1(I22) = df_dx1*l_lower[2]*l_lower[2] + f * dl2_dx1 * l_lower[2] + f * l_lower[2] * dl2_dx1;
      dg_dx1(I23) = df_dx1*l_lower[2]*l_lower[3] + f * dl2_dx1 * l_lower[3] + f * l_lower[2] * dl3_dx1;
      dg_dx1(I33) = df_dx1*l_lower[3]*l_lower[3] + f * dl3_dx1 * l_lower[3] + f * l_lower[3] * dl3_dx1;

      // Set y-derivatives of covariant components
      dg_dx2(I00) = df_dx2*l_lower[0]*l_lower[0] + f * dl0_dx2 * l_lower[0] + f * l_lower[0] * dl0_dx2;
      dg_dx2(I01) = df_dx2*l_lower[0]*l_lower[1] + f * dl0_dx2 * l_lower[1] + f * l_lower[0] * dl1_dx2;
      dg_dx2(I02) = df_dx2*l_lower[0]*l_lower[2] + f * dl0_dx2 * l_lower[2] + f * l_lower[0] * dl2_dx2;
      dg_dx2(I03) = df_dx2*l_lower[0]*l_lower[3] + f * dl0_dx2 * l_lower[3] + f * l_lower[0] * dl3_dx2;
      dg_dx2(I11) = df_dx2*l_lower[1]*l_lower[1] + f * dl1_dx2 * l_lower[1] + f * l_lower[1] * dl1_dx2;
      dg_dx2(I12) = df_dx2*l_lower[1]*l_lower[2] + f * dl1_dx2 * l_lower[2] + f * l_lower[1] * dl2_dx2;
      dg_dx2(I13) = df_dx2*l_lower[1]*l_lower[3] + f * dl1_dx2 * l_lower[3] + f * l_lower[1] * dl3_dx2;
      dg_dx2(I22) = df_dx2*l_lower[2]*l_lower[2] + f * dl2_dx2 * l_lower[2] + f * l_lower[2] * dl2_dx2;
      dg_dx2(I23) = df_dx2*l_lower[2]*l_lower[3] + f * dl2_dx2 * l_lower[3] + f * l_lower[2] * dl3_dx2;
      dg_dx2(I33) = df_dx2*l_lower[3]*l_lower[3] + f * dl3_dx2 * l_lower[3] + f * l_lower[3] * dl3_dx2;

      // Set z-derivatives of covariant components
      dg_dx3(I00) = df_dx3*l_lower[0]*l_lower[0] + f * dl0_dx3 * l_lower[0] + f * l_lower[0] * dl0_dx3;
      dg_dx3(I01) = df_dx3*l_lower[0]*l_lower[1] + f * dl0_dx3 * l_lower[1] + f * l_lower[0] * dl1_dx3;
      dg_dx3(I02) = df_dx3*l_lower[0]*l_lower[2] + f * dl0_dx3 * l_lower[2] + f * l_lower[0] * dl2_dx3;
      dg_dx3(I03) = df_dx3*l_lower[0]*l_lower[3] + f * dl0_dx3 * l_lower[3] + f * l_lower[0] * dl3_dx3;
      dg_dx3(I11) = df_dx3*l_lower[1]*l_lower[1] + f * dl1_dx3 * l_lower[1] + f * l_lower[1] * dl1_dx3;
      dg_dx3(I12) = df_dx3*l_lower[1]*l_lower[2] + f * dl1_dx3 * l_lower[2] + f * l_lower[1] * dl2_dx3;
      dg_dx3(I13) = df_dx3*l_lower[1]*l_lower[3] + f * dl1_dx3 * l_lower[3] + f * l_lower[1] * dl3_dx3;
      dg_dx3(I22) = df_dx3*l_lower[2]*l_lower[2] + f * dl2_dx3 * l_lower[2] + f * l_lower[2] * dl2_dx3;
      dg_dx3(I23) = df_dx3*l_lower[2]*l_lower[3] + f * dl2_dx3 * l_lower[3] + f * l_lower[2] * dl3_dx3;
      dg_dx3(I33) = df_dx3*l_lower[3]*l_lower[3] + f * dl3_dx3 * l_lower[3] + f * l_lower[3] * dl3_dx3;



    /////Secondary Black hole/////


      //these are derivatives with respect to xprime (or X Y Z)
      //lab frame is x y z
      //be careful with difference between X and x, Y and y, Z and z
      //Note that these derivative are of l_mu pre boost
      Real dlprime_dX1[4], dlprime_dX2[4], dlprime_dX3[4];


      sqrt_term =  2.0*SQR(rprime)-SQR(Rprime) + SQR(aprime);
      rsq_p_asq = SQR(rprime) + SQR(aprime);

      Real fprime_over_q = 2.0 * SQR(rprime)*rprime / (SQR(SQR(rprime)) + SQR(aprime)*SQR(zprime));


      Real dfprime_dX1 = q * SQR(fprime_over_q)*xprime/(2.0*std::pow(rprime,3)) * 
                          ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
      //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
      Real dfprime_dX2 = q * SQR(fprime_over_q)*yprime/(2.0*std::pow(rprime,3)) * 
                          ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
      Real dfprime_dX3 = q * SQR(fprime_over_q)*zprime/(2.0*std::pow(rprime,5)) * 
                          ( ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(aprime*rprime)) ;
      //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
      dlprime_dX1[1] = xprime*rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + rprime/( rsq_p_asq );
      // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
      dlprime_dX2[1] = yprime*rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ aprime/( rsq_p_asq );
      dlprime_dX3[1] = zprime/rprime * ( SQR(aprime)*xprime - 2.0*aprime*rprime*yprime - SQR(rprime)*xprime )/( (rsq_p_asq) * ( sqrt_term ) ) ;
      dlprime_dX1[2] = xprime*rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - aprime/( rsq_p_asq );
      dlprime_dX2[2] = yprime*rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + rprime/( rsq_p_asq );
      dlprime_dX3[2] = zprime/rprime * ( SQR(aprime)*yprime + 2.0*aprime*rprime*xprime - SQR(rprime)*yprime )/( (rsq_p_asq) * ( sqrt_term ) );
      dlprime_dX1[3] = - xprime*zprime/(rprime) /( sqrt_term );
      dlprime_dX2[3] = - yprime*zprime/(rprime) /( sqrt_term );
      dlprime_dX3[3] = - SQR(zprime)/(SQR(rprime)*rprime) * ( rsq_p_asq )/( sqrt_term ) + 1.0/rprime;


      dlprime_dX1[0] = 0.0;
      dlprime_dX2[0] = 0.0;
      dlprime_dX3[0] = 0.0;


      //Coordinate dependence vectors.  Should be the same as Lambda but 
      //being explicit to be safe
      Real dX1_dx1 = 1.0 + (Lorentz-1.0)*nx*nx ;
      Real dX1_dx2 =       (Lorentz-1.0)*nx*ny ;
      Real dX1_dx3 =       (Lorentz-1.0)*nx*nz ;

      Real dX2_dx1 =       (Lorentz-1.0)*ny*nx ;
      Real dX2_dx2 = 1.0 + (Lorentz-1.0)*ny*ny ;  
      Real dX2_dx3 =       (Lorentz-1.0)*ny*nz ;

      Real dX3_dx1 =       (Lorentz-1.0)*nz*nx ;
      Real dX3_dx2 =       (Lorentz-1.0)*nz*ny ;  
      Real dX3_dx3 = 1.0 + (Lorentz-1.0)*nz*nz ;


      //derivatives of boosted vectors
      Real dlprime_dX1_transformed[4], dlprime_dX2_transformed[4], dlprime_dX3_transformed[4];


      // dlprime_dx1[1] = dl1prime_dX1 * dX1_dx1 + dl1prime_dX2 * dX2_dx1 + dl1prime_dX3 * dX3_dx1;
      // dlprime_dx2[1] = dl1prime_dX1 * dX1_dx2 + dl1prime_dX2 * dX2_dx2 + dl1prime_dX3 * dX3_dx2;
      // dlprime_dx3[1] = dl1prime_dX1 * dX1_dx3 + dl1prime_dX2 * dX2_dx3 + dl1prime_dX3 * dX3_dx3;
      
      // dlprime_dx1[2] = dl2prime_dX1 * dX1_dx1 + dl2prime_dX2 * dX2_dx1 + dl2prime_dX3 * dX3_dx1;
      // dlprime_dx2[2] = dl2prime_dX1 * dX1_dx2 + dl2prime_dX2 * dX2_dx2 + dl2prime_dX3 * dX3_dx2;
      // dlprime_dx3[2] = dl2prime_dX1 * dX1_dx3 + dl2prime_dX2 * dX2_dx3 + dl2prime_dX3 * dX3_dx3;

      // dlprime_dx1[3] = dl3prime_dX1 * dX1_dx1 + dl3prime_dX2 * dX2_dx1 + dl3prime_dX3 * dX3_dx1;
      // dlprime_dx2[3] = dl3prime_dX1 * dX1_dx2 + dl3prime_dX2 * dX2_dx2 + dl3prime_dX3 * dX3_dx2;
      // dlprime_dx3[3] = dl3prime_dX1 * dX1_dx3 + dl3prime_dX2 * dX2_dx3 + dl3prime_dX3 * dX3_dx3;




      //Boost spatial derivative vectors
      //Can do this because Lambda doesn't depend on spatial coordinates
      matrix_multiply_vector_lefthandside(Lambda,dlprime_dX1,dlprime_dX1_transformed);
      matrix_multiply_vector_lefthandside(Lambda,dlprime_dX2,dlprime_dX2_transformed);
      matrix_multiply_vector_lefthandside(Lambda,dlprime_dX3,dlprime_dX3_transformed);

      // Real dl0_dx1_tmp = dl0prime_dx1;
      // Real dl0_dx2_tmp = dl0prime_dx2;
      // Real dl0_dx3_tmp = dl0prime_dx3;

      // Real dl1_dx1_tmp = dl1prime_dx1;
      // Real dl1_dx2_tmp = dl1prime_dx2;
      // Real dl1_dx3_tmp = dl1prime_dx3;

      // Real dl2_dx1_tmp = dl2prime_dx1;
      // Real dl2_dx2_tmp = dl2prime_dx2;
      // Real dl2_dx3_tmp = dl2prime_dx3;

      // Real dl3_dx1_tmp = dl3prime_dx1;
      // Real dl3_dx2_tmp = dl3prime_dx2;
      // Real dl3_dx3_tmp = dl3prime_dx3;





      // l_lowerprime[0] = (l0 - dx_bh2_dt * l1 - dy_bh2_dt * l2 - dz_bh2_dt * l3);
      // l_lowerprime[1] = (l1 - dx_bh2_dt * l0);
      // l_lowerprime[2] = (l2 - dy_bh2_dt * l0);
      // l_lowerprime[3] = (l3 - dz_bh2_dt * l0);


      // BoostLowerVector(dl0prime_dx1,dl1prime_dx1,dl2prime_dx1,dl3prime_dx1,
      //                  dl0prime_dx1,dl1prime_dx1,dl2prime_dx1,dl3prime_dx1);
      // BoostLowerVector(dl0prime_dx2,dl1prime_dx2,dl2prime_dx1,dl3prime_dx2,
      //                  dl0prime_dx2,dl1prime_dx2,dl2prime_dx1,dl3prime_dx2);
      // BoostLowerVector(dl0prime_dx3,dl1prime_dx3,dl2prime_dx1,dl3prime_dx3,
      //                  dl0prime_dx3,dl1prime_dx3,dl2prime_dx1,dl3prime_dx3);


      // dl0prime_dx1 = (dl0_dx1_tmp - dx_bh2_dt * dl1_dx1_tmp - dy_bh2_dt * dl2_dx1_tmp - dz_bh2_dt * dl3_dx1_tmp); 
      // dl0prime_dx2 = (dl0_dx2_tmp - dx_bh2_dt * dl1_dx2_tmp - dy_bh2_dt * dl2_dx2_tmp - dz_bh2_dt * dl3_dx2_tmp); 
      // dl0prime_dx3 = (dl0_dx3_tmp - dx_bh2_dt * dl1_dx3_tmp - dy_bh2_dt * dl2_dx3_tmp - dz_bh2_dt * dl3_dx3_tmp);  


      // dl1prime_dx1 = (dl1_dx1_tmp - dx_bh2_dt * dl0_dx1_tmp); 
      // dl1prime_dx2 = (dl1_dx2_tmp - dx_bh2_dt * dl0_dx2_tmp); 
      // dl1prime_dx3 = (dl1_dx3_tmp - dx_bh2_dt * dl0_dx3_tmp); 


      // dl2prime_dx1 = (dl2_dx1_tmp - dy_bh2_dt * dl0_dx1_tmp); 
      // dl2prime_dx2 = (dl2_dx2_tmp - dy_bh2_dt * dl0_dx2_tmp); 
      // dl2prime_dx3 = (dl2_dx3_tmp - dy_bh2_dt * dl0_dx3_tmp); 

      // dl3prime_dx1 = (dl3_dx1_tmp - dz_bh2_dt * dl0_dx1_tmp); 
      // dl3prime_dx2 = (dl3_dx2_tmp - dz_bh2_dt * dl0_dx2_tmp); 
      // dl3prime_dx3 = (dl3_dx3_tmp - dz_bh2_dt * dl0_dx3_tmp); 


      //partial derivatives in t
      // Real dl0prime_dt = - ax_bh2 * l1 - ay_bh2 * l2 - az_bh2 *l3;
      // Real dl1prime_dt = - ax_bh2 * l0;
      // Real dl2prime_dt = - ay_bh2 * l0;
      // Real dl3prime_dt = - az_bh2 * l0;


      AthenaArray<Real> dgprime_dX1, dgprime_dX2, dgprime_dX3;
      dgprime_dX1.NewAthenaArray(NMETRIC);
      dgprime_dX2.NewAthenaArray(NMETRIC);
      dgprime_dX3.NewAthenaArray(NMETRIC);

      // // // Set x-derivatives of covariant components
      // dgprime_dx1(I00) = dfprime_dx1*l_lowerprime_transformed[0]*l_lowerprime_transformed[0] + fprime * dlprime_dx1_transformed[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_dx1_transformed[0] ;
      // dgprime_dx1(I01) = dfprime_dx1*l_lowerprime_transformed[0]*l_lowerprime_transformed[1] + fprime * dlprime_dx1_transformed[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_dx1_transformed[1];
      // dgprime_dx1(I02) = dfprime_dx1*l_lowerprime_transformed[0]*l_lowerprime_transformed[2] + fprime * dlprime_dx1_transformed[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_dx1_transformed[2];
      // dgprime_dx1(I03) = dfprime_dx1*l_lowerprime_transformed[0]*l_lowerprime_transformed[3] + fprime * dlprime_dx1_transformed[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_dx1_transformed[3];
      // dgprime_dx1(I11) = dfprime_dx1*l_lowerprime_transformed[1]*l_lowerprime_transformed[1] + fprime * dlprime_dx1_transformed[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_dx1_transformed[1];
      // dgprime_dx1(I12) = dfprime_dx1*l_lowerprime_transformed[1]*l_lowerprime_transformed[2] + fprime * dlprime_dx1_transformed[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_dx1_transformed[2];
      // dgprime_dx1(I13) = dfprime_dx1*l_lowerprime_transformed[1]*l_lowerprime_transformed[3] + fprime * dlprime_dx1_transformed[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_dx1_transformed[3];
      // dgprime_dx1(I22) = dfprime_dx1*l_lowerprime_transformed[2]*l_lowerprime_transformed[2] + fprime * dlprime_dx1_transformed[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_dx1_transformed[2];
      // dgprime_dx1(I23) = dfprime_dx1*l_lowerprime_transformed[2]*l_lowerprime_transformed[3] + fprime * dlprime_dx1_transformed[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_dx1_transformed[3];
      // dgprime_dx1(I33) = dfprime_dx1*l_lowerprime_transformed[3]*l_lowerprime_transformed[3] + fprime * dlprime_dx1_transformed[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_dx1_transformed[3];

      // // Set y-derivatives of covariant components
      // dgprime_dx2(I00) = dfprime_dx2*l_lowerprime_transformed[0]*l_lowerprime_transformed[0] + fprime * dlprime_dx2_transformed[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_dx2_transformed[0];
      // dgprime_dx2(I01) = dfprime_dx2*l_lowerprime_transformed[0]*l_lowerprime_transformed[1] + fprime * dlprime_dx2_transformed[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_dx2_transformed[1];
      // dgprime_dx2(I02) = dfprime_dx2*l_lowerprime_transformed[0]*l_lowerprime_transformed[2] + fprime * dlprime_dx2_transformed[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_dx2_transformed[2];
      // dgprime_dx2(I03) = dfprime_dx2*l_lowerprime_transformed[0]*l_lowerprime_transformed[3] + fprime * dlprime_dx2_transformed[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_dx2_transformed[3];
      // dgprime_dx2(I11) = dfprime_dx2*l_lowerprime_transformed[1]*l_lowerprime_transformed[1] + fprime * dlprime_dx2_transformed[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_dx2_transformed[1];
      // dgprime_dx2(I12) = dfprime_dx2*l_lowerprime_transformed[1]*l_lowerprime_transformed[2] + fprime * dlprime_dx2_transformed[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_dx2_transformed[2];
      // dgprime_dx2(I13) = dfprime_dx2*l_lowerprime_transformed[1]*l_lowerprime_transformed[3] + fprime * dlprime_dx2_transformed[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_dx2_transformed[3];
      // dgprime_dx2(I22) = dfprime_dx2*l_lowerprime_transformed[2]*l_lowerprime_transformed[2] + fprime * dlprime_dx2_transformed[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_dx2_transformed[2];
      // dgprime_dx2(I23) = dfprime_dx2*l_lowerprime_transformed[2]*l_lowerprime_transformed[3] + fprime * dlprime_dx2_transformed[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_dx2_transformed[3];
      // dgprime_dx2(I33) = dfprime_dx2*l_lowerprime_transformed[3]*l_lowerprime_transformed[3] + fprime * dlprime_dx2_transformed[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_dx2_transformed[3];

      // // Set z-derivatives of covariant components
      // dgprime_dx3(I00) = dfprime_dx3*l_lowerprime_transformed[0]*l_lowerprime_transformed[0] + fprime * dlprime_dx3_transformed[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_dx3_transformed[0];
      // dgprime_dx3(I01) = dfprime_dx3*l_lowerprime_transformed[0]*l_lowerprime_transformed[1] + fprime * dlprime_dx3_transformed[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_dx3_transformed[1];
      // dgprime_dx3(I02) = dfprime_dx3*l_lowerprime_transformed[0]*l_lowerprime_transformed[2] + fprime * dlprime_dx3_transformed[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_dx3_transformed[2];
      // dgprime_dx3(I03) = dfprime_dx3*l_lowerprime_transformed[0]*l_lowerprime_transformed[3] + fprime * dlprime_dx3_transformed[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_dx3_transformed[3];
      // dgprime_dx3(I11) = dfprime_dx3*l_lowerprime_transformed[1]*l_lowerprime_transformed[1] + fprime * dlprime_dx3_transformed[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_dx3_transformed[1];
      // dgprime_dx3(I12) = dfprime_dx3*l_lowerprime_transformed[1]*l_lowerprime_transformed[2] + fprime * dlprime_dx3_transformed[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_dx3_transformed[2];
      // dgprime_dx3(I13) = dfprime_dx3*l_lowerprime_transformed[1]*l_lowerprime_transformed[3] + fprime * dlprime_dx3_transformed[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_dx3_transformed[3];
      // dgprime_dx3(I22) = dfprime_dx3*l_lowerprime_transformed[2]*l_lowerprime_transformed[2] + fprime * dlprime_dx3_transformed[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_dx3_transformed[2];
      // dgprime_dx3(I23) = dfprime_dx3*l_lowerprime_transformed[2]*l_lowerprime_transformed[3] + fprime * dlprime_dx3_transformed[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_dx3_transformed[3];
      // dgprime_dx3(I33) = dfprime_dx3*l_lowerprime_transformed[3]*l_lowerprime_transformed[3] + fprime * dlprime_dx3_transformed[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_dx3_transformed[3];


      // Derivatives in primed coordinates (black hole frame)
      // // Set x-derivatives of covariant components
      dgprime_dX1(I00) = dfprime_dX1*l_lowerprime_transformed[0]*l_lowerprime_transformed[0] + fprime * dlprime_dX1_transformed[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_dX1_transformed[0];
      dgprime_dX1(I01) = dfprime_dX1*l_lowerprime_transformed[0]*l_lowerprime_transformed[1] + fprime * dlprime_dX1_transformed[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_dX1_transformed[1];
      dgprime_dX1(I02) = dfprime_dX1*l_lowerprime_transformed[0]*l_lowerprime_transformed[2] + fprime * dlprime_dX1_transformed[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_dX1_transformed[2];
      dgprime_dX1(I03) = dfprime_dX1*l_lowerprime_transformed[0]*l_lowerprime_transformed[3] + fprime * dlprime_dX1_transformed[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_dX1_transformed[3];
      dgprime_dX1(I11) = dfprime_dX1*l_lowerprime_transformed[1]*l_lowerprime_transformed[1] + fprime * dlprime_dX1_transformed[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_dX1_transformed[1];
      dgprime_dX1(I12) = dfprime_dX1*l_lowerprime_transformed[1]*l_lowerprime_transformed[2] + fprime * dlprime_dX1_transformed[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_dX1_transformed[2];
      dgprime_dX1(I13) = dfprime_dX1*l_lowerprime_transformed[1]*l_lowerprime_transformed[3] + fprime * dlprime_dX1_transformed[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_dX1_transformed[3];
      dgprime_dX1(I22) = dfprime_dX1*l_lowerprime_transformed[2]*l_lowerprime_transformed[2] + fprime * dlprime_dX1_transformed[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_dX1_transformed[2];
      dgprime_dX1(I23) = dfprime_dX1*l_lowerprime_transformed[2]*l_lowerprime_transformed[3] + fprime * dlprime_dX1_transformed[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_dX1_transformed[3];
      dgprime_dX1(I33) = dfprime_dX1*l_lowerprime_transformed[3]*l_lowerprime_transformed[3] + fprime * dlprime_dX1_transformed[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_dX1_transformed[3];

      // Set y-derivatives of covariant components
      dgprime_dX2(I00) = dfprime_dX2*l_lowerprime_transformed[0]*l_lowerprime_transformed[0] + fprime * dlprime_dX2_transformed[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_dX2_transformed[0];
      dgprime_dX2(I01) = dfprime_dX2*l_lowerprime_transformed[0]*l_lowerprime_transformed[1] + fprime * dlprime_dX2_transformed[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_dX2_transformed[1];
      dgprime_dX2(I02) = dfprime_dX2*l_lowerprime_transformed[0]*l_lowerprime_transformed[2] + fprime * dlprime_dX2_transformed[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_dX2_transformed[2];
      dgprime_dX2(I03) = dfprime_dX2*l_lowerprime_transformed[0]*l_lowerprime_transformed[3] + fprime * dlprime_dX2_transformed[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_dX2_transformed[3];
      dgprime_dX2(I11) = dfprime_dX2*l_lowerprime_transformed[1]*l_lowerprime_transformed[1] + fprime * dlprime_dX2_transformed[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_dX2_transformed[1];
      dgprime_dX2(I12) = dfprime_dX2*l_lowerprime_transformed[1]*l_lowerprime_transformed[2] + fprime * dlprime_dX2_transformed[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_dX2_transformed[2];
      dgprime_dX2(I13) = dfprime_dX2*l_lowerprime_transformed[1]*l_lowerprime_transformed[3] + fprime * dlprime_dX2_transformed[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_dX2_transformed[3];
      dgprime_dX2(I22) = dfprime_dX2*l_lowerprime_transformed[2]*l_lowerprime_transformed[2] + fprime * dlprime_dX2_transformed[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_dX2_transformed[2];
      dgprime_dX2(I23) = dfprime_dX2*l_lowerprime_transformed[2]*l_lowerprime_transformed[3] + fprime * dlprime_dX2_transformed[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_dX2_transformed[3];
      dgprime_dX2(I33) = dfprime_dX2*l_lowerprime_transformed[3]*l_lowerprime_transformed[3] + fprime * dlprime_dX2_transformed[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_dX2_transformed[3];

      // Set z-derivatives of covariant components
      dgprime_dX3(I00) = dfprime_dX3*l_lowerprime_transformed[0]*l_lowerprime_transformed[0] + fprime * dlprime_dX3_transformed[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_dX3_transformed[0];
      dgprime_dX3(I01) = dfprime_dX3*l_lowerprime_transformed[0]*l_lowerprime_transformed[1] + fprime * dlprime_dX3_transformed[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_dX3_transformed[1];
      dgprime_dX3(I02) = dfprime_dX3*l_lowerprime_transformed[0]*l_lowerprime_transformed[2] + fprime * dlprime_dX3_transformed[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_dX3_transformed[2];
      dgprime_dX3(I03) = dfprime_dX3*l_lowerprime_transformed[0]*l_lowerprime_transformed[3] + fprime * dlprime_dX3_transformed[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_dX3_transformed[3];
      dgprime_dX3(I11) = dfprime_dX3*l_lowerprime_transformed[1]*l_lowerprime_transformed[1] + fprime * dlprime_dX3_transformed[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_dX3_transformed[1];
      dgprime_dX3(I12) = dfprime_dX3*l_lowerprime_transformed[1]*l_lowerprime_transformed[2] + fprime * dlprime_dX3_transformed[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_dX3_transformed[2];
      dgprime_dX3(I13) = dfprime_dX3*l_lowerprime_transformed[1]*l_lowerprime_transformed[3] + fprime * dlprime_dX3_transformed[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_dX3_transformed[3];
      dgprime_dX3(I22) = dfprime_dX3*l_lowerprime_transformed[2]*l_lowerprime_transformed[2] + fprime * dlprime_dX3_transformed[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_dX3_transformed[2];
      dgprime_dX3(I23) = dfprime_dX3*l_lowerprime_transformed[2]*l_lowerprime_transformed[3] + fprime * dlprime_dX3_transformed[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_dX3_transformed[3];
      dgprime_dX3(I33) = dfprime_dX3*l_lowerprime_transformed[3]*l_lowerprime_transformed[3] + fprime * dlprime_dX3_transformed[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_dX3_transformed[3];




      //Convert derivatives from d/dX to d/dx
      //Could be a dg/dT * dT/dx term but no dependence on T in g

      // // Set x-derivatives of covariant components
      dg_dx1(I00) += dgprime_dX1(I00) * dX1_dx1 + dgprime_dX2(I00) * dX2_dx1 + dgprime_dX3(I00) * dX3_dx1 ;
      dg_dx1(I01) += dgprime_dX1(I01) * dX1_dx1 + dgprime_dX2(I01) * dX2_dx1 + dgprime_dX3(I01) * dX3_dx1 ;
      dg_dx1(I02) += dgprime_dX1(I02) * dX1_dx1 + dgprime_dX2(I02) * dX2_dx1 + dgprime_dX3(I02) * dX3_dx1 ;
      dg_dx1(I03) += dgprime_dX1(I03) * dX1_dx1 + dgprime_dX2(I03) * dX2_dx1 + dgprime_dX3(I03) * dX3_dx1 ;
      dg_dx1(I11) += dgprime_dX1(I11) * dX1_dx1 + dgprime_dX2(I11) * dX2_dx1 + dgprime_dX3(I11) * dX3_dx1 ;
      dg_dx1(I12) += dgprime_dX1(I12) * dX1_dx1 + dgprime_dX2(I12) * dX2_dx1 + dgprime_dX3(I12) * dX3_dx1 ;
      dg_dx1(I13) += dgprime_dX1(I13) * dX1_dx1 + dgprime_dX2(I13) * dX2_dx1 + dgprime_dX3(I13) * dX3_dx1 ;
      dg_dx1(I22) += dgprime_dX1(I22) * dX1_dx1 + dgprime_dX2(I22) * dX2_dx1 + dgprime_dX3(I22) * dX3_dx1 ;
      dg_dx1(I23) += dgprime_dX1(I23) * dX1_dx1 + dgprime_dX2(I23) * dX2_dx1 + dgprime_dX3(I23) * dX3_dx1 ;
      dg_dx1(I33) += dgprime_dX1(I33) * dX1_dx1 + dgprime_dX2(I33) * dX2_dx1 + dgprime_dX3(I33) * dX3_dx1 ;

      // Set y-derivatives of covariant components
      dg_dx2(I00) += dgprime_dX1(I00) * dX1_dx2 + dgprime_dX2(I00) * dX2_dx2 + dgprime_dX3(I00) * dX3_dx2 ;
      dg_dx2(I01) += dgprime_dX1(I01) * dX1_dx2 + dgprime_dX2(I01) * dX2_dx2 + dgprime_dX3(I01) * dX3_dx2 ;
      dg_dx2(I02) += dgprime_dX1(I02) * dX1_dx2 + dgprime_dX2(I02) * dX2_dx2 + dgprime_dX3(I02) * dX3_dx2 ;
      dg_dx2(I03) += dgprime_dX1(I03) * dX1_dx2 + dgprime_dX2(I03) * dX2_dx2 + dgprime_dX3(I03) * dX3_dx2 ;
      dg_dx2(I11) += dgprime_dX1(I11) * dX1_dx2 + dgprime_dX2(I11) * dX2_dx2 + dgprime_dX3(I11) * dX3_dx2 ;
      dg_dx2(I12) += dgprime_dX1(I12) * dX1_dx2 + dgprime_dX2(I12) * dX2_dx2 + dgprime_dX3(I12) * dX3_dx2 ;
      dg_dx2(I13) += dgprime_dX1(I13) * dX1_dx2 + dgprime_dX2(I13) * dX2_dx2 + dgprime_dX3(I13) * dX3_dx2 ;
      dg_dx2(I22) += dgprime_dX1(I22) * dX1_dx2 + dgprime_dX2(I22) * dX2_dx2 + dgprime_dX3(I22) * dX3_dx2 ;
      dg_dx2(I23) += dgprime_dX1(I23) * dX1_dx2 + dgprime_dX2(I23) * dX2_dx2 + dgprime_dX3(I23) * dX3_dx2 ;
      dg_dx2(I33) += dgprime_dX1(I33) * dX1_dx2 + dgprime_dX2(I33) * dX2_dx2 + dgprime_dX3(I33) * dX3_dx2 ;

      // Set z-derivatives of covariant components
      dg_dx3(I00) += dgprime_dX1(I00) * dX1_dx3 + dgprime_dX2(I00) * dX2_dx3 + dgprime_dX3(I00) * dX3_dx3 ;
      dg_dx3(I01) += dgprime_dX1(I01) * dX1_dx3 + dgprime_dX2(I01) * dX2_dx3 + dgprime_dX3(I01) * dX3_dx3 ;
      dg_dx3(I02) += dgprime_dX1(I02) * dX1_dx3 + dgprime_dX2(I02) * dX2_dx3 + dgprime_dX3(I02) * dX3_dx3 ;
      dg_dx3(I03) += dgprime_dX1(I03) * dX1_dx3 + dgprime_dX2(I03) * dX2_dx3 + dgprime_dX3(I03) * dX3_dx3 ;
      dg_dx3(I11) += dgprime_dX1(I11) * dX1_dx3 + dgprime_dX2(I11) * dX2_dx3 + dgprime_dX3(I11) * dX3_dx3 ;
      dg_dx3(I12) += dgprime_dX1(I12) * dX1_dx3 + dgprime_dX2(I12) * dX2_dx3 + dgprime_dX3(I12) * dX3_dx3 ;
      dg_dx3(I13) += dgprime_dX1(I13) * dX1_dx3 + dgprime_dX2(I13) * dX2_dx3 + dgprime_dX3(I13) * dX3_dx3 ;
      dg_dx3(I22) += dgprime_dX1(I22) * dX1_dx3 + dgprime_dX2(I22) * dX2_dx3 + dgprime_dX3(I22) * dX3_dx3 ;
      dg_dx3(I23) += dgprime_dX1(I23) * dX1_dx3 + dgprime_dX2(I23) * dX2_dx3 + dgprime_dX3(I23) * dX3_dx3 ;
      dg_dx3(I33) += dgprime_dX1(I33) * dX1_dx3 + dgprime_dX2(I33) * dX2_dx3 + dgprime_dX3(I33) * dX3_dx3 ;


      Real x_bh2,y_bh2,z_bh2;
      get_bh_position(t,&x_bh2,&y_bh2,&z_bh2);


      //Partial derivatives with respect to T of coordinate relatiosn
      //Note that Lambda as defined in this function is the inverse transformation, so formally this should
      //be the derivative of the inverse of that
      //but only the 0 components are affected by the inverse  (e.g., Lambda(I11) = Lambda_inv(I11))
      Real dX_dt = dLambda_dt(I11) * (x - x_bh2) - Lambda(I11) * dx_bh2_dt + 
                   dLambda_dt(I12) * (y - y_bh2) - Lambda(I12) * dy_bh2_dt + 
                   dLambda_dt(I13) * (z - z_bh2) - Lambda(I13) * dz_bh2_dt ;

      Real dY_dt = dLambda_dt(I12) * (x - x_bh2) - Lambda(I12) * dx_bh2_dt + 
                   dLambda_dt(I22) * (y - y_bh2) - Lambda(I22) * dy_bh2_dt + 
                   dLambda_dt(I23) * (z - z_bh2) - Lambda(I23) * dz_bh2_dt ;

      Real dZ_dt = dLambda_dt(I13) * (x - x_bh2) - Lambda(I13) * dx_bh2_dt + 
                   dLambda_dt(I23) * (y - y_bh2) - Lambda(I23) * dy_bh2_dt + 
                   dLambda_dt(I33) * (z - z_bh2) - Lambda(I33) * dz_bh2_dt ;




      // Set t-derivatives of covariant components
      // d/dt = partial_t + partial_X dX/dt
      // first do latter part
      dg_dt(I00) = (dX_dt * dgprime_dX1(I00) + dY_dt * dgprime_dX2(I00) + dZ_dt * dgprime_dX3(I00) );
      dg_dt(I01) = (dX_dt * dgprime_dX1(I01) + dY_dt * dgprime_dX2(I01) + dZ_dt * dgprime_dX3(I01) );
      dg_dt(I02) = (dX_dt * dgprime_dX1(I02) + dY_dt * dgprime_dX2(I02) + dZ_dt * dgprime_dX3(I02) );
      dg_dt(I03) = (dX_dt * dgprime_dX1(I03) + dY_dt * dgprime_dX2(I03) + dZ_dt * dgprime_dX3(I03) );
      dg_dt(I11) = (dX_dt * dgprime_dX1(I11) + dY_dt * dgprime_dX2(I11) + dZ_dt * dgprime_dX3(I11) );
      dg_dt(I12) = (dX_dt * dgprime_dX1(I12) + dY_dt * dgprime_dX2(I12) + dZ_dt * dgprime_dX3(I12) );
      dg_dt(I13) = (dX_dt * dgprime_dX1(I13) + dY_dt * dgprime_dX2(I13) + dZ_dt * dgprime_dX3(I13) );
      dg_dt(I22) = (dX_dt * dgprime_dX1(I22) + dY_dt * dgprime_dX2(I22) + dZ_dt * dgprime_dX3(I22) );
      dg_dt(I23) = (dX_dt * dgprime_dX1(I23) + dY_dt * dgprime_dX2(I23) + dZ_dt * dgprime_dX3(I23) );
      dg_dt(I33) = (dX_dt * dgprime_dX1(I33) + dY_dt * dgprime_dX2(I33) + dZ_dt * dgprime_dX3(I33) );

      Real dlprime_transformed_dt[4];


      //Since direct t dependence only shows up through Lambda, can take dLambda_dt and use that
      //for transforming vectors
      //Note need to transform lowerprime *before* transformation
      matrix_multiply_vector_lefthandside(dLambda_dt,l_lowerprime,dlprime_transformed_dt);



      dg_dt(I00) += fprime * dlprime_transformed_dt[0] * l_lowerprime_transformed[0] + fprime * l_lowerprime_transformed[0] * dlprime_transformed_dt[0];
      dg_dt(I01) += fprime * dlprime_transformed_dt[0] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[0] * dlprime_transformed_dt[1];;
      dg_dt(I02) += fprime * dlprime_transformed_dt[0] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[0] * dlprime_transformed_dt[2];;
      dg_dt(I03) += fprime * dlprime_transformed_dt[0] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[0] * dlprime_transformed_dt[3];;
      dg_dt(I11) += fprime * dlprime_transformed_dt[1] * l_lowerprime_transformed[1] + fprime * l_lowerprime_transformed[1] * dlprime_transformed_dt[1];;
      dg_dt(I12) += fprime * dlprime_transformed_dt[1] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[1] * dlprime_transformed_dt[2];;
      dg_dt(I13) += fprime * dlprime_transformed_dt[1] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[1] * dlprime_transformed_dt[3];;
      dg_dt(I22) += fprime * dlprime_transformed_dt[2] * l_lowerprime_transformed[2] + fprime * l_lowerprime_transformed[2] * dlprime_transformed_dt[2];;
      dg_dt(I23) += fprime * dlprime_transformed_dt[2] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[2] * dlprime_transformed_dt[3];;
      dg_dt(I33) += fprime * dlprime_transformed_dt[3] * l_lowerprime_transformed[3] + fprime * l_lowerprime_transformed[3] * dlprime_transformed_dt[3];;


      dgprime_dX1.DeleteAthenaArray();
      dgprime_dX2.DeleteAthenaArray();
      dgprime_dX3.DeleteAthenaArray();


}
  Lambda.DeleteAthenaArray();
  dLambda_dt.DeleteAthenaArray();


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

  // Real x1p = x1 + DEL * r;
  // Real x1m = x1 - DEL * r;

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

  // Real x2p = x2 + DEL * r;
  // Real x2m = x2 - DEL * r;

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
  
  // Real x3p = x3 + DEL * r;
  // Real x3m = x3 - DEL * r;

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
