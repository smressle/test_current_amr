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
void boosted_BH_metric_addition(Real q_rat,Real xprime, Real yprime, Real zprime, Real rprime, Real Rprime, Real vx, Real vy, Real vz,Real ax, Real ay, Real az,AthenaArray<Real> &g_pert );
void single_bh_metric(Real a, Real x1, Real x2, Real x3, ParameterInput *pin,AthenaArray<Real> &g);


void NobleCooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);


void smoothed_bh_metric(Real t, Real x1, Real x2, Real x3,ParameterInput *pin,AthenaArray<Real> &g);




// Global variables
static Real m;                                  // black hole parameters
static int sample_n_r, sample_n_theta;             // number of cells in 2D sample grid
static int sample_n_phi;                           // number of cells in 3D sample grid
static Real dfloor,pfloor;                         // density and pressure floors
static Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters


static Real q;          // black hole mass and spin
static Real m_tot;     // total black hole mass
static Real t0; //time at which second BH is at polar axis
static Real field_norm;

static Real black_hole_smoothing_radius; // radius inside which to smooth the metric.
static Real black_hole_smoothing_radius_before_restart; // radius inside which to smooth the metric.




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


enum b_configs {vertical, normal, renorm, MAD,multi_loop};

static Real k_adi;                      // hydro parameters
static Real rin, r_peak, l, rho_max;            // fixed torus parameters
static Real psi, sin_psi, cos_psi;                 // tilt parameters
static Real log_h_edge, log_h_peak;                // calculated torus parameters
static Real pgas_over_rho_peak, rho_peak;          // more calculated torus parameters
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
static Real sample_r_rat;                          // sample grid geometric spacing ratio
static Real sample_cutoff;                         // density cutoff for sample grid
static Real x1_min, x1_max, x2_min, x2_max;        // 2D limits in chosen coordinates
static Real x3_min, x3_max;                        // 3D limits in chosen coordinates
static Real r_min, r_max, theta_min, theta_max;    // limits in r,theta for 2D samples
static Real phi_min, phi_max;                      // limits in phi for 3D samples
static Real pert_amp, pert_kr, pert_kz;            // parameters for initial perturbations

// static Real rh;                            
        // horizon radius
// Constants Needed for Torus
static Real lin;
static Real c_const;
static Real n_pow;
static Real ud_t_in;
static Real kappa_init;




//----------------------------------------------------------------------------------------
// Functions for Chakrabarti Torus

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




  // if (MAGNETIC_FIELDS_ENABLED) field_norm =  pin->GetReal("problem", "field_norm");



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

  black_hole_smoothing_radius = 4.0;
  black_hole_smoothing_radius_before_restart = 4.0;





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
  m_tot = 1.0+q;


  // if (MAGNETIC_FIELDS_ENABLED) EnrollUserExplicitEMFSourceFunction(emf_source);

  // fprintf(stderr,"Done with set_orbit_arrays \n");


  gamma_max = pin->GetOrAddReal("hydro", "gamma_max", 1000.0);


  Real gam = pin->GetReal("hydro", "gamma");

      //SEE DE VILLIERS+ 2003 https://arxiv.org/pdf/astro-ph/0307260.pdf

  Real a = 0;

  Real rc = r_peak;
  Real lc = l_kep(a,rc);


    // return 1.0/np.sqrt( - (gtphi(r,a,theta) + gtt(r,a,theta)*l) / (l*gphiphi(r,a,theta) + l**2.0*gtphi(r,a,theta) )  )

  Real lambda_in = std::sqrt(-gphiphi(rin,a,PI/2.0)/gtt(rin,a,PI/2.0) ); //lambda_func(rin,a,PI/2.0,lin)
  Real lambda_c = std::sqrt(-gphiphi(rc,a,PI/2.0)/gtt(rc,a,PI/2.0) ); //3lambda_func(rc,a,PI/2.0,lc)


  lin = lc/std::exp(n_pow*std::log(lambda_c/lambda_in) );
  c_const = lc/std::pow(lambda_c,n_pow);

  Real alpha_pow = (2.0*n_pow-2.0)/n_pow;

  ud_t_in = -1.0/std::sqrt( - (gitt(rin,a,PI/2.0) - 2.0*lin*gitphi(rin,a,PI/2.0) + SQR(lin)*giphiphi(rin,a,PI/2.0) ) );



  // Compute Peak Density //
  Real denom_sq = -( gitt(rc,a,PI/2.0) - 2.0*lc*gitphi(rc,a,PI/2.0) + SQR(lc)*giphiphi(rc,a,PI/2.0) );
  Real ud_t_c = -1.0/std::sqrt(denom_sq);
  Real eps_c = 1.0/gam * (ud_t_in * f(lin,c_const,n_pow)/(ud_t_c * f(lc,c_const,n_pow)) -1.0);
  rho_peak = std::pow( (eps_c * (gam-1.0)/k_adi), (1.0/(gam-1.0)) );
  pgas_over_rho_peak = eps_c * (gam-1.0);
  
  kappa_init = k_adi * std::pow(rho_peak,gam-1.0);

   fprintf(stderr,"eps_c: %g gam: %g, k_adi: %g ud_t_in: %g rin: %g rc: %g lc: %g udtc: %g c_const: %g f: %g %g kappa_init: %g\n",eps_c,gam,k_adi,ud_t_in,rin,rc,lc,ud_t_c,c_const,
     f(lc,c_const,n_pow), f(lin,c_const,n_pow),kappa_init);

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
  int refine = 0;

    Real DX,DY,DZ;
    Real dx,dy,dz;
  get_uniform_box_spacing(pmb->pmy_mesh->mesh_size,&DX,&DY,&DZ);
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);


  Real total_box_radius = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2.0;


  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  // if (current_level >=max_refinement_level) return 0;

  int any_in_refinement_region = 0;
  int any_at_current_level=0;


  int max_level_required = 0;


  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);


  //first loop: check if any part of block is within refinement levels for secondary black hole

if (max_second_bh_refinement_level>0){

    for (int k = pmb->ks; k<=pmb->ke;k++){
      for(int j=pmb->js; j<=pmb->je; j++) {
        for(int i=pmb->is; i<=pmb->ie; i++) {

            
              Real x = pmb->pcoord->x1v(i);
              Real y = pmb->pcoord->x2v(j);
              Real z = pmb->pcoord->x3v(k);

              Real xprime,yprime,zprime,rprime,Rprime;
              get_prime_coords(2,x,y,z, orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
              Real box_radius = 12.0; //total_box_radius/std::pow(2.,max_second_bh_refinement_level-2.0)*0.9999;

              Real z_radius = 1.53125;

              
              Real mesh_block_widthx = pmb->block_size.nx1 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx1*1.0);
              Real mesh_block_widthy = pmb->block_size.nx2 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx2*1.0);
              Real mesh_block_widthz = pmb->block_size.nx3 * z_radius*2.0/(pmb->pmy_mesh->mesh_size.nx3*1.0);

          
  
              if (xprime<(box_radius-mesh_block_widthx/2.0) && xprime > -(box_radius-mesh_block_widthx/2.0) && 
                yprime<(box_radius-mesh_block_widthy/2.0) && yprime > -(box_radius-mesh_block_widthy/2.0) && 
                zprime<(z_radius-mesh_block_widthz/2.0) && zprime > -(z_radius-mesh_block_widthz/2.0) ){
                max_level_required=max_second_bh_refinement_level;
                any_in_refinement_region=1;

                if (current_level < max_second_bh_refinement_level){
      
                  orbit_quantities.DeleteAthenaArray();
                    return  1;
                }
                if (current_level==max_second_bh_refinement_level) any_at_current_level=1;
              }


            
            }

          }
        }
      
        

    //second loop: check if any part of block is within refinement levels for primary black hole

    for (int k = pmb->ks; k<=pmb->ke;k++){
      for(int j=pmb->js; j<=pmb->je; j++) {
        for(int i=pmb->is; i<=pmb->ie; i++) {

            
              Real x = pmb->pcoord->x1v(i);
              Real y = pmb->pcoord->x2v(j);
              Real z = pmb->pcoord->x3v(k);

              Real xprime,yprime,zprime,rprime,Rprime;
              get_prime_coords(1,x,y,z, orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
              Real box_radius = 12.0; //total_box_radius/std::pow(2.,max_second_bh_refinement_level-2.0)*0.9999;

              Real z_radius = 1.53125;

              
              Real mesh_block_widthx = pmb->block_size.nx1 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx1*1.0);
              Real mesh_block_widthy = pmb->block_size.nx2 * box_radius*2.0/(pmb->pmy_mesh->mesh_size.nx2*1.0);
              Real mesh_block_widthz = pmb->block_size.nx3 * z_radius*2.0/(pmb->pmy_mesh->mesh_size.nx3*1.0);

          
  
              if (xprime<(box_radius-mesh_block_widthx/2.0) && xprime > -(box_radius-mesh_block_widthx/2.0) && 
                yprime<(box_radius-mesh_block_widthy/2.0) && yprime > -(box_radius-mesh_block_widthy/2.0) && 
                zprime<(z_radius-mesh_block_widthz/2.0) && zprime > -(z_radius-mesh_block_widthz/2.0) ){
                max_level_required=max_second_bh_refinement_level;
                any_in_refinement_region=1;

                if (current_level < max_second_bh_refinement_level){
      
                  orbit_quantities.DeleteAthenaArray();
                    return  1;
                }
                if (current_level==max_second_bh_refinement_level) any_at_current_level=1;
              }


            
            }

          }
        }
}

  //third loop: resolve circumbinary disk

  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
          
          for (int n_level = 1; n_level<=max_smr_refinement_level; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

            Real z_radius = box_radius;


            if (total_box_radius>1000){
              if (n_level==1) z_radius = 392.0*0.9999;
              if (n_level==2) z_radius = 196.0*0.9999;
              if (n_level==3) z_radius = 98.0*0.9999;
              if (n_level==4) z_radius = 49.0*0.9999;
              if (n_level==5) z_radius = 24.5*0.9999;
              // if (n_level==5) z_radius = 12.25*0.9999;
              // if (n_level==6) z_radius = 2.4*0.9999;
              // if (n_level==7) z_radius = 1.2*0.9999;

              if (n_level>=3) box_radius = total_box_radius/std::pow(2.,n_level-2)*0.9999;
            }
            else{
              if (n_level==1) z_radius = 196.0*0.9999;
              if (n_level==2) z_radius = 98.0*0.9999;
              if (n_level==3) z_radius = 49.0*0.9999;
              if (n_level==4) z_radius = 24.5*0.9999;
              // if (n_level==5) z_radius = 12.25*0.9999;
              // if (n_level==6) z_radius = 2.4*0.9999;
              // if (n_level==7) z_radius = 1.2*0.9999;

              if (n_level>=2) box_radius = total_box_radius/std::pow(2.,n_level-2)*0.9999;
            }

 
            if (x<box_radius && x > -box_radius && y<box_radius
              && y > -box_radius && z<z_radius && z > -z_radius ){


              if (n_level>max_level_required) max_level_required=n_level;
              any_in_refinement_region = 1;
              if (current_level < n_level){
                  orbit_quantities.DeleteAthenaArray();
                  return  1;
              }
              if (current_level==n_level) any_at_current_level=1;
            }



          
          }

  }
 }
}


orbit_quantities.DeleteAthenaArray();
if (current_level>max_level_required) return -1;
else if (current_level==max_level_required) return 0;
else return 1;
}


void get_Chakrabarti_torus_single_BH(ParameterInput *pin, Real x,Real y, Real z, Real a, Real *rho, Real *press, Real *vel1, Real *vel2, Real *vel3, bool *is_in_torus){


    Real gam = pin->GetReal("hydro", "gamma");

    AthenaArray<Real> g_single_bh, gi_single_bh;
    g_single_bh.NewAthenaArray(NMETRIC);
    gi_single_bh.NewAthenaArray(NMETRIC);


    single_bh_metric(a,x,y,z, pin,g_single_bh);
    bool invertible = gluInvertMatrix(g_single_bh,gi_single_bh);


        // Calculate Boyer-Lindquist coordinates of cell
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(x,y,z,0,0,a, &r,
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
      *is_in_torus = false;

      rho_sol = 0.0;
      ug_sol = 0.0;
      pgas_sol = 0.0;
      uu_t_sol = 1.0;
      uu_phi_sol = 0.0;
    }
    else{
      *is_in_torus = true;

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


    // velocities in Boyer-Lindquist coordinates
    Real u0_bl, u1_bl, u2_bl, u3_bl;

    u0_bl = uu_t_sol;
    u1_bl = 0.0;
    u2_bl = 0.0;
    u3_bl = uu_phi_sol;

    Real u0, u1, u2, u3;
    TransformVector(u0_bl, u1_bl, u2_bl, u3_bl, x, y, z, a,&u0, &u1, &u2, &u3);


    Real uu1 = u1 - gi_single_bh(I01)/gi_single_bh(I00) * u0;
    Real uu2 = u2 - gi_single_bh(I02)/gi_single_bh(I00) * u0;
    Real uu3 = u3 - gi_single_bh(I03)/gi_single_bh(I00) * u0;

    g_single_bh.DeleteAthenaArray();
    gi_single_bh.DeleteAthenaArray();

    *rho = rho_sol;
    *press = pgas_sol;
    *vel1 = uu1;
    *vel2 = uu2;
    *vel3 = uu3;

    return;

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

        Real rho_sol,pgas_sol,uu1,uu2,uu3;
        bool is_in_torus;
        get_Chakrabarti_torus_single_BH(pin, pcoord->x1v(i)/m_tot, pcoord->x2v(j)/m_tot, pcoord->x3v(k)/m_tot, a, 
                                        &rho_sol, &pgas_sol, &uu1, &uu2, &uu3, &is_in_torus);

        // uu1 = uu1*m_tot;
        // uu2 = uu2*m_tot;
        // uu3 = uu3*m_tot;



        in_torus(k,j,i) = is_in_torus;


        // fprintf(stderr,"xyz: %g %g %g \n rho: %g P: %g uu: %g %g %g mtot: %g rho_peak: %g in torus: %d\n",pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), rho_sol,pgas_sol,uu1,uu2,uu2,m_tot,rho_peak, is_in_torus);

        // Calculate background primitives
        Real rho = rho_min * std::pow(r, rho_pow);
        Real pgas = pgas_min * std::pow(r, pgas_pow);


        Real perturbation = 0.0;
        // Overwrite primitives inside torus
        if (in_torus(k,j,i) ) {

          int seed = Globals::my_rank * block_size.nx1*block_size.nx2*block_size.nx3+ (k - ks) * block_size.nx2 * block_size.nx1 + (j - js) * block_size.nx1 + i - is;
          generator.seed(seed);
          perturbation = uniform(generator);

          // Calculate thermodynamic variables
          rho = rho_sol /rho_peak;
          pgas = pgas_sol / rho_peak;


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

        if (std::isnan(rho)){
          fprintf(stderr,"ISNAN in rho at xyz: %g %g %g \n g: %g %g %g %g %g %g %g %g %g %g \n gi: %g %g %g %g %g %g %g %g %g %g \n",
            pcoord->x1v(i), pcoord->x2v(j),pcoord->x3v(k),
            g(I00,i),g(I01,i),g(I02,i),g(I03,i),g(I11,i),g(I22,i),g(I33,i),g(I12,i),g(I13,i),g(I23,i),
            gi(I00,i),gi(I01,i),gi(I02,i),gi(I03,i),gi(I11,i),gi(I22,i),gi(I33,i),gi(I12,i),gi(I13,i),gi(I23,i));
        }
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
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k_torus),0,0,0,
                &r, &theta, &phi);
            int k_torus = k;
            int j_torus = j;
            int i_torus = i;

            if (k_torus==ku+1) k_torus =ku;
            if (j_torus==ju+1) j_torus =ju;
            if (i_torus==iu+1) i_torus =iu;


            if (r >= rin) {
              if (in_torus(k_torus,j_torus,i_torus) == true) {
                Real rho = phydro->w(IDN,k_torus,j_torus,i_torus);
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

             int k_torus = k;
            int j_torus = j;
            int i_torus = i;

            if (k_torus==ku+1) k_torus =ku;
            if (j_torus==ju+1) j_torus =ju;
            if (i_torus==iu+1) i_torus =iu;
            Real r, theta, phi;

            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k_torus),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k_torus,j_torus,i_torus) == true) {
                Real rho = phydro->w(IDN,k_torus,j_torus,i_torus);
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
            int k_torus = k;
            int j_torus = j;
            int i_torus = i;

            if (k_torus==ku+1) k_torus =ku;
            if (j_torus==ju+1) j_torus =ju;
            if (i_torus==iu+1) i_torus =iu;
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k_torus),0,0,0,
                &r, &theta, &phi);
            if (r >= rin) {
              if (in_torus(k_torus,j_torus,i_torus) == true) {
                Real rho = phydro->w(IDN,k_torus,j_torus,i_torus);
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
            TransformAphi(a_phi_edges(k,j+1,i),pcoord->x1f(i), pcoord->x2f(j+1),pcoord->x3v(k),0,
                &tmp,&tmp,&Az_2);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2f(j),pcoord->x3v(k),0,
                &tmp,&tmp,&Az_1);
                  

            pfield->b.x1f(k,j,i) = 1.0/std::sqrt(-det) * (Az_2-Az_1) / (pcoord->dx2f(j) );

            //d Ay/dz
            Real  Ay_2,Ay_1;
            TransformAphi(a_phi_edges(k+1,j,i),pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k+1), 0,
                &tmp,&Ay_2,&tmp);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k), 0,
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
            TransformAphi(a_phi_edges(k+1,j,i),pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k+1),0,
                &Ax_2,&tmp,&tmp);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k), 0, 
                &Ax_1,&tmp,&tmp);
                  

            pfield->b.x2f(k,j,i) = 1.0/std::sqrt(-det) * (Ax_2-Ax_1) / (pcoord->dx3f(k) );

            //d Az/dx
            Real Az_2,Az_1;
            TransformAphi(a_phi_edges(k,j,i+1),pcoord->x1f(i+1), pcoord->x2f(j),pcoord->x3v(k), 0, 
                &tmp,&tmp,&Az_2);
            TransformAphi(a_phi_edges(k,j,i)  ,pcoord->x1f(i), pcoord->x2f(j),pcoord->x3v(k),0,
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
            TransformAphi(a_phi_edges(k,j,i+1),pcoord->x1f(i+1), pcoord->x2v(j),pcoord->x3f(k),0,
                &tmp,&Ay_2,&tmp);
            TransformAphi(a_phi_edges(k,j,i),  pcoord->x1f(i), pcoord->x2v(j),pcoord->x3f(k),0,
                &tmp,&Ay_1,&tmp);
                  

            pfield->b.x3f(k,j,i) = 1.0/std::sqrt(-det) * (Ay_2-Ay_1) / (pcoord->dx1f(i) );

            //d Ax/dy
            Real Ax_2,Ax_1;
            TransformAphi(a_phi_edges(k,j+1,i),pcoord->x1v(i), pcoord->x2f(j+1),pcoord->x3f(k),0,
                &Ax_2,&tmp,&tmp);
            TransformAphi(a_phi_edges(k,j,i),  pcoord->x1v(i), pcoord->x2f(j),pcoord->x3f(k),0,
                &Ax_1,&tmp,&tmp);

            pfield->b.x3f(k,j,i) -= 1.0/std::sqrt(-det) * (Ax_2-Ax_1) / (pcoord->dx2f(j) );

            pfield->b.x3f(k,j,i) *= normalization;

            if (std::isnan(pfield->b.x3f(k,j,i))){
              fprintf(stderr,"NAN in field \n det: %g Ax_2: %g Ax_1: %g \n", det,Ax_2,Ax_1);
            
            }
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



void set_orbit_arrays(std::string orbit_file_name){
      FILE *input_file;
        if ((input_file = fopen(orbit_file_name.c_str(), "r")) == NULL)   
               fprintf(stderr, "Cannot open %s, %s\n", "input_file",orbit_file_name.c_str());



    fscanf(input_file, "%i %lf \n", &nt, &q);
    // int nt = 10;
    // q = 1.0;

       
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

  Real orbital_radius = std::sqrt( SQR(orbit_quantities(IX1)) + SQR(orbit_quantities(IY1)) + SQR(orbit_quantities(IZ1)) );

  Real excision_radius = 0.0; //orbital_radius * 1.2;



   for (int k=kl; k<=ku; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=jl; j<=ju; ++j) {
      pmb->pcoord->CellMetric(k, j, il, iu, g, gi);
#pragma simd
      for (int i=il; i<=iu; ++i) {



          Real x = pmb->pcoord->x1v(i);
          Real y = pmb->pcoord->x2v(j);
          Real z = pmb->pcoord->x3v(k);

          if (std::isnan(pmb->pfield->bcc(IB1,k,j,i))) {
            fprintf(stderr,"NAN in field before inner boundary at xyz: %g %g %g \n", x,y,z);
                  fprintf(stderr,"rho: %g  P: %g v: %g %g %g \n", prim(IDN,k,j,i),prim(IPR,k,j,i),
                    prim(IVX,k,j,i),prim(IVY,k,j,i),prim(IVZ,k,j,i));
                  for (int n=0; n<NMETRIC; ++n) fprintf(stderr,"n: %d g: %g gi: %g \n",n,g(n,i),gi(n,i));
                  exit(0);
          }
            

          Real pseudo_r = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
          Real t = pmb->pmy_mesh->metric_time;

          Real xprime,yprime,zprime,rprime,Rprime;

          get_prime_coords(1,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);

          Real thprime,phiprime;
          GetBoyerLindquistCoordinates(xprime,yprime,zprime,a1x,a1y,a1z, &rprime, &thprime, &phiprime);





          if (pseudo_r <= excision_radius){


              // Calculate normal frame Lorentz factor
              Real uu1 = 0.0;
              Real uu2 = 0.0;
              Real uu3 = 0.0;
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


              // uu1 = u1prime - gi(I01,i) / gi(I00,i) * u0prime;
              // uu2 = u2prime - gi(I02,i) / gi(I00,i) * u0prime;
              // uu3 = u3prime - gi(I03,i) / gi(I00,i) * u0prime;

              
              prim(IDN,k,j,i) = dfloor;
              prim(IVX,k,j,i) = uu1;
              prim(IVY,k,j,i) = uu2;
              prim(IVZ,k,j,i) = uu3;
              prim(IPR,k,j,i) = pfloor;

          }

          if (rprime < rh*0.8) {
            rprime = rh*0.8;
            convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a1x,a1y,a1z,&xprime,&yprime,&zprime);
          }

          if (rprime < rh or rprime < black_hole_smoothing_radius){

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

          if (rprime < rh2 or rprime < black_hole_smoothing_radius){

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


              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(2,t,u0,u1,u2,u3, orbit_quantities,&u0prime,&u1prime,&u2prime,&u3prime);


 

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


            // if (std::isnan(pmb->pfield->bcc(IB1,k,j,i))) {
            //       fprintf(stderr,"NAN in field after inner boundary at xyz: %g %g %g \n", x,y,z);
            //       fprintf(stderr,"rho: %g  P: %g v: %g %g %g \n", prim(IDN,k,j,i),prim(IPR,k,j,i),
            //         prim(IVX,k,j,i),prim(IVY,k,j,i),prim(IVZ,k,j,i));
            //       for (int n=0; n<NMETRIC; ++n) fprintf(stderr,"n: %d g: %g gi: %g \n",n,g(n,i),gi(n,i));
            //       exit(0);
            //     }
            




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

  // Real k_target = 0.0005;



  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(pmb->pmy_mesh->metric_time,orbit_quantities);

  Real a1 = std::sqrt( SQR(orbit_quantities(IA1X)) + SQR(orbit_quantities(IA1Y)) + SQR(orbit_quantities(IA1Z)));
  Real a2 = std::sqrt( SQR(orbit_quantities(IA2X)) + SQR(orbit_quantities(IA2Y)) + SQR(orbit_quantities(IA2Z)));



  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
      for (int i=pmb->is; i<=pmb->ie; ++i) {


        // if (std::isnan(pmb->pfield->bcc(IB1,k,j,i))) {
        //   fprintf(stderr,"NAN in field before cooling at xyz: %g %g %g \n", pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));
        //           fprintf(stderr,"rho: %g  P: %g v: %g %g %g \n", prim(IDN,k,j,i),prim(IPR,k,j,i),
        //             prim(IVX,k,j,i),prim(IVY,k,j,i),prim(IVZ,k,j,i));
        //           exit(0);
        // }
            


        Real r_com,theta_com, phi_com;
        GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), 0,0,0,
          &r_com,&theta_com, &phi_com);

        Real binary_separation_distance = std::sqrt( 
                                              SQR(orbit_quantities(IX1)-orbit_quantities(IX2)) + 
                                              SQR(orbit_quantities(IY1)-orbit_quantities(IY2)) +
                                              SQR(orbit_quantities(IZ1)-orbit_quantities(IZ2)) );

        Real Omega_com = std::sqrt(1.0+q)/( std::pow(r_com,1.5) );

        Real xprime,yprime,zprime,rprime_1,Rprime;
        get_prime_coords(1,pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), orbit_quantities, &xprime,&yprime, &zprime, &rprime_1,&Rprime);
        Real radius, theta,phi;
        GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), orbit_quantities(IA1X),orbit_quantities(IA1Y),orbit_quantities(IA1Z),
          &radius,&theta, &phi);

        Real rprime_2;

        get_prime_coords(2,pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), orbit_quantities,&xprime,&yprime, &zprime, &rprime_2,&Rprime);
        
        Real rhprime1 = ( 1.0 + std::sqrt(SQR(1.0)-SQR(a1)) );
        Real rhprime2 = ( q + std::sqrt(SQR(q)-SQR(a2)) );

        Real ug = prim(IPR,k,j,i)/(gamma_adi-1.0);

        Real Omega_primary = 1.0/( std::pow(rprime_1,1.5) + a1);
        Real Omega_secondary = 1.0/(q+SMALL) * 1.0/( std::pow(rprime_2/(q+SMALL),1.5) + a2/(q+SMALL));

        // Omega_secondary = Omega_secondary*q;

        Real r_isco_primary = risco_calc_general( 1, a1, m );
        Real r_isco_secondary = risco_calc_general( 1, a2/(q+SMALL), q ); //neads a/M, returns isco in units of M_1

        if (rprime_1<r_isco_primary) Omega_primary = 1.0/( std::pow(r_isco_primary,1.5) + a1);
        if (rprime_2<r_isco_secondary) Omega_secondary = 1.0/(q+SMALL) * 1.0/( std::pow(r_isco_secondary/(q+SMALL),1.5) + a2/(q+SMALL));


        Real Y = prim(IPR,k,j,i)/std::pow(prim(IDN,k,j,i), gamma_adi)/kappa_init;


        Real L_cool;

        Real Omega_com_1point5 = std::sqrt(1.0+q)/( std::pow(1.5*binary_separation_distance,1.5) );

        // Avara 2023

        if (r_com>binary_separation_distance){
          L_cool = Omega_com * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );

        }
        else if (rprime_1<0.45*binary_separation_distance){
          L_cool = Omega_primary * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        }
        else if (rprime_2<0.45*binary_separation_distance){
          L_cool = Omega_secondary * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        }
        else{
          L_cool = Omega_com_1point5 * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        }
        // if (Target_Temperature>=Target_Temperature_secondary){
        //   L_cool = Omega * ug * std::sqrt( Y-1.0 +  std::fabs(Y-1.0) );
        // }
        // else{
        //   L_cool = Omega_secondary * ug * std::sqrt( Y_secondary-1.0 +  std::fabs(Y_secondary-1.0) );
        // }

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

        if (rprime_1 < rhprime1) L_cool = 0.0;
        if (rprime_2 < rhprime2) L_cool = 0.0;


        // Real L_cool_max = 0.2 * ug /dt; /// L_cool * dt < 0.2 * ug
        // if (L_cool> L_cool_max) L_cool = L_cool_max;

        cons(IEN,k,j,i) += -dt * L_cool * u_0; 
        cons(IM1,k,j,i) += -dt * L_cool * u_1;
        cons(IM2,k,j,i) += -dt * L_cool * u_2;
        cons(IM3,k,j,i) += -dt * L_cool * u_3;

        Real ug_frac = dt * L_cool/ug;



        // if (std::isnan(pmb->pfield->bcc(IB1,k,j,i))) {
        //   fprintf(stderr,"NAN in field after cooling at xyz: %g %g %g \n", pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));
        //           fprintf(stderr,"rho: %g  P: %g v: %g %g %g \n", prim(IDN,k,j,i),prim(IPR,k,j,i),
        //             prim(IVX,k,j,i),prim(IVY,k,j,i),prim(IVZ,k,j,i));
        //           exit(0);
        // }
            

        // pmb->user_out_var(0,k,j,i) = L_cool * u_0;

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
  // if (std::fabs(a_dot_x_prime))
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
    fprintf(stderr,"Choose a valid BH_INDEX!!!: %d",BH_INDEX);
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

/// Keep divB=0 with new metric

void  MeshBlock::PreserveDivbNewMetric(ParameterInput *pin){
  int SCALE_DIVERGENCE = false; 
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
        

        smoothed_bh_metric(pmy_mesh->time,pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old1m);
        smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pin,g_old2m);
        smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pin,g_old3m);

        smoothed_bh_metric(pmy_mesh->time,pcoord->x1f(i+1), pcoord->x2v(j), pcoord->x3v(k), pin,g_old1p);
        smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2f(j+1), pcoord->x3v(k), pin,g_old2p);
        smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k+1), pin,g_old3p);

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


   for (int k=kl; k<=ku; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=jl; j<=ju; ++j) {
      pcoord->CellMetric(k, j, il, iu,g, gi);
#pragma simd
      for (int i=il; i<=iu; ++i) {

                // Prepare scratch arrays
        AthenaArray<Real> g_tmp,g_old,gi_old,g_diff;
        g_tmp.NewAthenaArray(NMETRIC);
        g_old.NewAthenaArray(NMETRIC);
        gi_old.NewAthenaArray(NMETRIC);
        g_diff.NewAthenaArray(NMETRIC);
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

        smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old);

        bool invertible =gluInvertMatrix(g_old,gi_old);



        g_diff(I00) = g_tmp(I00) - g_old(I00);
        g_diff(I01) = g_tmp(I01) - g_old(I01);
        g_diff(I02) = g_tmp(I02) - g_old(I02);
        g_diff(I03) = g_tmp(I03) - g_old(I03);
        g_diff(I11) = g_tmp(I11) - g_old(I11);
        g_diff(I12) = g_tmp(I12) - g_old(I12);
        g_diff(I13) = g_tmp(I13) - g_old(I13);
        g_diff(I22) = g_tmp(I22) - g_old(I22);
        g_diff(I23) = g_tmp(I23) - g_old(I23);
        g_diff(I33) = g_tmp(I33) - g_old(I33);


        Real uu1 = phydro->w(IVX,k,j,i);
        Real uu2 = phydro->w(IVY,k,j,i);
        Real uu3 = phydro->w(IVZ,k,j,i);
        Real tmp = g_old(I11)*uu1*uu1 + 2.0*g_old(I12)*uu1*uu2 + 2.0*g_old(I13)*uu1*uu3
                 + g_old(I22)*uu2*uu2 + 2.0*g_old(I23)*uu2*uu3
                 + g_old(I33)*uu3*uu3;
        Real gamma = std::sqrt(1.0 + tmp);
        // user_out_var(0,k,j,i) = gamma;

        // Calculate 4-velocity
        Real alpha = std::sqrt(-1.0/gi_old(I00));
        Real u0 = gamma/alpha;
        Real u1 = uu1 - alpha * gamma * gi_old(I01);
        Real u2 = uu2 - alpha * gamma * gi_old(I02);
        Real u3 = uu3 - alpha * gamma * gi_old(I03);

        Real b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
        Real b_sq = 0.0;

        if (MAGNETIC_FIELDS_ENABLED) {
    
                // Calculate 4-magnetic field
                Real bb1 = pfield->bcc(IB1,k,j,i);
                Real bb2 = pfield->bcc(IB2,k,j,i);
                Real bb3 = pfield->bcc(IB3,k,j,i);
                b0 = g_old(I01)*u0*bb1 + g_old(I02)*u0*bb2 + g_old(I03)*u0*bb3
                        + g_old(I11)*u1*bb1 + g_old(I12)*u1*bb2 + g_old(I13)*u1*bb3
                        + g_old(I12)*u2*bb1 + g_old(I22)*u2*bb2 + g_old(I23)*u2*bb3
                        + g_old(I13)*u3*bb1 + g_old(I23)*u3*bb2 + g_old(I33)*u3*bb3;
                b1 = (bb1 + b0 * u1) / u0;
                b2 = (bb2 + b0 * u2) / u0;
                b3 = (bb3 + b0 * u3) / u0;
                Real b_0, b_1, b_2, b_3;

                b_0 = g_old(I00)*b0 + g_old(I01)*b1 + g_old(I02)*b2 + g_old(I03)*b3;
                b_1 = g_old(I01)*b0 + g_old(I11)*b1 + g_old(I12)*b2 + g_old(I13)*b3;
                b_2 = g_old(I02)*b0 + g_old(I12)*b1 + g_old(I22)*b2 + g_old(I23)*b3;
                b_3 = g_old(I03)*b0 + g_old(I13)*b1 + g_old(I23)*b2 + g_old(I33)*b3;
                b_sq = b_0*b0 + b_1*b1 + b_2*b2 + b_3*b3;
            
      }

        Real gamma_adi = peos->GetGamma();
        Real wtot = phydro->w(IDN,k,j,i) + gamma_adi/(gamma_adi-1.0) * phydro->w(IPR,k,j,i) + b_sq;
        Real ptot = phydro->w(IPR,k,j,i) + 0.5*b_sq;
        Real tt[NMETRIC];
        tt[I00] = wtot * u0 * u0 + ptot * g_old(I00) - b0 * b0;
        tt[I01] = wtot * u0 * u1 + ptot * g_old(I01) - b0 * b1;
        tt[I02] = wtot * u0 * u2 + ptot * g_old(I02) - b0 * b2;
        tt[I03] = wtot * u0 * u3 + ptot * g_old(I03) - b0 * b3;
        tt[I11] = wtot * u1 * u1 + ptot * g_old(I11) - b1 * b1;
        tt[I12] = wtot * u1 * u2 + ptot * g_old(I12) - b1 * b2;
        tt[I13] = wtot * u1 * u3 + ptot * g_old(I13) - b1 * b3;
        tt[I22] = wtot * u2 * u2 + ptot * g_old(I22) - b2 * b2;
        tt[I23] = wtot * u2 * u3 + ptot * g_old(I23) - b2 * b3;
        tt[I33] = wtot * u3 * u3 + ptot * g_old(I33) - b3 * b3;


        // addition of perturber is like changing dg/dt in one timestep, cooresponding to a 
        // source term of 1/2 dgmu nu/dt T^mu nu
        // so mulitply by dt to get addition, with is the secondary part of the metric.  
        Real s_E = 0.0;
        for (int n = 0; n < NMETRIC; ++n) {
          s_E += g_diff(n) * tt[n];
        }
        s_E -= 0.5 * (  g_diff(I00) * tt[I00]
                      + g_diff(I11) * tt[I11]
                      + g_diff(I22) * tt[I22]
                      + g_diff(I33) * tt[I33]);


        phydro->u(IEN,k,j,i) += s_E;

        Real det_old = Determinant(g_old);

         Real fac = std::sqrt(-det_old)/std::sqrt(-det_new);
          for (int n_cons=IDN; n_cons<= IEN; ++n_cons){
            phydro->u(n_cons,k,j,i) *=fac;
          }

        g_tmp.DeleteAthenaArray();
        g_old.DeleteAthenaArray();
        gi_old.DeleteAthenaArray();
        g_diff.DeleteAthenaArray();

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

        if (dir==0) smoothed_bh_metric(pmy_mesh->time,pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pin,g_old);
        if (dir==1) smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pin,g_old);
        if (dir==2) smoothed_bh_metric(pmy_mesh->time,pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pin,g_old);


        Real det_old = Determinant(g_old);


        if (dir==0) pfield->b.x1f(k,j,i) *= std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==1) pfield->b.x2f(k,j,i) *= std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==2) pfield->b.x3f(k,j,i) *= std::sqrt(-det_old)/std::sqrt(-det_new);


        if (dir==0 && i>=is && i<=ie  && j<=je && j>=js && k<=ke && k>=ks) face1rat_used(k,j,i) = std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==1 && i>=is && i<=ie  && j<=je && j>=js && k<=ke && k>=ks) face2rat_used(k,j,i) = std::sqrt(-det_old)/std::sqrt(-det_new);
        if (dir==2 && i>=is && i<=ie  && j<=je && j>=js && k<=ke && k>=ks) face3rat_used(k,j,i) = std::sqrt(-det_old)/std::sqrt(-det_new);


        g_tmp.DeleteAthenaArray();
        g_old.DeleteAthenaArray();

      }
    }
  }
}




  Real divb,divbmax;
  divbmax=0;
  // AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pfield->b;


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

  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  } else {
    bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  }

  // Initialize conserved values
  // if (MAGNETIC_FIELDS_ENABLED) {
  //   peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
  //       kl, ku);
  // } else {
  //   peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  //   bb.DeleteAthenaArray();
  // }


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

  // fprintf(stderr,"black hole smoothing radius: %g \n a1: %g %g %g \n a2: %g %g %g \n v1: %g %g %g v2: %g %g %g \n",black_hole_smoothing_radius,
  //   a1x,a1y,a1z,a2x,a2y,a2z,v1x,v1y,v1z,v2x,v2y,v2z);

  if (rprime<black_hole_smoothing_radius){
      Real thprime,phiprime;
      GetBoyerLindquistCoordinates(xprime,yprime,zprime,a1x,a1y,a1z, &rprime, &thprime, &phiprime);
      if (rprime < black_hole_smoothing_radius) {
            rprime = black_hole_smoothing_radius;
            convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a1x,a1y,a1z,&xprime,&yprime,&zprime);
      }

  }


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

  if (rprime<black_hole_smoothing_radius){
    Real thprime,phiprime;
    GetBoyerLindquistCoordinates(xprime,yprime,zprime,a1x,a1y,a1z, &rprime, &thprime, &phiprime);
    if (rprime < black_hole_smoothing_radius) {
          rprime = black_hole_smoothing_radius;
          convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a2x,a2y,a2z,&xprime,&yprime,&zprime);
    }

  }

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

void metric_for_derivatives_smoothed(Real t, Real x1, Real x2, Real x3, AthenaArray<Real> &orbit_quantities,
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


  if (rprime<black_hole_smoothing_radius_before_restart){
      Real thprime,phiprime;
      GetBoyerLindquistCoordinates(xprime,yprime,zprime,a1x,a1y,a1z, &rprime, &thprime, &phiprime);
      if (rprime < black_hole_smoothing_radius) {
            rprime = black_hole_smoothing_radius;
            convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a1x,a1y,a1z,&xprime,&yprime,&zprime);
      }

  }


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

  if (rprime<black_hole_smoothing_radius_before_restart){
    Real thprime,phiprime;
    GetBoyerLindquistCoordinates(xprime,yprime,zprime,a1x,a1y,a1z, &rprime, &thprime, &phiprime);
    if (rprime < black_hole_smoothing_radius) {
          rprime = black_hole_smoothing_radius;
          convert_spherical_to_cartesian_ks(rprime,thprime,phiprime, a2x,a2y,a2z,&xprime,&yprime,&zprime);
    }

  }

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

      Real xprime,yprime,zprime,rprime,Rprime,Rprime1;
      get_prime_coords(1,x,y,z,orbit_quantities,&xprime,&yprime,&zprime,&rprime,&Rprime1);

      get_prime_coords(2,x,y,z,orbit_quantities,&xprime,&yprime,&zprime,&rprime,&Rprime);

      if (Rprime<=a2 or Rprime1<=a1){

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


void smoothed_bh_metric(Real t, Real x1, Real x2, Real x3,ParameterInput *pin,AthenaArray<Real> &g){


  m = pin->GetReal("coord", "m");

  //////////////Perturber Black Hole//////////////////

  t0 = pin->GetOrAddReal("problem","t0", 0.0);
  Real x = x1;
  Real y = x2;
  Real z = x3;

  AthenaArray<Real> orbit_quantities;
  orbit_quantities.NewAthenaArray(Norbit);

  get_orbit_quantities(t,orbit_quantities);

  metric_for_derivatives_smoothed(t,x1,x2,x3,orbit_quantities,g);



  orbit_quantities.DeleteAthenaArray();
  return;

}

void single_bh_metric(Real a, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

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

