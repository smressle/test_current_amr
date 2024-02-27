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

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);
bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv);


void get_prime_coords(int BH_INDEX,Real x, Real y, Real z, AthenaArray<Real> &orbit_quantities,Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ);

void single_bh_metric(Real x1, Real x2, Real x3, ParameterInput *pin,AthenaArray<Real> &g);

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

// Global variables
static Real m, a;                                  // black hole parameters
static Real beta_min;                              // min ratio of gas to mag pressure
static int sample_n_r, sample_n_theta;             // number of cells in 2D sample grid
static int sample_n_phi;                           // number of cells in 3D sample grid
static Real dfloor,pfloor;                         // density and pressure floors
static Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters

// static Real rh;                                    // horizon radius
static Real bondi_radius;  // b^2/rho at inner radius


static Real q;          // black hole mass and spin
//Real q; 
// static Real r_inner_boundary,r_inner_boundary_2;
// static Real rh2;
// static Real eccentricity, tau, mean_angular_motion;
static Real t0; //time at which second BH is at polar axis
static Real orbit_inclination;

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


  bondi_radius  = pin->GetReal("problem", "bondi_radius");

  if (MAGNETIC_FIELDS_ENABLED) beta_min = pin->GetReal("problem", "beta_min");


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

  t0 = pin->GetOrAddReal("problem","t0", 1e4);
  m =pin->GetReal("coord", "m");

  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);


  std::string orbit_file_name;
  orbit_file_name =  pin->GetOrAddString("problem","orbit_filename", "orbits.in");
  set_orbit_arrays(orbit_file_name);

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

  // Get mass and spin of black hole
  m = pcoord->GetMass();
  // q = pin->GetOrAddReal("problem", "q", 0.1);
  // aprime = q * pin->GetOrAddReal("problem", "a_bh2", 0.0);
  // r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);

  t0 = pin->GetOrAddReal("problem","t0", 1e4);

  // orbit_inclination = pin->GetOrAddReal("problem","orbit_inclination",0.0);


  // rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  //r_inner_boundary = rh/2.0;


    // Get mass of black hole
  // Real m2 = q;

  // rh2 =  ( m2 + std::sqrt( SQR(m2) - SQR(aprime)) );
  //r_inner_boundary_2 = rh2/2.0;

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
  // Real bh1_focus_radius = 3.125;

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
            get_prime_coords(2,x,y,z, orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            //Real box_radius = bh2_focus_radius * std::pow(2.,max_second_bh_refinement_level - n_level)*0.9999;
            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

        
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
            get_prime_coords(1,x,y,z,orbit_quantities, &xprime,&yprime, &zprime, &rprime,&Rprime);
            Real box_radius = total_box_radius/std::pow(2.,n_level)*0.9999;

          

             // if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
             //   fprintf(stderr,"current level (SMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
             //    }
            if (xprime<box_radius && xprime > -box_radius && yprime<box_radius
              && yprime > -box_radius && zprime<box_radius && zprime > -box_radius ){


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

  Real cs_0 = 1.0/std::sqrt(bondi_radius);
  Real rho_0 = 1.0;
  Real gam = peos->GetGamma();
  Real P_0 = SQR(cs_0)*rho_0/gam;

  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);

// Ensure a different initial random seed for each meshblock.
  int64_t iseed = -1 - gid;

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
            &r, &theta, &phi);
        Real u0 = std::sqrt(-1.0/g(I00,i));
        Real uu1 = 0.0 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = 0.0 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = 0.0 - gi(I03,i)/gi(I00,i) * u0;

        Real amp = 0.00;
        //if (std::fabs(a)<1e-1) amp = 0.01;
        Real rval = amp*(ran2(&iseed) - 0.5);
    
        // if (r<r_cut){
        //   phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = 0.0;
        //   phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = 0.0;
        //   phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
        //   phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
        //   phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
        // }
        // else{ 
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho_0;
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = P_0;
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      // }
      }
    }
  }


  AthenaArray<Real> &g_ = ruser_meshblock_data[0];
  AthenaArray<Real> &gi_ = ruser_meshblock_data[1];


  

  // Impose density and pressure floors
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),0.0, 0.0, a, &r,
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



    fscanf(input_file, "%i %f \n", &nt, &q);
    // int nt = 10;
    q = 0.1;

       
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

          Real x = pmb->pcoord->x1v(i);
          Real y = pmb->pcoord->x2v(j);
          Real z = pmb->pcoord->x3v(k);
          Real t = pmb->pmy_mesh->metric_time;

          Real xprime,yprime,zprime,rprime,Rprime;

          get_prime_coords(1,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);


          if (rprime < rh){

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



              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(1,t,u0,u1,u2,u3, orbit_quantities,&u0prime,&u1prime,&u2prime,&u3prime);




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



          get_prime_coords(2,x,y,z, orbit_quantities,&xprime,&yprime, &zprime, &rprime,&Rprime);


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



              Real u0prime,u1prime,u2prime,u3prime;
              BoostVector(2,t,u0,u1,u2,u3, orbit_quantities,&u0prime,&u1prime,&u2prime,&u3prime);
 


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

  Real xbh,ybh,zbh,ax,ay,az,vxbh,vybh,vzbh

  
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

  t0 = pin->GetOrAddReal("problem","t0", 1e4);

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


Real EquationOfState::GetRadius(Real x1, Real x2, Real x3,  Real a){

  Real r, th, phi;
  GetBoyerLindquistCoordinates(x1,x2,x3,0,0,a, &r, &th, &phi);
  return r;
}

Real EquationOfState::GetRadius2(Real x1, Real x2, Real x3){

  return -1.0;
  // Real xprime,yprime,zprime,rprime,Rprime;
  // get_prime_coords(2,x1,x2,x3, pmy_block_->pmy_mesh->time, &xprime,&yprime,&zprime,&rprime, &Rprime);

  // return rprime;
}
