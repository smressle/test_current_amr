//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_bondi.cpp
//  \brief Problem generator for spherically symmetric black hole accretion.

// C++ headers
#include <cmath>  // abs(), NAN, pow(), sqrt()

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>


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
#include "../utils/utils.hpp" //ran2()


// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
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
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);
static void GetCKSCoordinates(Real r, Real th, Real phi, Real *x, Real *y, Real *z);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void TransformCKSLowerVector(Real a0_cks, Real a1_cks, Real a2_cks, Real a3_cks, Real r,
                     Real theta, Real phi, Real x , Real y, Real z,Real *pa0, Real *pa1, Real *pa2, Real *pa3);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);


void  Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);


int RefinementCondition(MeshBlock *pmb);

void get_prime_coords(Real x, Real y, Real z, Real t, Real *xprime,Real *yprime,Real *zprime,Real *rprime, Real *Rprime);
void get_bh_position(Real t, Real *xbh, Real *ybh, Real *zbh);

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);



// Global variables
static Real m, a,aprime,q;          // black hole mass and spin
static Real dfloor,pfloor;                         // density and pressure floors
static Real r_inner_boundary,r_inner_boundary_2;

static Real rh;
static Real rh2;
static Real r_bh2;
static Real Omega_bh2;
static Real r_cut;        // initial condition cut off
static Real r_cut_prime;
static Real bondi_radius;  // b^2/rho at inner radius
static Real field_norm;  
static Real magnetic_field_inclination;   
static Real rho_min_,pgas_min_, rho_pow_,pgas_pow_;
Real x1_harm_max;  //maximum x1 coordinate for hyper-exponentiation   
Real cpow2,npow2;  //coordinate parameters for hyper-exponentiation
Real rbr; //break radius for hyper-exp.

int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
int max_second_bh_refinement_level = 0;  /*Maximum allowed level of refinement for AMR on secondary BH */
int max_smr_refinement_level = 0; /*Maximum allowed level of refinement for SMR on primary BH */

static Real SMALL = 1e-5;


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
// Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters

  bondi_radius  = pin->GetReal("problem", "bondi_radius");

  r_cut = pin->GetReal("problem", "r_cut");
  r_cut_prime = pin->GetReal("problem", "r_cut_prime");
  field_norm =  pin->GetReal("problem", "field_norm");

  magnetic_field_inclination = pin->GetOrAddReal("problem","field_inclination",0.0);


  max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);

  max_second_bh_refinement_level = pin->GetOrAddReal("problem","max_bh2_refinement",0);
  max_smr_refinement_level = pin->GetOrAddReal("problem","max_smr_refinement",0);

  if (max_second_bh_refinement_level>max_refinement_level) max_second_bh_refinement_level = max_refinement_level;
  if (max_smr_refinement_level>max_refinement_level) max_smr_refinement_level = max_refinement_level;



  if (max_refinement_level>0) max_refinement_level = max_refinement_level -1;
  if (max_second_bh_refinement_level>0) max_second_bh_refinement_level = max_second_bh_refinement_level -1;
  if (max_smr_refinement_level>0) max_smr_refinement_level = max_smr_refinement_level - 1;

  EnrollUserBoundaryFunction(INNER_X1, CustomInnerX1);
  EnrollUserBoundaryFunction(OUTER_X1, CustomOuterX1);
  EnrollUserBoundaryFunction(OUTER_X2, CustomOuterX2);
  EnrollUserBoundaryFunction(INNER_X2, CustomInnerX2);
  EnrollUserBoundaryFunction(OUTER_X3, CustomOuterX3);
  EnrollUserBoundaryFunction(INNER_X3, CustomInnerX3);

  EnrollUserMetric(Cartesian_GR);

  EnrollUserRadSourceFunction(inner_boundary);

  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);



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

  int N_user_vars = 6;
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }

  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1);


    // Get mass of black hole
  m = pcoord->GetMass();
  a = pcoord->GetSpin();
  q = pin->GetOrAddReal("problem", "q", 0.1);
  aprime = q * pin->GetOrAddReal("problem", "a_bh2", 0.0);
  r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);

  Real v_bh2 = 1.0/std::sqrt(r_bh2);
  Omega_bh2 = 0.0; //
  //Omega_bh2 = v_bh2/r_bh2;


  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  r_inner_boundary = rh/2.0;

  // Get mass of black hole
  Real m2 = q;

  rh2 = m2 * ( 1.0 + std::sqrt(1.0-SQR(aprime)) );

  r_inner_boundary_2 = rh2/2.0;

  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));

  rho_min_ = pin->GetOrAddReal("hydro", "rho_min", (1024*(FLT_MIN)));

  rho_pow_ = pin->GetOrAddReal("hydro", "rho_pow", 0.0);
  pgas_min_ = pin->GetOrAddReal("hydro", "pgas_min", (1024*(FLT_MIN)));
  pgas_pow_ = pin->GetOrAddReal("hydro", "pgas_pow", 0.0);


  return;
}


    static Real exp_cut_off(Real r){

      if (r<=rh) return 0.0;
      else if (r<= r_cut) return std::exp(5 * (r-r_cut)/r);
      else return 1.0;
    }

    static Real Ax_func(Real x,Real y, Real z){

      return 0.0 * field_norm;
    }
    static Real Ay_func(Real x, Real y, Real z){
      return (-z * std::sin(magnetic_field_inclination) + x * std::cos(magnetic_field_inclination) ) * field_norm;  //x 
    }
    static Real Az_func(Real x, Real y, Real z){
      return 0.0 * field_norm;
    }



static void KS_to_CKS(Real a0_ks, Real a1_ks, Real a2_ks, Real a3_ks, Real x1,
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
    Real x = x1;
    Real y = x2;
    Real z = x3;

    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
    *pa0 = a0_ks ;
    *pa1 = a1_ks * ( (r*x+a*y)/(SQR(r) + SQR(a))) + 
           a2_ks * x*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
           a3_ks * y; 
    *pa2 = a1_ks * ( (r*y-a*x)/(SQR(r) + SQR(a))) + 
           a2_ks * y*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
           a3_ks * x;
    *pa3 = a1_ks * z/r - 
           a2_ks * r * std::sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  return;
}

void get_ub(Real x, Real y, Real z, Real *ut_cks, Real *ux_cks, Real *uy_cks, Real *uz_cks){


        Real r, theta, phi;
        GetBoyerLindquistCoordinates(x,y,z,&r, &theta, &phi);

        Real delta = SQR(r) - 2.0*m*r + SQR(a);
        Real sigma = SQR(r) + SQR(a*std::cos(theta));

        Real gitt_KS = -(1.0 + 2.0*m*r/sigma);
        Real gitr_KS = 2.0*m*r/sigma;

        Real grt_KS = 2.0*m*r/sigma;
        Real grr_KS = 1.0 + 2.0*m*r/sigma;

        Real ut_KS = std::sqrt(-gitt_KS);
        Real ur_KS = -1.0/std::sqrt(-gitt_KS) * gitr_KS;


        KS_to_CKS(ut_KS,ur_KS,0,0, x,y,z, &(*ut_cks), 
          &(*ux_cks), &(*uy_cks), &(*uz_cks));

}

//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets primitive and conserved variables according to input primitives
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Parameters
  Real cs_0 = 1.0/std::sqrt(bondi_radius);
  Real rho_0 = 1.0;
  Real gam = peos->GetGamma();
  Real P_0 = SQR(cs_0)*rho_0/gam;

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

  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );



  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);

// Ensure a different initial random seed for each meshblock.
  int64_t iseed = -1 - gid;

  AthenaArray<Real> g_tmp,gi_tmp;
  g_tmp.NewAthenaArray(NMETRIC);
  gi_tmp.NewAthenaArray(NMETRIC);

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
            &r, &theta, &phi);
        // Real u0 = std::sqrt(-1.0/g(I00,i));
        // Real uu1 = 0.0 - gi(I01,i)/gi(I00,i) * u0;
        // Real uu2 = 0.0 - gi(I02,i)/gi(I00,i) * u0;
        // Real uu3 = 0.0 - gi(I03,i)/gi(I00,i) * u0;

        Real amp = 0.00;
        //if (std::fabs(a)<1e-1) amp = 0.01;
        Real rval = amp*(ran2(&iseed) - 0.5);
    
        Real xprime,yprime,zprime,rprime,Rprime;
        get_prime_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), pmy_mesh->time, &xprime,&yprime, &zprime, &rprime, &Rprime);


        Real ut_cks,ux_cks,uy_cks,uz_cks;
        get_ub(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),&ut_cks, &ux_cks, &uy_cks, &uz_cks);



      //   Real g_raised[4][4];

      //   g_raised[0][0] = g(I00,i)*gi(I00,i) + g(I01,i)*gi(I01,i) + g(I02,i)*gi(I02,i) + g(I03,i)*gi(I03,i);
      //   g_raised[0][1] = g(I00,i)*gi(I01,i) + g(I01,i)*gi(I11,i) + g(I02,i)*gi(I12,i) + g(I03,i)*gi(I13,i);
      //   g_raised[1][0] = g(I01,i)*gi(I00,i) + g(I11,i)*gi(I01,i) + g(I12,i)*gi(I02,i) + g(I13,i)*gi(I03,i);
      //   g_raised[0][2] = g(I00,i)*gi(I02,i) + g(I01,i)*gi(I12,i) + g(I02,i)*gi(I22,i) + g(I03,i)*gi(I23,i);
      //   g_raised[2][0] = g(I02,i)*gi(I00,i) + g(I12,i)*gi(I01,i) + g(I22,i)*gi(I02,i) + g(I23,i)*gi(I03,i);
      //   g_raised[0][3] = g(I00,i)*gi(I03,i) + g(I01,i)*gi(I13,i) + g(I02,i)*gi(I23,i) + g(I03,i)*gi(I33,i);
      //   g_raised[3][0] = g(I03,i)*gi(I00,i) + g(I13,i)*gi(I01,i) + g(I23,i)*gi(I02,i) + g(I33,i)*gi(I03,i);
      //   g_raised[1][1] = g(I01,i)*gi(I01,i) + g(I11,i)*gi(I11,i) + g(I12,i)*gi(I12,i) + g(I13,i)*gi(I13,i);
      //   g_raised[2][1] = g(I02,i)*gi(I01,i) + g(I12,i)*gi(I11,i) + g(I22,i)*gi(I12,i) + g(I23,i)*gi(I13,i);
      //   g_raised[1][2] = g(I01,i)*gi(I02,i) + g(I11,i)*gi(I12,i) + g(I12,i)*gi(I22,i) + g(I13,i)*gi(I23,i);     
      //   g_raised[2][2] = g(I02,i)*gi(I02,i) + g(I12,i)*gi(I12,i) + g(I22,i)*gi(I22,i) + g(I23,i)*gi(I23,i);  
      //   g_raised[2][3] = g(I02,i)*gi(I03,i) + g(I12,i)*gi(I13,i) + g(I22,i)*gi(I23,i) + g(I23,i)*gi(I33,i);
      //   g_raised[3][2] = g(I03,i)*gi(I02,i) + g(I13,i)*gi(I12,i) + g(I23,i)*gi(I22,i) + g(I33,i)*gi(I23,i);
      //   g_raised[3][1] = g(I03,i)*gi(I01,i) + g(I13,i)*gi(I11,i) + g(I23,i)*gi(I12,i) + g(I33,i)*gi(I13,i);
      //   g_raised[1][3] = g(I01,i)*gi(I03,i) + g(I11,i)*gi(I13,i) + g(I12,i)*gi(I23,i) + g(I13,i)*gi(I33,i);
      //   g_raised[3][3] = g(I03,i)*gi(I03,i) + g(I13,i)*gi(I13,i) + g(I23,i)*gi(I23,i) + g(I33,i)*gi(I33,i);

      // //   if (i==15 && j==15 && k==15){
      // //   for (int mu =0; mu<=3; ++mu){
      // //     for (int nu = 0; nu<=3; ++nu){

      // //       fprintf(stderr,"mu: %d nu: %d g_mu^nu: %g \n",mu, nu ,g_raised[mu][nu]);


      // //     }
      // //   }

      // //   fprintf(stderr,"Determinant: %g \n", Determinant(g));
      // // }

      // g_tmp(I00) = g(I00,i);
      // g_tmp(I01) = g(I01,i);
      // g_tmp(I02) = g(I02,i);
      // g_tmp(I03) = g(I03,i);
      // g_tmp(I11) = g(I11,i);
      // g_tmp(I12) = g(I12,i);
      // g_tmp(I13) = g(I13,i);
      // g_tmp(I22) = g(I22,i);
      // g_tmp(I23) = g(I23,i);
      // g_tmp(I33) = g(I33,i);
      
      // gi_tmp(I00) = gi(I00,i);
      // gi_tmp(I01) = gi(I01,i);
      // gi_tmp(I02) = gi(I02,i);
      // gi_tmp(I03) = gi(I03,i);
      // gi_tmp(I11) = gi(I11,i);
      // gi_tmp(I12) = gi(I12,i);
      // gi_tmp(I13) = gi(I13,i);
      // gi_tmp(I22) = gi(I22,i);
      // gi_tmp(I23) = gi(I23,i);
      // gi_tmp(I33) = gi(I33,i);

      // Real det = std::sqrt(-Determinant(g_tmp));
      // Real deti = std::sqrt(-Determinant(gi_tmp));
      // if (r>rh and rprime>rh2){
      //   if (std::fabs(1.0 -det)>1e-4 || std::fabs(1.0-deti) >1e-4 ) 
      //     fprintf(stderr, "Problem with determinant at r = %g th= %g phi = %g !! \n x y z: %g %g %g \ndet: %g deti: %g \n", 
      //       r,theta,phi,pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),det,deti );
      //           for (int mu =0; mu<=3; ++mu){
      //     for (int nu = 0; nu<=3; ++nu){

      //       if ( (mu==nu) && (std::fabs(g_raised[mu][nu] - 1.0)> 1e-4) ) 
      //         fprintf(stderr,"Problem with metric at r = %g !! \n mu = %d nu = %d\n g_raised: %g ", r,mu,nu,g_raised[mu][nu]);
      //       else if ( ( mu != nu) && (std::fabs(g_raised[mu][nu])>1e-4) )
      //         fprintf(stderr,"Problem with metric at r = %g !! \n mu = %d nu = %d\n g_raised: %g\n", r,mu,nu, g_raised[mu,nu]);

      //     }
      //   }
      // }



        Real uu1 = 0.0; //ux_cks + gi(I01,i)/std::abs(gi(I00,i)) * ut_cks ; // ut = gamma/alpha = sqrt(-gitt)
        Real uu2 = 0.0; //uy_cks + gi(I02,i)/std::abs(gi(I00,i)) * ut_cks ;
        Real uu3 = 0.0; //uz_cks + gi(I03,i)/std::abs(gi(I00,i)) * ut_cks ;
        if (r<r_cut || rprime<r_cut_prime){


          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho_min_  * std::pow(r,rho_pow_);
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas_min_ * std::pow(r,pgas_pow_);
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
        }
        else{ 
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho_0 * (1.0 + 2.0*rval);
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = P_0;
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }


      }
    }
  }

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {


    Real delta =0.0; // 1e-1;  //perturbation in B-field amplitude
    Real pert = 0.0;


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

            Real r,theta,phi;
            Real x_coord;
            if (i<= iu) x_coord = pcoord->x1v(i);
            else x_coord = pcoord->x1v(iu) + pcoord->dx1v(iu);
            GetBoyerLindquistCoordinates(x_coord,pcoord->x2f(j),pcoord->x3f(k), &r, &theta,&phi);

            pert = delta * std::cos(phi);
            Real x,y,z;
            GetCKSCoordinates(x_coord,pcoord->x2f(j),pcoord->x3f(k),&x,&y,&z);
            Real Ax = Ax_func(x,y,z) * (1 + pert);
            Real Ay = Ay_func(x,y,z) * (1 + pert);
            Real Az = Az_func(x,y,z) * (1 + pert);

            // Real Ar,Ath,Aphi,A0;;

            // TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A1(k,j,i) = Ax * exp_cut_off(r);

            Real y_coord;
            if (j<= ju) y_coord = pcoord->x2v(j);
            else y_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);
            GetBoyerLindquistCoordinates(pcoord->x1f(i),y_coord,pcoord->x3f(k), &r, &theta,&phi);
            pert = delta * std::cos(phi);
            GetCKSCoordinates(pcoord->x1f(i),y_coord,pcoord->x3f(k),&x,&y,&z);
            Ax = Ax_func(x,y,z) * (1 + pert);
            Ay = Ay_func(x,y,z) * (1 + pert);
            Az = Az_func(x,y,z) * (1 + pert);
            //TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A2(k,j,i) = Ay * exp_cut_off(r);

            Real z_coord;
            if (k<= ku) z_coord = pcoord->x3v(k);
            else z_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
            GetBoyerLindquistCoordinates(pcoord->x1f(i),pcoord->x2f(j),z_coord, &r, &theta,&phi);
            pert = delta * std::cos(phi);
            GetCKSCoordinates(pcoord->x1f(i),pcoord->x2f(j),z_coord,&x,&y,&z);
            Ax = Ax_func(x,y,z) * (1 + pert);
            Ay = Ay_func(x,y,z) * (1 + pert);
            Az = Az_func(x,y,z) * (1 + pert);
            //TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A3(k,j,i) = Az * exp_cut_off(r);



            }
          }
        }


      // Initialize interface fields
    AthenaArray<Real> area;
    area.NewAthenaArray(ncells1+1);

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      // reset loop limits for polar boundary
      for (int j=jl; j<=ju+1; ++j) {
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
        for (int j=jl; j<=ju+1; ++j) {
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



  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();
  g_tmp.DeleteAthenaArray();
  gi_tmp.DeleteAthenaArray();

    // Call user work function to set output variables
  UserWorkInLoop();
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


// int RefinementCondition(MeshBlock *pmb)
// {
//   int refine = 0;


//   for (int k = pmb->ks; k<=pmb->ke;k++){
//     for(int j=pmb->js; j<=pmb->je; j++) {
//       for(int i=pmb->is; i<=pmb->ie; i++) {
                


//             Real new_r_in;
//             Real x = pmb->pcoord->x1v(i);
//             Real y = pmb->pcoord->x2v(j);
//             Real z = pmb->pcoord->x3v(k);

//             Real xprime,yprime,zprime,rprime, Rprime;
//             Real t = pmb->pmy_mesh->time;

//             get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime, &Rprime);


//              if (rprime<rh2*10.0)  return  1;
//               }
//             }
//           }
 
//   return -1;
// }

int RefinementCondition(MeshBlock *pmb)
{
  int refine = 0;

    Real DX,DY,DZ;
    Real dx,dy,dz;
  get_uniform_box_spacing(pmb->pmy_mesh->mesh_size,&DX,&DY,&DZ);
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);


  Real total_box_radius = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2.0;
  Real bh2_focus_radius = 3.125*q;
  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  if (current_level >=max_refinement_level) return 0;

  int any_at_current_level=0;


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
              if (current_level < n_level){

                  return  1;
              }
            }

            if (current_level==n_level) any_at_current_level=1;

          
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
              if (current_level < n_level){
                  //fprintf(stderr,"current level: %d n_level: %d box_radius: %g \n xmin: %g ymin: %g zmin: %g xmax: %g ymax: %g zmax: %g\n",current_level,
                    //n_level,box_radius,pmb->block_size.x1min,pmb->block_size.x2min,pmb->block_size.x3min,pmb->block_size.x1max,pmb->block_size.x2max,pmb->block_size.x3max);
                  return  1;
              }
            }


            if (current_level==n_level) any_at_current_level=1;

          
          }

  }
 }
}

if (any_at_current_level==1) return 0;
  return -1;
}


/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){


  Real r,th,ph;
  AthenaArray<Real> g, gi;
  g.InitWithShallowCopy(pmb->ruser_meshblock_data[0]);
  gi.InitWithShallowCopy(pmb->ruser_meshblock_data[1]);



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

g.DeleteAthenaArray();
gi.DeleteAthenaArray();



}
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{
  int i, j, k, kprime;
  int is, ie, js, je, ks, ke;


  apply_inner_boundary_condition(pmb,prim);

  return;
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
  AthenaArray<Real> g, gi;
  g.InitWithShallowCopy(ruser_meshblock_data[0]);
  gi.InitWithShallowCopy(ruser_meshblock_data[1]);


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
static void GetCKSCoordinates(Real r, Real th, Real phi, Real *x, Real *y, Real *z){
  *x = r;
  *y = th ;
  *z = phi; 

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

static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  if (COORDINATE_SYSTEM == "schwarzschild") {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (COORDINATE_SYSTEM == "kerr-schild") {
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl + a/delta * a1_bl;
  }
  return;
}

static void TransformCKSLowerVector(Real a0_cks, Real a1_cks, Real a2_cks, Real a3_cks, Real r,
                     Real theta, Real phi, Real x , Real y, Real z,Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

    *pa0 = a0_cks ; 
    *pa1 =         (std::sin(theta)*std::cos(phi)) * a1_cks 
                 + (std::sin(theta)*std::sin(phi)) * a2_cks 
                 + (std::cos(theta)              ) * a3_cks ;

    *pa2 =         (std::cos(theta) * (r*std::cos(phi) + a*std::sin(phi) ) ) * a1_cks
                 + (std::cos(theta) * (r*std::sin(phi) - a*std::cos(phi) ) ) * a2_cks
                 + (-r*std::sin(theta)                                     ) * a3_cks;

    *pa3 =          -y * a1_cks 
                + (x) * a2_cks;


  return;
}


void get_bh_position(Real t, Real *xbh, Real *ybh, Real *zbh){

  *xbh = 0.0;
  *ybh = r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
  *zbh = r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);


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
  Real q = pin->GetOrAddReal("problem", "q", 0.1);
  Real aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check


 // Real t = 10000;
    // Position of black hole

  Real x = x1;
  Real y = x2;
  Real z = x3;

  if (std::fabs(z)<SMALL) z= SMALL;

  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x=  0.1;
    y = 0.1;
    z = 0.1;
  }

  Real r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);
  Real v_bh2 = 1.0/std::sqrt(r_bh2);
  Real Omega_bh2 = v_bh2/r_bh2;
  Real x_bh2 = 0.0;
  Real y_bh2 = r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
  Real z_bh2 = r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);

  Real xprime = x - x_bh2;
  Real yprime = y - y_bh2;
  Real zprime = z - z_bh2;



  Real dx_bh2_dt = 0.0;
  Real dy_bh2_dt =  2.0*PI*Omega_bh2 * r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);
  Real dz_bh2_dt = -2.0*PI*Omega_bh2 * r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
  if (std::fabs(zprime)<SMALL) zprime= SMALL;
  Real Rprime = std::sqrt(SQR(xprime) + SQR(yprime) + SQR(zprime));
  Real rprime = SQR(Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(zprime) );
  rprime = std::sqrt(rprime/2.0);



/// prevent metric from getting nan sqrt(-gdet)
  Real thprime  = std::acos(zprime/rprime);
  Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
                              (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

  Real rhprime = q * ( 1.0 + std::sqrt(1.0-SQR(aprime)) );
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
void delta_cks_metric_inverse(ParameterInput *pin,Real t, Real x1, Real x2, Real x3,AthenaArray<Real> &delta_g_inv){
  Real q = pin->GetOrAddReal("problem", "q", 0.1);
  Real aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check

  Real x = x1;
  Real y = x2;
  Real z = x3;

  if (std::fabs(z)<SMALL) z= SMALL;

  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x=  0.1;
    y = 0.1;
    z = 0.1;
  }

 // Real t = 10000;
    // Position of black hole

  Real r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);
  Real v_bh2 = 1.0/std::sqrt(r_bh2);
  Real Omega_bh2 = v_bh2/r_bh2;
  Real x_bh2 = 0.0;
  Real y_bh2 = r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
  Real z_bh2 = r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);

  Real xprime = x - x_bh2;
  Real yprime = y - y_bh2;
  Real zprime = z - z_bh2;


  Real dx_bh2_dt = 0.0;
  Real dy_bh2_dt =  2.0*PI*Omega_bh2 * r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);
  Real dz_bh2_dt = -2.0*PI*Omega_bh2 * r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);
  if (std::fabs(zprime)<SMALL) zprime= SMALL;
  Real Rprime = std::sqrt(SQR(xprime) + SQR(yprime) + SQR(zprime));
  Real rprime = SQR(Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(zprime) );
  rprime = std::sqrt(rprime/2.0);



/// prevent metric from gettin nan sqrt(-gdet)
  Real thprime  = std::acos(zprime/rprime);
  Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
                              (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

  Real rhprime = q * ( 1.0 + std::sqrt(1.0-SQR(aprime)) );
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
  delta_g_inv(I00) = -fprime * l_upperprime[0]*l_upperprime[0];
  delta_g_inv(I01) = -fprime * l_upperprime[0]*l_upperprime[1];
  delta_g_inv(I02) = -fprime * l_upperprime[0]*l_upperprime[2];
  delta_g_inv(I03) = -fprime * l_upperprime[0]*l_upperprime[3];
  delta_g_inv(I11) = -fprime * l_upperprime[1]*l_upperprime[1];
  delta_g_inv(I12) = -fprime * l_upperprime[1]*l_upperprime[2];
  delta_g_inv(I13) = -fprime * l_upperprime[1]*l_upperprime[3];
  delta_g_inv(I22) = -fprime * l_upperprime[2]*l_upperprime[2];
  delta_g_inv(I23) = -fprime * l_upperprime[2]*l_upperprime[3];
  delta_g_inv(I33) = -fprime * l_upperprime[3]*l_upperprime[3];

}


#define DEL 1e-7
void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  a = pin->GetReal("coord", "a");
  Real a_spin =a;

  if ((std::fabs(z)<SMALL) && (z>=0)) z= SMALL;
  if ((std::fabs(z)<SMALL) && (z<0)) z= -SMALL;

  // if ((std::fabs(x)<SMALL) && (x>=0)) x= SMALL;
  // if ((std::fabs(x)<SMALL) && (x<0)) x= -SMALL;

  // if ((std::fabs(y)<SMALL) && (y>=0)) y= SMALL;
  // if ((std::fabs(y)<SMALL) && (y<0)) y= -SMALL;  

  if ( (std::fabs(x)<0.1) && (std::fabs(y)<0.1) && (std::fabs(z)<0.1) ){
    x= 0.1;
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



  //////////////Perturber Black Hole//////////////////

  q = pin->GetOrAddReal("problem", "q", 1.0);
  aprime= q * pin->GetOrAddReal("problem", "a_bh2", 0.0);  //I think this factor of q is right..check


 // Real t = 10000;
    // Position of black hole

  r_bh2 = pin->GetOrAddReal("problem", "r_bh2", 20.0);
  Real v_bh2 = 1.0/std::sqrt(r_bh2);
  //Omega_bh2 = v_bh2/r_bh2;
  Omega_bh2 = 0.0;

  Real xprime,yprime,zprime,rprime,Rprime;
  get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);


  Real dx_bh2_dt = 0.0;
  Real dy_bh2_dt =  2.0*PI*Omega_bh2 * r_bh2 * std::cos(2.0*PI*Omega_bh2 * t);
  Real dz_bh2_dt = -2.0*PI*Omega_bh2 * r_bh2 * std::sin(2.0*PI*Omega_bh2 * t);




/// prevent metric from gettin nan sqrt(-gdet)
  Real thprime  = std::acos(zprime/rprime);
  Real phiprime = std::atan2( (rprime*yprime-aprime*xprime)/(SQR(rprime) + SQR(aprime) ), 
                              (aprime*yprime+rprime*xprime)/(SQR(rprime) + SQR(aprime) )  );

  Real rhprime = q * ( 1.0 + std::sqrt(1.0-SQR(aprime)) );
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
  g(I00) = eta[0] + f * l_lower[0]*l_lower[0] + fprime * l_lowerprime[0]*l_lowerprime[0];
  g(I01) =          f * l_lower[0]*l_lower[1] + fprime * l_lowerprime[0]*l_lowerprime[1];
  g(I02) =          f * l_lower[0]*l_lower[2] + fprime * l_lowerprime[0]*l_lowerprime[2];
  g(I03) =          f * l_lower[0]*l_lower[3] + fprime * l_lowerprime[0]*l_lowerprime[3];
  g(I11) = eta[1] + f * l_lower[1]*l_lower[1] + fprime * l_lowerprime[1]*l_lowerprime[1];
  g(I12) =          f * l_lower[1]*l_lower[2] + fprime * l_lowerprime[1]*l_lowerprime[2];
  g(I13) =          f * l_lower[1]*l_lower[3] + fprime * l_lowerprime[1]*l_lowerprime[3];
  g(I22) = eta[2] + f * l_lower[2]*l_lower[2] + fprime * l_lowerprime[2]*l_lowerprime[2];
  g(I23) =          f * l_lower[2]*l_lower[3] + fprime * l_lowerprime[2]*l_lowerprime[3];
  g(I33) = eta[3] + f * l_lower[3]*l_lower[3] + fprime * l_lowerprime[3]*l_lowerprime[3];




  // // Set contravariant components
  g_inv(I00) = eta[0] - f * l_upper[0]*l_upper[0] - fprime * l_upperprime[0]*l_upperprime[0];
  g_inv(I01) =        - f * l_upper[0]*l_upper[1] - fprime * l_upperprime[0]*l_upperprime[1];
  g_inv(I02) =        - f * l_upper[0]*l_upper[2] - fprime * l_upperprime[0]*l_upperprime[2];
  g_inv(I03) =        - f * l_upper[0]*l_upper[3] - fprime * l_upperprime[0]*l_upperprime[3];
  g_inv(I11) = eta[1] - f * l_upper[1]*l_upper[1] - fprime * l_upperprime[1]*l_upperprime[1];
  g_inv(I12) =        - f * l_upper[1]*l_upper[2] - fprime * l_upperprime[1]*l_upperprime[2];
  g_inv(I13) =        - f * l_upper[1]*l_upper[3] - fprime * l_upperprime[1]*l_upperprime[3];
  g_inv(I22) = eta[2] - f * l_upper[2]*l_upper[2] - fprime * l_upperprime[2]*l_upperprime[2];
  g_inv(I23) =        - f * l_upper[2]*l_upper[3] - fprime * l_upperprime[2]*l_upperprime[3];
  g_inv(I33) = eta[3] - f * l_upper[3]*l_upper[3] - fprime * l_upperprime[3]*l_upperprime[3];


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

  sqrt_term =  2.0*SQR(rprime)-SQR(Rprime) + SQR(aprime);
  rsq_p_asq = SQR(rprime) + SQR(aprime);

  Real dfprime_dx1 = q * SQR(fprime/q)*xprime/(2.0*std::pow(rprime,3)) * 
                      ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
  //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
  Real dfprime_dx2 = q * SQR(fprime/q)*yprime/(2.0*std::pow(rprime,3)) * 
                      ( ( 3.0*SQR(aprime*zprime)-SQR(rprime)*SQR(rprime) ) )/ sqrt_term ;
  Real dfprime_dx3 = q * SQR(fprime/q)*zprime/(2.0*std::pow(rprime,5)) * 
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
  dgprime_dx1(I00) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx1 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx1;
  dgprime_dx1(I01) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx1;
  dgprime_dx1(I02) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx1;
  dgprime_dx1(I03) = dfprime_dx1*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx1;
  dgprime_dx1(I11) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx1 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx1;
  dgprime_dx1(I12) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx1;
  dgprime_dx1(I13) = dfprime_dx1*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx1;
  dgprime_dx1(I22) = dfprime_dx1*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx1 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx1;
  dgprime_dx1(I23) = dfprime_dx1*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx1;
  dgprime_dx1(I33) = dfprime_dx1*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx1 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx1;

  // Set y-derivatives of covariant components
  dgprime_dx2(I00) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx2 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx2;
  dgprime_dx2(I01) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx2;
  dgprime_dx2(I02) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx2;
  dgprime_dx2(I03) = dfprime_dx2*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx2;
  dgprime_dx2(I11) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx2 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx2;
  dgprime_dx2(I12) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx2;
  dgprime_dx2(I13) = dfprime_dx2*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx2;
  dgprime_dx2(I22) = dfprime_dx2*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx2 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx2;
  dgprime_dx2(I23) = dfprime_dx2*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx2;
  dgprime_dx2(I33) = dfprime_dx2*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx2 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx2;

  // Set z-derivatives of covariant components
  dgprime_dx3(I00) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx3 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx3;
  dgprime_dx3(I01) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx3;
  dgprime_dx3(I02) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx3;
  dgprime_dx3(I03) = dfprime_dx3*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx3;
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
  dg_dt(I01) = -1.0 * (dx_bh2_dt * dgprime_dx1(I01) + dy_bh2_dt * dgprime_dx2(I01) + dz_bh2_dt * dgprime_dx3(I01) );;
  dg_dt(I02) = -1.0 * (dx_bh2_dt * dgprime_dx1(I02) + dy_bh2_dt * dgprime_dx2(I02) + dz_bh2_dt * dgprime_dx3(I02) );;
  dg_dt(I03) = -1.0 * (dx_bh2_dt * dgprime_dx1(I03) + dy_bh2_dt * dgprime_dx2(I03) + dz_bh2_dt * dgprime_dx3(I03) );;
  dg_dt(I11) = -1.0 * (dx_bh2_dt * dgprime_dx1(I11) + dy_bh2_dt * dgprime_dx2(I11) + dz_bh2_dt * dgprime_dx3(I11) );;
  dg_dt(I12) = -1.0 * (dx_bh2_dt * dgprime_dx1(I12) + dy_bh2_dt * dgprime_dx2(I12) + dz_bh2_dt * dgprime_dx3(I12) );;
  dg_dt(I13) = -1.0 * (dx_bh2_dt * dgprime_dx1(I13) + dy_bh2_dt * dgprime_dx2(I13) + dz_bh2_dt * dgprime_dx3(I13) );;
  dg_dt(I22) = -1.0 * (dx_bh2_dt * dgprime_dx1(I22) + dy_bh2_dt * dgprime_dx2(I22) + dz_bh2_dt * dgprime_dx3(I22) );;
  dg_dt(I23) = -1.0 * (dx_bh2_dt * dgprime_dx1(I23) + dy_bh2_dt * dgprime_dx2(I23) + dz_bh2_dt * dgprime_dx3(I23) );;
  dg_dt(I33) = -1.0 * (dx_bh2_dt * dgprime_dx1(I33) + dy_bh2_dt * dgprime_dx2(I33) + dz_bh2_dt * dgprime_dx3(I33) );;


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