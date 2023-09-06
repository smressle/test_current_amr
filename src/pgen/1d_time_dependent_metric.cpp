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


void  Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);

static Real Determinant(const AthenaArray<Real> &g);
static Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
    Real a31, Real a32, Real a33);
static Real Determinant(Real a11, Real a12, Real a21, Real a22);
bool gluInvertMatrix(AthenaArray<Real> &m, AthenaArray<Real> &inv);

void get_t_from_prime(Real tprime,Real xprime,Real yprime,Real zprime,Real *t);
void Binary_BH_Metric(Real t, Real x1, Real x2, Real x3,
  AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt);

Real v_func(Real t);
Real acc_func(Real t);

// Global variables
static Real dfloor,pfloor;                         // density and pressure floors
static Real vmax, Omega;
// static Real rh;                                    // horizon radius


static Real SMALL = 1e-5;

#define EP 1e-11


void get_Lambda(Real t, Real x, Real Lambda[2][2],Real Lambda_inverse[2][2]){

  Real v = v_func(t);
  Real Lorentz = 1.0/std::sqrt(1.0-SQR(v));
  Real acc = acc_func(t);

  Lambda[0][0] =  Lorentz + SQR(Lorentz)*Lorentz * acc * (v*t-x);  //dtprime/dt
  Lambda[0][1] =  - Lorentz * v;
  Lambda[1][1] = Lorentz;
  Lambda[1][0] = - Lorentz * v + SQR(Lorentz)*Lorentz * acc * (v*acc*x -t);

  Real det = (Lambda[0][0] * Lambda[1][1]) - (Lambda[0][1] * Lambda[1][0]);

  Lambda_inverse[0][0] = Lambda[1][1] / det;
  Lambda_inverse[0][1] = -Lambda[0][1] / det;
  Lambda_inverse[1][0] = -Lambda[1][0] / det;
  Lambda_inverse[1][1] = Lambda[0][0] / det;

  return;
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


  // // Enroll boundary functions

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) EnrollUserBoundaryFunction(BoundaryFace::inner_x1, CustomInnerX1);
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) EnrollUserBoundaryFunction(BoundaryFace::outer_x1, CustomOuterX1);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x2, CustomOuterX2);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x2, CustomInnerX2);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x3, CustomOuterX3);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x3, CustomInnerX3);



  vmax = 0.1;
  Real period = 10.0;
  Omega = 2.0 * PI / (period);

    //Enroll metric
  EnrollUserMetric(Cartesian_GR);

  if (METRIC_EVOLUTION)  EnrollUserMetricWithoutPin(Binary_BH_Metric);

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


  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1+NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1+NGHOST);


  int N_user_vars = 11;
  AllocateUserOutputVariables(N_user_vars);


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



        // Calculate background primitives
        Real rho = 1.0;
        Real pgas = 1.0;


        Real ut = 1.0;
        Real ux = 0.0;
        Real uy = 0.0;
        Real uz = 0.0;

        Real t;
        Real xprime = pcoord->x1v(i);
        Real tprime = pmy_mesh->time;
        get_t_from_prime(tprime,xprime,pcoord->x2v(j), pcoord->x3v(k),&t);

        Real v = v_func(t);
        Real Lorentz = 1.0/std::sqrt(1.0-SQR(v));
        Real acc = acc_func(t);

        Real x = Lorentz * (xprime + v * tprime);

        Real Lambda_inverse[2][2],Lambda[2][2];
        get_Lambda(t,x, Lambda,Lambda_inverse);



        Real u0 = Lambda[0][0] * ut + Lambda[0][1] * ux;
        Real u1 = Lambda[1][0] * ut + Lambda[1][1] * ux;
        Real u2 = uy; 
        Real u3 = uz;
        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;
        


        if (pcoord->x1v(i)>=-0.5 and pcoord->x1v(i)<=0.5) rho =  (1.0+ std::cos(2.0*PI*pcoord->x1v(i)/1.0))*0.5+1.0;
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
  g_tmp.DeleteAthenaArray();
  gi_tmp.DeleteAthenaArray();


  AthenaArray<Real> &g_ = ruser_meshblock_data[0];
  AthenaArray<Real> &gi_ = ruser_meshblock_data[1];




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



void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // Prepare scratch arrays
  AthenaArray<Real> &g = ruser_meshblock_data[0];
  AthenaArray<Real> &gi = ruser_meshblock_data[1];

  AthenaArray<Real> g_tmp;
  g_tmp.NewAthenaArray(NMETRIC);
  // Go through all cells
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pcoord->CellMetric(k, j, is, ie, g, gi);
      for (int i = is; i <= ie; ++i) {
        // Calculate normal-frame Lorentz factor
        user_out_var(0,k,j,i) = g(I00,i); 
        user_out_var(1,k,j,i) = g(I01,i); 
        user_out_var(2,k,j,i) = g(I02,i); 
        user_out_var(3,k,j,i) = g(I03,i); 
        user_out_var(4,k,j,i) = g(I11,i); 
        user_out_var(5,k,j,i) = g(I12,i); 
        user_out_var(6,k,j,i) = g(I13,i); 
        user_out_var(7,k,j,i) = g(I22,i); 
        user_out_var(8,k,j,i) = g(I23,i); 
        user_out_var(9,k,j,i) = g(I33,i); 

        for (int n=0; n<NMETRIC; ++n) g_tmp(n) = g(n,i);
        user_out_var(10,k,j,i) = Determinant(g_tmp);
      }
    }
  }

  g_tmp.DeleteAthenaArray();
  return;
}

void  MeshBlock::PreserveDivbNewMetric(ParameterInput *pin){
return;
}




Real v_func(Real t){
  return vmax * std::sin(Omega * t);
}
Real acc_func(Real t){
  return Omega * vmax * std::cos(Omega * t);
}



  Real func(Real tprime_, Real xprime_,Real t_){
    Real v = v_func(t_);
    Real Lorentz = 1.0/std::sqrt(1.0-SQR(v));
    return Lorentz * (tprime_ + v * xprime_) - t_;
  }
void get_t_from_prime(Real tprime,Real xprime,Real yprime,Real zprime,Real *t){

  Real Lorentz_max = 1.0/std::sqrt(1.0-SQR(vmax));
  Real b = Lorentz_max * (tprime + std::abs(vmax * xprime)) + SMALL;
  // Real a = Lorentz_max * (tprime - std::abs(vmax * xprime)) - SMALL;
  Real a = tprime - Lorentz_max * std::abs(vmax * xprime) - SMALL;

  if (func(tprime,xprime,a) * func(tprime,xprime,b) >= 0) {
      fprintf(stderr,"BAD a and b!! \n a: %g b: %g tprime: %g xprime: %g vmax: %g \n", a,b,tprime,xprime,vmax);
      exit(0);
   }
   Real c = a;
   while ((b-a) >= EP) {
      // Find middle point
      c = (a+b)/2;
      // Check if middle point is root
      if (func(tprime,xprime,c) == 0.0)
         break;
       // Decide the side to repeat the steps
      else if (func(tprime,xprime,c)*func(tprime,xprime,a) < 0)
         b = c;
      else
         a = c;
   }
   //cout << "The value of root is : " << c;



   *t = c;
   return;


}

void Cartesian_GR(Real t, Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{


  //////////////Perturber Black Hole//////////////////

  Binary_BH_Metric(t,x1,x2,x3,g,g_inv,dg_dx1,dg_dx2,dg_dx3,dg_dt);

  return;

}

void metric_for_derivatives(Real tprime, Real x1prime, Real x2prime, Real x3prime,
    AthenaArray<Real> &g)
{
  Real xprime = x1prime;
  Real yprime = x2prime;
  Real zprime = x3prime;

  Real t;
  get_t_from_prime(tprime,xprime,yprime,zprime,&t);

  Real v = v_func(t);
  Real Lorentz = 1.0/std::sqrt(1.0-SQR(v));
  Real acc = acc_func(t);

  Real x = Lorentz * (xprime + v * tprime);

  Real eta[2][2];

  eta[0][0] = -1.0;
  eta[0][1] = 0.0;
  eta[1][0] = 0.0;
  eta[1][1] = 1.0;

  Real Lambda_inverse[2][2], Lambda[2][2];
  get_Lambda(t, x, Lambda,Lambda_inverse);

  //g_mu_nu = eta_alpha_beta Lambda_inverse[alpha][mu] Lambda_inverse[beta][nu]


  g(I00) = 0.0;
  g(I01) = 0.0;
  g(I11) = 0.0;
  for (int alpha = 0; alpha <= 2; ++alpha) {
    for (int beta = 0; beta <= 2; ++beta) {
      
      g(I00) += eta[alpha][beta] * Lambda_inverse[alpha][0] * Lambda_inverse[beta][0];
      g(I01) += eta[alpha][beta] * Lambda_inverse[alpha][0] * Lambda_inverse[beta][1];
      g(I11) += eta[alpha][beta] * Lambda_inverse[alpha][1] * Lambda_inverse[beta][1];

      }
  }

  // Set covariant components
  g(I02) = 0.0;  
  g(I03) = 0.0;
  g(I12) = 0.0;
  g(I13) = 0.0;
  g(I22) = 1.0;
  g(I23) = 0.0;
  g(I33) = 1.0;




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
void Binary_BH_Metric(Real tprime, Real x1prime, Real x2prime, Real x3prime,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3, AthenaArray<Real> &dg_dt)
{


  metric_for_derivatives(tprime,x1prime,x2prime,x3prime,g);

  bool invertible = gluInvertMatrix(g,g_inv);

  if (invertible==false) {
    fprintf(stderr,"Non-invertible matrix at xyz: %g %g %g\n", x1prime,x2prime,x3prime);
  }



  AthenaArray<Real> gp,gm;


  // Real det = Determinant(g);
  // if (det>=0){
  //   fprintf(stderr, "sqrt -g is nan!! xyz: %g %g %g xyzbh: %g %g %g \n",x,y,z,orbit_quantities(IX2),orbit_quantities(IY2),orbit_quantities(IZ2));
  //   exit(0);
  // }


  gp.NewAthenaArray(NMETRIC);
  // gm.NewAthenaArray(NMETRIC);

  Real x1p = x1prime + DEL; // * rprime;
  // Real x1m = x1 - DEL; // * rprime;
  Real x1m = x1prime;

  metric_for_derivatives(tprime,x1p,x2prime,x3prime,gp);
  // metric_for_derivatives(t,x1m,x2,x3,gm);

    // // Set x-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx1(n) = (gp(n)-gm(n))/(x1p-x1m);
  // }
    for (int n = 0; n < NMETRIC; ++n) {
     dg_dx1(n) = (gp(n)-g(n))/(x1p-x1m);
  }

  Real x2p = x2prime + DEL; // * rprime;
  // Real x2m = x2 - DEL; // * rprime;
  Real x2m = x2prime;

  metric_for_derivatives(tprime,x1prime,x2p,x3prime,gp);
  // metric_for_derivatives(t,x1,x2m,x3,gm);
    // // Set y-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx2(n) = (gp(n)-gm(n))/(x2p-x2m);
  // }
  for (int n = 0; n < NMETRIC; ++n) {
     dg_dx2(n) = (gp(n)-g(n))/(x2p-x2m);
  }
  
  Real x3p = x3prime + DEL; // * rprime;
  // Real x3m = x3 - DEL; // * rprime;
  Real x3m = x3prime;

  metric_for_derivatives(tprime,x1prime,x2prime,x3p,gp);
  // metric_for_derivatives(t,x1,x2,x3m,gm);

    // // Set z-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dx3(n) = (gp(n)-gm(n))/(x3p-x3m);
  // }
    for (int n = 0; n < NMETRIC; ++n) {
     dg_dx3(n) = (gp(n)-g(n))/(x3p-x3m);
  }

  Real tp = tprime + DEL ;
  Real tm = tprime;
  // Real tm = t - DEL ;

  metric_for_derivatives(tp,x1prime,x2prime,x3prime,gp);

  // get_orbit_quantities(tm,orbit_quantities);
  // metric_for_derivatives(tm,x1,x2,x3,gm);
    // // Set t-derivatives of covariant components
  // for (int n = 0; n < NMETRIC; ++n) {
  //    dg_dt(n) = (gp(n)-gm(n))/(tp-tm);
  // }
  for (int n = 0; n < NMETRIC; ++n) {
     dg_dt(n) = (gp(n)-g(n))/(tp-tm);
  }

  gp.DeleteAthenaArray();
  // gm.DeleteAthenaArray();

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
  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, ngh+2);
  gi.NewAthenaArray(NMETRIC,ngh+2);
  // g.NewAthenaArray(NMETRIC);
  // gi.NewAthenaArray(NMETRIC);
  // Initialize primitive values
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pco->CellMetric(k, j, is-ngh,is-1, g, gi);
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {


        Real rho = 1.0;
        Real pgas = 1.0;
        Real ut = 1.0;
        Real ux = 0.0;
        Real uy = 0.0;
        Real uz = 0.0;

        Real t;
        Real xprime = pco->x1v(is-i);

        Real tprime = time;


        
        // if (std::abs((time - (pmb->pmy_mesh->time + dt))) < 1e-2*dt && pmy_mesh->update_metric_this_timestep) {
        //   tprime = time;
        //   // fprintf(stderr,"end of stage in boundary!! \n t: %g old_t: %g dt: %g t_end_step: %g \n",time,pmb->pmy_mesh->time,dt, pmb->pmy_mesh->time + dt);
        // }
        // else{
        //   tprime = pmb->pmy_mesh->metric_time;
        // }

        // metric_for_derivatives(tprime,xprime,pco->x2v(j), pco->x3v(k),g);

        // bool invertible = gluInvertMatrix(g,gi);

        get_t_from_prime(tprime,xprime,pco->x2v(j), pco->x3v(k),&t);

        Real v = v_func(t);
        Real Lorentz = 1.0/std::sqrt(1.0-SQR(v));
        Real acc = acc_func(t);

        Real x = Lorentz * (xprime + v * tprime);

        Real Lambda_inverse[2][2],Lambda[2][2];
        get_Lambda(t,x, Lambda,Lambda_inverse);



        Real u0 = Lambda[0][0] * ut + Lambda[0][1] * ux;
        Real u1 = Lambda[1][0] * ut + Lambda[1][1] * ux;
        Real u2 = uy; 
        Real u3 = uz;
        Real uu1 = u1 - gi(I01,is-i)/gi(I00,is-i) * u0;
        Real uu2 = u2 - gi(I02,is-i)/gi(I00,is-i) * u0;
        Real uu3 = u3 - gi(I03,is-i)/gi(I00,is-i) * u0;
        // Real uu1 = u1 - gi(I01)/gi(I00) * u0;
        // Real uu2 = u2 - gi(I02)/gi(I00) * u0;
        // Real uu3 = u3 - gi(I03)/gi(I00) * u0;

        prim(IDN,k,j,is-i) = rho;
        prim(IPR,k,j,is-i) = pgas;
        prim(IVX,k,j,is-i) = uu1;
        prim(IVY,k,j,is-i) = uu2;
        prim(IVZ,k,j,is-i) = uu3;


      }
    }
  }

    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();
  

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
  
  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, ie+ngh+1);
  gi.NewAthenaArray(NMETRIC,ie+ngh+1);
  // g.NewAthenaArray(NMETRIC);
  // gi.NewAthenaArray(NMETRIC);
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pco->CellMetric(k, j, ie+1,ie+ngh, g, gi);
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {

        Real rho = 1.0;
        Real pgas = 1.0;
        Real ut = 1.0;
        Real ux = 0.0;
        Real uy = 0.0;
        Real uz = 0.0;

        Real t;

        // Real tprime;
        
        // if (std::abs((time - (pmb->pmy_mesh->time + dt))) < 1e-2*dt && pmy_mesh->update_metric_this_timestep) {
        //   tprime = time;
        //   // fprintf(stderr,"end of stage in boundary!! \n t: %g old_t: %g dt: %g t_end_step: %g \n",time,pmb->pmy_mesh->time,dt, pmb->pmy_mesh->time + dt);
        // }
        // else{
        //   tprime = pmb->pmy_mesh->metric_time;
        // }
        Real tprime = time;
        Real xprime = pco->x1v(ie+i);

        // metric_for_derivatives(tprime,xprime,pco->x2v(j), pco->x3v(k),g);

        // bool invertible = gluInvertMatrix(g,gi);

        get_t_from_prime(tprime,xprime,pco->x2v(j), pco->x3v(k),&t);

        Real v = v_func(t);
        Real Lorentz = 1.0/std::sqrt(1.0-SQR(v));
        Real acc = acc_func(t);

        Real x = Lorentz * (xprime + v * tprime);

        Real Lambda_inverse[2][2],Lambda[2][2];
        get_Lambda(t,x, Lambda,Lambda_inverse);


        Real u0 = Lambda[0][0] * ut + Lambda[0][1] * ux;
        Real u1 = Lambda[1][0] * ut + Lambda[1][1] * ux;
        Real u2 = uy; 
        Real u3 = uz;
        Real uu1 = u1 - gi(I01,ie+i)/gi(I00,ie+i) * u0;
        Real uu2 = u2 - gi(I02,ie+i)/gi(I00,ie+i) * u0;
        Real uu3 = u3 - gi(I03,ie+i)/gi(I00,ie+i) * u0;
        // Real uu1 = u1 - gi(I01)/gi(I00) * u0;
        // Real uu2 = u2 - gi(I02)/gi(I00) * u0;
        // Real uu3 = u3 - gi(I03)/gi(I00) * u0;

        prim(IDN,k,j,ie+i) = rho;
        prim(IPR,k,j,ie+i) = pgas;
        prim(IVX,k,j,ie+i) = uu1;
        prim(IVY,k,j,ie+i) = uu2;
        prim(IVZ,k,j,ie+i) = uu3;

        // fprintf(stderr,"xprime: %g tprime: %g x: %g t: %g \n v: %g Lorentz: %g gi01: %g gi00: %g \n u0: %g u1: %g \n Lambda: %g %g %g %g \n vel1: %g \n",
        //   xprime,tprime,x,t,v,Lorentz,gi(I01,ie+i),gi(I00,ie+i),u0,u1,Lambda[0][0],Lambda[0][1],Lambda[1][0],Lambda[1][1],uu1);

      }
    }}

    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();

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
