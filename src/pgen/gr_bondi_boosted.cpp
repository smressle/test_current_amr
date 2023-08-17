//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

// C headers

// C++ headers
#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()

// Athena++ headers
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput



#include <cfloat>

// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void CustomInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomInnerX3(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) ;
void CustomOuterX3(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar);
void apply_inner_boundary_condition_in_boundary_function(MeshBlock *pmb,Coordinates *pcoord, AthenaArray<Real> &prim,FaceField &b, Real time,
                    int is, int ie, int js, int je, int ks, int ke,int ngh);

void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar);
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
// namespace {
void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                  Real *ptheta, Real *pphi);
void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real x1,
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
void BoostVector(Real a0, Real a1, Real a2, Real a3, Real x1,
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
void CalculatePrimitives(Real r, Real temp_min, Real temp_max, Real *prho,
                         Real *ppgas, Real *put, Real *pur);
Real TemperatureMin(Real r, Real t_min, Real t_max);
Real TemperatureBisect(Real r, Real t_min, Real t_max);
Real TemperatureResidual(Real t, Real r);

// Global variables
Real m, a;          // black hole mass and spin
Real n_adi, k_adi;  // hydro parameters
Real r_crit;        // sonic point radius
Real c1, c2;        // useful constants
Real bsq_over_rho;  // b^2/rho at inner radius

static Real dfloor,pfloor;                         // density and pressure floors
static Real rh;   

Real temp_max,temp_min;

Real aprime,q,m2;          // black hole mass and spin
Real r_inner_boundary,r_inner_boundary_2;
Real r_inner_bondi_boundary,r_outer_bondi_boundary;
Real rh2;
Real v_bh2;
Real Omega_bh2;
int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
int max_second_bh_refinement_level = 0;  /*Maximum allowed level of refinement for AMR on secondary BH */
int max_smr_refinement_level = 0; /*Maximum allowed level of refinement for SMR on primary BH */

static Real SMALL = 1e-5;
// } // namespace


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


  temp_min = 1.0e-2;  // lesser temperature root must be greater than this
  temp_max = 1.0e1;   // greater temperature root must be less than this

  // Read problem parameters
  k_adi = pin->GetReal("hydro", "k_adi");
  r_crit = pin->GetReal("problem", "r_crit");
  bsq_over_rho = 0.0;
  if (MAGNETIC_FIELDS_ENABLED) {
    bsq_over_rho = pin->GetReal("problem", "bsq_over_rho");
  }


  max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);

  max_second_bh_refinement_level = pin->GetOrAddReal("problem","max_bh2_refinement",0);
  max_smr_refinement_level = pin->GetOrAddReal("problem","max_smr_refinement",0);

  if (max_second_bh_refinement_level>max_refinement_level) max_second_bh_refinement_level = max_refinement_level;
  if (max_smr_refinement_level>max_refinement_level) max_smr_refinement_level = max_refinement_level;



  if (max_refinement_level>0) max_refinement_level = max_refinement_level -1;
  if (max_second_bh_refinement_level>0) max_second_bh_refinement_level = max_second_bh_refinement_level -1;
  if (max_smr_refinement_level>0) max_smr_refinement_level = max_smr_refinement_level - 1;


  if (COORDINATE_SYSTEM == "gr_user") EnrollUserMetric(Cartesian_GR);

  if (METRIC_EVOLUTION)  EnrollUserMetricWithoutPin(Binary_BH_Metric);

  if (COORDINATE_SYSTEM == "gr_user") EnrollUserRadSourceFunction(inner_boundary_source_function);
  // Enroll boundary functions
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x1, FixedBoundary);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary);
    // Enroll boundary functions
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, CustomInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, CustomOuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, CustomOuterX2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, CustomInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, CustomOuterX3);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, CustomInnerX3);



  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);



  a = pin->GetReal("coord", "a");
  m = pin->GetReal("coord", "m");
  m2 = pin->GetOrAddReal("problem", "q", 0.1);

    // Get ratio of specific heats
  Real gamma_adi = pin->GetReal("hydro", "gamma");
  n_adi = 1.0/(gamma_adi-1.0);
  // Prepare various constants for determining primitives
  Real u_crit_sq = m2/(2.0*r_crit);                                          // (HSW 71)
  Real u_crit = -std::sqrt(u_crit_sq);
  Real t_crit = n_adi/(n_adi+1.0) * u_crit_sq/(1.0-(n_adi+3.0)*u_crit_sq);  // (HSW 74)
  c1 = std::pow(t_crit, n_adi) * u_crit * SQR(r_crit);                      // (HSW 68)
  c2 = SQR(1.0 + (n_adi+1.0) * t_crit) * (1.0 - 3.0*m2/(2.0*r_crit));        // (HSW 69)

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

  v_bh2 = pin->GetOrAddReal("problem", "vbh", 0.05);



  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  r_inner_boundary = rh/2.0;

  r_inner_bondi_boundary = 3.0;
  r_outer_bondi_boundary = 10.0;


    // Get mass of black hole
  m2 = q;

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
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  // Get mass and spin of black hole
  m = pcoord->GetMass();
  a = pcoord->GetSpin();


  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);


  // Initialize primitive values
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i=il; i<=iu; ++i) {
        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r,
                                     &theta, &phi);

        Real xprime,yprime,zprime,rprime,Rprime;
        get_prime_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
        Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
        BoostVector(u0,u1,u2,u3, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
        Real uu1 = u1prime - gi(I01,i)/gi(I00,i) * u0prime;
        Real uu2 = u2prime - gi(I02,i)/gi(I00,i) * u0prime;
        Real uu3 = u3prime - gi(I03,i)/gi(I00,i) * u0prime;
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
        phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
        phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }
    }
  }

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {
    // Find normalization
    Real r, theta, phi;
    // GetBoyerLindquistCoordinates(pcoord->x1f(is), pcoord->x2v((jl+ju)/2),
    //                              pcoord->x3v((kl+ku)/2), &r, &theta, &phi);

    r = 3.0;
    Real rho, pgas, ut, ur;
    CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
    Real bbr = 1.0/SQR(r);
    Real bt = 1.0/(1.0-2.0*m2/r) * bbr * ur;
    Real br = (bbr + bt * ur) / ut;
    Real bsq = -(1.0-2.0*m2/r) * SQR(bt) + 1.0/(1.0-2.0*m2/r) * SQR(br);
    Real bsq_over_rho_actual = bsq/rho;
    Real normalization = std::sqrt(bsq_over_rho/bsq_over_rho_actual);

    // Set face-centered field
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          // Set B^1
          if (j != ju+1 && k != ku+1) {
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);

            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);

            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), &b0prime, &b1prime, &b2prime, &b3prime);
            pfield->b.x1f(k,j,i) = b1prime * u0prime - b0prime * u1prime;
          }

          // Set B^2
          if (i != iu+1 && k != ku+1) {
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);
            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), &b0prime, &b1prime, &b2prime, &b3prime);
            pfield->b.x2f(k,j,i) = b2prime * u0prime - b0prime * u2prime;
          }

          // Set B^3
          if (i != iu+1 && j != ju+1) {
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);
            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), &b0prime, &b1prime, &b2prime, &b3prime);
            pfield->b.x3f(k,j,i) = b3prime * u0prime - b0prime * u3prime;


           if (std::isnan(pfield->b.x3f(k,j,i))) 
              fprintf(stderr,"Bx1f nan! r: %g bbr: %g bt: %g br: %g \n ur: %g ur %g \n b0-3: %g %g %g %g \n u0-3: %g %g %g %g \n",
                              rprime,bbr,bt,br,ur,ur,b0,b1,b2,b3,u0,u1,u2,u3);
          }
        }
      }
    }

    // Calculate cell-centered magnetic field
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
                                       ku);
  }

  // Initialize conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
                             kl, ku);

  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();
  return;
}


/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar){


  Real r,th,ph;
  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];

  Real Lorentz = std::sqrt(1.0/(1.0 - SQR(v_bh2)));


  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  int jl = pmb->js;
  int ju = pmb->je;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }



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


          Real xbh,ybh,zbh;
          get_bh_position(t, &xbh,&ybh,&zbh);
          Real fake_xprime = x-xbh;
          Real fake_yprime = y-ybh;
          Real fake_zprime = z-zbh;



          if (std::fabs(fake_zprime)<SMALL) fake_zprime= SMALL;
          Real fake_Rprime = std::sqrt(SQR(fake_xprime) + SQR(fake_yprime) + SQR(fake_zprime));
          Real fake_rprime = SQR(fake_Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(fake_Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(fake_zprime) );
          fake_rprime = std::sqrt(fake_rprime/2.0);

         if (fake_rprime<r_inner_bondi_boundary || fake_rprime>r_outer_bondi_boundary){

          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {

            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &r,
                                         &theta, &phi);
            Real rho, pgas, ut, ur;
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real uu1 = u1prime - gi(I01,i)/gi(I00,i) * u0prime;
            Real uu2 = u2prime - gi(I02,i)/gi(I00,i) * u0prime;
            Real uu3 = u3prime - gi(I03,i)/gi(I00,i) * u0prime;
            prim(IDN,k,j,i) = rho;
            prim(IVX,k,j,i) = uu1;
            prim(IVY,k,j,i) = uu2;
            prim(IVZ,k,j,i) = uu3;
            prim(IPR,k,j,i) = pgas;


         }

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


if (MAGNETIC_FIELDS_ENABLED) {


    Real r,theta,phi;
    r = 3.0;
    Real rho, pgas, ut, ur;
    CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
    Real bbr = 1.0/SQR(r);
    Real bt = 1.0/(1.0-2.0*m2/r) * bbr * ur;
    Real br = (bbr + bt * ur) / ut;
    Real bsq = -(1.0-2.0*m2/r) * SQR(bt) + 1.0/(1.0-2.0*m2/r) * SQR(br);
    Real bsq_over_rho_actual = bsq/rho;
    Real normalization = std::sqrt(bsq_over_rho/bsq_over_rho_actual);



   for (int k=pmb->ks; k<=pmb->ke+1; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je+1; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
#pragma simd
      for (int i=pmb->is; i<=pmb->ie+1; ++i) {


        GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);



        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real t = pmb->pmy_mesh->time;

        Real xprime,yprime,zprime,rprime,Rprime;

        if (j !=pmb->je+1 && k!=pmb->ke+1){
          get_prime_coords(pmb->pcoord->x1f(i),pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){
          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {


                            // if (j != ju+1 && k != ku+1) {
            GetBoyerLindquistCoordinates(pmb->pcoord->x1f(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pmb->pcoord->x1f(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);

            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pmb->pcoord->x1f(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);

            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pmb->pcoord->x1f(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &b0prime, &b1prime, &b2prime, &b3prime);
            pmb->pfield->b.x1f(k,j,i) = b1prime * u0prime - b0prime * u1prime;
            pmb->pfield->b1.x1f(k,j,i) = pmb->pfield->b.x1f(k,j,i);
        }
      }

        if (i!=pmb->ie+1 && k!=pmb->ke+1){
          get_prime_coords(pmb->pcoord->x1v(i),pmb->pcoord->x2f(j),pmb->pcoord->x3v(k), t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){
          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {
            GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2f(j), pmb->pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2f(j), pmb->pcoord->x3v(k), pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pmb->pcoord->x1v(i), pmb->pcoord->x2f(j), pmb->pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);
            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pmb->pcoord->x1v(i), pmb->pcoord->x2f(j), pmb->pcoord->x3v(k), &b0prime, &b1prime, &b2prime, &b3prime);
            pmb->pfield->b.x2f(k,j,i)  = b2prime * u0prime - b0prime * u2prime;
            pmb->pfield->b1.x2f(k,j,i) = pmb->pfield->b.x2f(k,j,i);
        }
      }

        if (i!=pmb->ie+1 && j!=pmb->je+1){
          get_prime_coords(pmb->pcoord->x1v(i),pmb->pcoord->x2v(j),pmb->pcoord->x3f(k), t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){
          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {

            GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3f(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3f(k), pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3f(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);
            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3f(k), &b0prime, &b1prime, &b2prime, &b3prime);
            pmb->pfield->b.x3f(k,j,i) = b3prime * u0prime - b0prime * u3prime;
            pmb->pfield->b1.x3f(k,j,i) = pmb->pfield->b.x3f(k,j,i);
        }
      }




}
}
}

}

}


void apply_inner_boundary_condition_in_boundary_function(MeshBlock *pmb,Coordinates *pcoord,AthenaArray<Real> &prim,FaceField &b, Real t,
                    int is, int ie, int js, int je, int ks, int ke,int ngh){
  Real r,th,ph;
  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];



   for (int k=ks; k<=ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=js; j<=je; ++j) {
      pcoord->CellMetric(k, j, is, ie, g, gi);
#pragma simd
      for (int i=is; i<=ie; ++i) {


         GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),pcoord->x3v(k), &r, &th, &ph);


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

          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);

          Real xprime,yprime,zprime,rprime,Rprime;

          get_prime_coords(x,y,z, t, &xprime,&yprime, &zprime, &rprime,&Rprime);



         if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){

          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {

            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r,
                                         &theta, &phi);
            Real rho, pgas, ut, ur;
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real uu1 = u1prime - gi(I01,i)/gi(I00,i) * u0prime;
            Real uu2 = u2prime - gi(I02,i)/gi(I00,i) * u0prime;
            Real uu3 = u3prime - gi(I03,i)/gi(I00,i) * u0prime;
            prim(IDN,k,j,i) = rho;
            prim(IVX,k,j,i) = uu1;
            prim(IVY,k,j,i) = uu2;
            prim(IVZ,k,j,i) = uu3;
            prim(IPR,k,j,i) = pgas;


         }

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


if (MAGNETIC_FIELDS_ENABLED) {


    Real r,theta,phi;
    r = 3.0;
    Real rho, pgas, ut, ur;
    CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
    Real bbr = 1.0/SQR(r);
    Real bt = 1.0/(1.0-2.0*m2/r) * bbr * ur;
    Real br = (bbr + bt * ur) / ut;
    Real bsq = -(1.0-2.0*m2/r) * SQR(bt) + 1.0/(1.0-2.0*m2/r) * SQR(br);
    Real bsq_over_rho_actual = bsq/rho;
    Real normalization = std::sqrt(bsq_over_rho/bsq_over_rho_actual);



   for (int k=ks; k<=ke+1; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=js; j<=je+1; ++j) {
      pcoord->CellMetric(k, j, is, ie, g, gi);
#pragma simd
      for (int i=is; i<=ie+1; ++i) {


        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),pcoord->x3v(k), &r, &th, &ph);



        Real x = pcoord->x1v(i);
        Real y = pcoord->x2v(j);
        Real z = pcoord->x3v(k);

        Real xprime,yprime,zprime,rprime,Rprime;

        if (j !=je+1 && k!=ke+1){
          get_prime_coords(pcoord->x1f(i),pcoord->x2v(j),pcoord->x3v(k), t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){
          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {


                            // if (j != ju+1 && k != ku+1) {
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);

            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);

            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k), &b0prime, &b1prime, &b2prime, &b3prime);
            b.x1f(k,j,i) = b1prime * u0prime - b0prime * u1prime;
        }
      }

        if (i!=ie+1 && k!=ke+1){
          get_prime_coords(pcoord->x1v(i),pcoord->x2f(j),pcoord->x3v(k), t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){
          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);
            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k), &b0prime, &b1prime, &b2prime, &b3prime);
            b.x2f(k,j,i)  = b2prime * u0prime - b0prime * u2prime;
        }
      }

        if (i!=ie+1 && j!=je+1){
          get_prime_coords(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3f(k), t, &xprime,&yprime, &zprime, &rprime,&Rprime);

          if (rprime<r_inner_bondi_boundary || rprime>r_outer_bondi_boundary){
          // if ( (std::abs(xprime)<r_inner_bondi_boundary  && std::abs(yprime)<r_inner_bondi_boundary  && std::abs(zprime)<r_inner_bondi_boundary ) ||
          //      (std::abs(xprime)>r_outer_bondi_boundary  && std::abs(yprime)>r_outer_bondi_boundary  && std::abs(zprime)>r_outer_bondi_boundary ) )
          // {

            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k),
                                         &r, &theta, &phi);
            Real xprime,yprime,zprime,rprime,Rprime;
            get_prime_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), pmb->pmy_mesh->time, &xprime,&yprime, &zprime, &rprime,&Rprime);
            CalculatePrimitives(rprime, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(rprime);
            bt = 1.0/(1.0-2.0*m2/rprime) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, xprime,yprime,zprime, &u0, &u1, &u2, &u3);
            Real u0prime(0.0), u1prime(0.0), u2prime(0.0), u3prime(0.0);
            BoostVector(u0,u1,u2,u3, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), &u0prime, &u1prime, &u2prime, &u3prime);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, xprime,yprime,zprime, &b0, &b1, &b2, &b3);
            Real b0prime(0.0), b1prime(0.0), b2prime(0.0), b3prime(0.0);
            BoostVector(b0,b1,b2,b3, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k), &b0prime, &b1prime, &b2prime, &b3prime);
            b.x3f(k,j,i) = b3prime * u0prime - b0prime * u3prime;
        }
      }




}
}
}

}
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
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CustomInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void CustomInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {

  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, ngh+2);
  gi.NewAthenaArray(NMETRIC,ngh+2);
  // Initialize primitive values
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->CellMetric(k, j, is-ngh,is-1, g, gi);
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {


        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(is-i), pcoord->x2v(j), pcoord->x3v(k), &r,
                                     &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(is-i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
        Real uu1 = u1 - gi(I01,is-i)/gi(I00,is-i) * u0;
        Real uu2 = u2 - gi(I02,is-i)/gi(I00,is-i) * u0;
        Real uu3 = u3 - gi(I03,is-i)/gi(I00,is-i) * u0;

        prim(IDN,k,j,is-i) = rho;
        prim(IPR,k,j,is-i) = pgas;
        prim(IVX,k,j,is-i) = uu1;
        prim(IVY,k,j,is-i) = uu2;
        prim(IVZ,k,j,is-i) = uu3;


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
        b.x1f(k,j,(is-i)) = 0.0; //pmb->ruser_meshblock_data[2](k,j,(is-i));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = 0.0; //pmb->ruser_meshblock_data[3](k,j,(is-i));
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



  // apply_inner_boundary_condition_in_boundary_function(pmb,pcoord,prim,b,time,is,ie,js,je,ks,ke,ngh);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CustomOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void CustomOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones

  AthenaArray<Real> g, gi,g_tmp,gi_tmp;
  g.NewAthenaArray(NMETRIC, ie+ngh+1);
  gi.NewAthenaArray(NMETRIC,ie+ngh+1);
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->CellMetric(k, j, ie+1,ie+ngh, g, gi);
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(ie+i), pcoord->x2v(j), pcoord->x3v(k), &r,
                                     &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(ie+i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
        Real uu1 = u1 - gi(I01,ie+i)/gi(I00,ie+i) * u0;
        Real uu2 = u2 - gi(I02,ie+i)/gi(I00,ie+i) * u0;
        Real uu3 = u3 - gi(I03,ie+i)/gi(I00,ie+i) * u0;

        prim(IDN,k,j,ie+i) = rho;
        prim(IPR,k,j,ie+i) = pgas;
        prim(IVX,k,j,ie+i) = uu1;
        prim(IVY,k,j,ie+i) = uu2;
        prim(IVZ,k,j,ie+i) = uu3;

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
        b.x1f(k,j,(ie+i+1)) = 0.0; //pmb->ruser_meshblock_data[2](k,j,(ie+i+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = 0.0; //pmb->ruser_meshblock_data[3](k,j,(ie+i));
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
//! \fn void CustomInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void CustomInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {

  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, ie+ngh+1);
  gi.NewAthenaArray(NMETRIC,ie+ngh+1);
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      pcoord->CellMetric(k, js-j, is,ie, g, gi);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {


        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(js-j), pcoord->x3v(k), &r,
                                     &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(js-j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;


        prim(IDN,k,js-j,i) = rho;
        prim(IPR,k,js-j,i) = pgas;
        prim(IVX,k,js-j,i) = uu1;
        prim(IVY,k,js-j,i) = uu2;
        prim(IVZ,k,js-j,i) = uu3;
      }
    }}

    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();


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
        b.x2f(k,(js-j),i) = 0.0; // pmb->ruser_meshblock_data[3](k,(js-j),i);
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
//! \fn void CustomOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void CustomOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones

  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, ie+ngh+1);
  gi.NewAthenaArray(NMETRIC,ie+ngh+1);
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      pcoord->CellMetric(k, je+j, is,ie, g, gi);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {

        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(je+j), pcoord->x3v(k), &r,
                                     &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(je+j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;

        prim(IDN,k,je+j,i) = rho;
        prim(IPR,k,je+j,i) = pgas;
        prim(IVX,k,je+j,i) = uu1;
        prim(IVY,k,je+j,i) = uu2;
        prim(IVZ,k,je+j,i) = uu3;
      }
    }}


    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();

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
        b.x2f(k,(je+j+1),i) = 0.0; // pmb->ruser_meshblock_data[3](k,(je+j+1),i);
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
//! \fn void CustomInnerX3(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void CustomInnerX3(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, ie+ngh+1);
  gi.NewAthenaArray(NMETRIC,ie+ngh+1);
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->CellMetric(ks-k, j, is,ie, g, gi);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {

        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(ks-k), &r,
                                     &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(ks-k), &u0, &u1, &u2, &u3);
        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;


        prim(IDN,ks-k,j,i) = rho;
        prim(IPR,ks-k,j,i) = pgas;
        prim(IVX,ks-k,j,i) = uu1;
        prim(IVY,ks-k,j,i) = uu2;
        prim(IVZ,ks-k,j,i) = uu3;
      }
    }}

    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();

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
        b.x2f((ks-k),j,i) = 0.0; // pmb->ruser_meshblock_data[3]((ks-k),j,i);
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
//! \fn void CustomOuterX3(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void CustomOuterX3(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, ie+ngh+1);
  gi.NewAthenaArray(NMETRIC,ie+ngh+1);
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->CellMetric(ke+k, j, is,ie, g, gi);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {

        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(ke+k), &r,
                                     &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(ke+k), &u0, &u1, &u2, &u3);
        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;

        prim(IDN,ke+k,j,i) = rho;
        prim(IPR,ke+k,j,i) = pgas;
        prim(IVX,ke+k,j,i) = uu1;
        prim(IVY,ke+k,j,i) = uu2;
        prim(IVZ,ke+k,j,i) = uu3;
      }
    }}

    g.DeleteAthenaArray();
    gi.DeleteAthenaArray();

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
        b.x2f((ke+k  ),j,i) = 0.0; //pmb->ruser_meshblock_data[3]((ke+k  ),j,i);
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

void  MeshBlock::PreserveDivbNewMetric(ParameterInput *pin){

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

  Real Lorentz = std::sqrt(1.0/(1.0 - SQR(v_bh2)));
  *xprime = x - xbh;
  *yprime = y - ybh;
  *zprime = Lorentz * (z - zbh);


  if (std::fabs(*zprime)<SMALL) *zprime= SMALL;
  *Rprime = std::sqrt(SQR(*xprime) + SQR(*yprime) + SQR(*zprime));
  *rprime = SQR(*Rprime) - SQR(aprime) + std::sqrt( SQR( SQR(*Rprime) - SQR(aprime) ) + 4.0*SQR(aprime)*SQR(*zprime) );
  *rprime = std::sqrt(*rprime/2.0);

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


// namespace {
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates
// Notes:
//   conversion is trivial in all currently implemented coordinate systems

void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                  Real *ptheta, Real *pphi) {
  if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0 ||
      std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) {
    *pr = x1;
    *ptheta = x2;
    *pphi = x3;
  }

  if (std::strcmp(COORDINATE_SYSTEM, "gr_user") == 0) {

    Real x = x1;
    Real y = x2;
    Real z = x3;
    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);

    *pr = r;
    *ptheta = std::acos(z/r);
    *pphi = std::atan2( (r*y-a*x)/(SQR(r)+SQR(a) ), (a*y+r*x)/(SQR(r) + SQR(a) )  );
  }

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

void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real x1,
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3){
  if (COORDINATE_SYSTEM == "schwarzschild") {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (COORDINATE_SYSTEM == "kerr-schild") {
    Real r = x1;
    Real delta = SQR(r) - 2.0*m2*r + SQR(a);
    *pa0 = a0_bl + 2.0*m2*r/delta * a1_bl;
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
    Real delta = SQR(r) - 2.0*m2*r + SQR(a);
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

void BoostVector(Real a0, Real a1, Real a2, Real a3, Real x1,
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3){

  Real gamma = std::sqrt(1.0/(1.0 - SQR(v_bh2)));
  Real beta = v_bh2;

  *pa0 = gamma * (a0 + beta * a3);
  *pa1 = a1;
  *pa2 = a2; 
  *pa3 = gamma * (a3 + beta * a0);

  return;

}

//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void CalculatePrimitives(Real r, Real temp_min, Real temp_max, Real *prho,
                         Real *ppgas, Real *put, Real *pur) {
  // Calculate solution to (HSW 76)
  Real temp_neg_res = TemperatureMin(r, temp_min, temp_max);
  Real temp;
  if (r <= r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(r, temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(r, temp_neg_res, temp_max);
  }

  // Calculate primitives
  Real rho = std::pow(temp/k_adi, n_adi);             // not same K as HSW
  Real pgas = temp * rho;
  Real ur = c1 / (SQR(r) * std::pow(temp, n_adi));    // (HSW 75)
  Real ut = std::sqrt(1.0/SQR(1.0-2.0*m2/r) * SQR(ur)
                      + 1.0/(1.0-2.0*m2/r));

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *put = ut;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

Real TemperatureMin(Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 100;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

Real TemperatureBisect(Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 20;
  const Real tol_residual = 1.0e-6;
  const Real tol_temperature = 1.0e-6;

  // Find initial residuals
  Real res_min = TemperatureResidual(t_min, r);
  Real res_max = TemperatureResidual(t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 && res_max < 0.0) || (res_min > 0.0 && res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 && res_min < 0.0) || (res_mid > 0.0 && res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

Real TemperatureResidual(Real t, Real r) {
  return SQR(1.0 + (n_adi+1.0) * t)
      * (1.0 - 2.0*m2/r + SQR(c1) / (SQR(SQR(r)) * std::pow(t, 2.0*n_adi))) - c2;
}
// } // namespace


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


  //BOOST //

  Real Lorentz = std::sqrt(1.0/(1.0 - SQR(v_bh2)));
  // Real Lorentz = 1.0;

  Real l0 = l_lowerprime[0];
  Real l3 = l_lowerprime[3];

  l_lowerprime[0] = Lorentz * (l0 - v_bh2 * l3);
  l_lowerprime[3] = Lorentz * (l3 - v_bh2 * l0);




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

  Real dl0_dx1_tmp = dl0prime_dx1;
  Real dl0_dx2_tmp = dl0prime_dx2;
  Real dl0_dx3_tmp = dl0prime_dx3;

  Real dl3_dx1_tmp = dl3prime_dx1;
  Real dl3_dx2_tmp = dl3prime_dx2;
  Real dl3_dx3_tmp = dl3prime_dx3;



  dl0prime_dx1 = Lorentz * (dl0_dx1_tmp - v_bh2 * dl3_dx1_tmp); 
  dl0prime_dx2 = Lorentz * (dl0_dx2_tmp - v_bh2 * dl3_dx2_tmp); 
  dl0prime_dx3 = Lorentz * (dl0_dx3_tmp - v_bh2 * dl3_dx3_tmp); 


  dl3prime_dx1 = Lorentz * (dl3_dx1_tmp - v_bh2 * dl0_dx1_tmp); 
  dl3prime_dx2 = Lorentz * (dl3_dx2_tmp - v_bh2 * dl0_dx2_tmp); 
  dl3prime_dx3 = Lorentz * (dl3_dx3_tmp - v_bh2 * dl0_dx3_tmp); 

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

  dgprime_dx3(I00) = Lorentz * (dfprime_dx3*l_lowerprime[0]*l_lowerprime[0] + fprime * dl0prime_dx3 * l_lowerprime[0] + fprime * l_lowerprime[0] * dl0prime_dx3);
                    // - 2.0 * v_bh2 * (dfprime_dx3 * l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3*l_lowerprime[3]
                    // +                fprime * l_lowerprime[0]*dl3prime_dx3) 
                    // +  SQR(v_bh2) * (dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx3*l_lowerprime[3] 
                    // +                fprime * l_lowerprime[3]*dl3prime_dx3);
  dgprime_dx3(I01) = Lorentz * (dfprime_dx3*l_lowerprime[0]*l_lowerprime[1] + fprime * dl0prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[0] * dl1prime_dx3);
                    // - v_bh2 * (dfprime_dx3 * l_lowerprime[1]*l_lowerprime[3] 
                    // +          fprime * dl1prime_dx3*l_lowerprime[3]
                    // +          fprime * l_lowerprime[1]*dl3prime_dx3);
  dgprime_dx3(I02) = Lorentz * (dfprime_dx3*l_lowerprime[0]*l_lowerprime[2] + fprime * dl0prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[0] * dl2prime_dx3);
                    // - v_bh2 * (dfprime_dx3 * l_lowerprime[2]*l_lowerprime[3] 
                    // +          fprime * dl2prime_dx3*l_lowerprime[3]
                    // +          fprime * l_lowerprime[2]*dl3prime_dx3);
  dgprime_dx3(I03) = Lorentz * (dfprime_dx3*l_lowerprime[0]*l_lowerprime[3] + fprime * dl0prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[0] * dl3prime_dx3);
                    // - v_bh2 * (dfprime_dx3 * l_lowerprime[3]*l_lowerprime[3] 
                    // +          fprime * dl3prime_dx3*l_lowerprime[3]
                    // +          fprime * l_lowerprime[3]*dl3prime_dx3);  
  dgprime_dx3(I11) = Lorentz * (dfprime_dx3*l_lowerprime[1]*l_lowerprime[1] + fprime * dl1prime_dx3 * l_lowerprime[1] + fprime * l_lowerprime[1] * dl1prime_dx3);
  dgprime_dx3(I12) = Lorentz * (dfprime_dx3*l_lowerprime[1]*l_lowerprime[2] + fprime * dl1prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[1] * dl2prime_dx3);
  dgprime_dx3(I13) = Lorentz * (dfprime_dx3*l_lowerprime[1]*l_lowerprime[3] + fprime * dl1prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[1] * dl3prime_dx3);
  dgprime_dx3(I22) = Lorentz * (dfprime_dx3*l_lowerprime[2]*l_lowerprime[2] + fprime * dl2prime_dx3 * l_lowerprime[2] + fprime * l_lowerprime[2] * dl2prime_dx3);
  dgprime_dx3(I23) = Lorentz * (dfprime_dx3*l_lowerprime[2]*l_lowerprime[3] + fprime * dl2prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[2] * dl3prime_dx3);
  dgprime_dx3(I33) = Lorentz * (dfprime_dx3*l_lowerprime[3]*l_lowerprime[3] + fprime * dl3prime_dx3 * l_lowerprime[3] + fprime * l_lowerprime[3] * dl3prime_dx3);





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
  dg_dt(I00) = -(dx_bh2_dt * dgprime_dx1(I00) + dy_bh2_dt * dgprime_dx2(I00) + dz_bh2_dt * dgprime_dx3(I00) );
  dg_dt(I01) = -(dx_bh2_dt * dgprime_dx1(I01) + dy_bh2_dt * dgprime_dx2(I01) + dz_bh2_dt * dgprime_dx3(I01) );
  dg_dt(I02) = -(dx_bh2_dt * dgprime_dx1(I02) + dy_bh2_dt * dgprime_dx2(I02) + dz_bh2_dt * dgprime_dx3(I02) );
  dg_dt(I03) = -(dx_bh2_dt * dgprime_dx1(I03) + dy_bh2_dt * dgprime_dx2(I03) + dz_bh2_dt * dgprime_dx3(I03) );
  dg_dt(I11) = -(dx_bh2_dt * dgprime_dx1(I11) + dy_bh2_dt * dgprime_dx2(I11) + dz_bh2_dt * dgprime_dx3(I11) );
  dg_dt(I12) = -(dx_bh2_dt * dgprime_dx1(I12) + dy_bh2_dt * dgprime_dx2(I12) + dz_bh2_dt * dgprime_dx3(I12) );
  dg_dt(I13) = -(dx_bh2_dt * dgprime_dx1(I13) + dy_bh2_dt * dgprime_dx2(I13) + dz_bh2_dt * dgprime_dx3(I13) );
  dg_dt(I22) = -(dx_bh2_dt * dgprime_dx1(I22) + dy_bh2_dt * dgprime_dx2(I22) + dz_bh2_dt * dgprime_dx3(I22) );
  dg_dt(I23) = -(dx_bh2_dt * dgprime_dx1(I23) + dy_bh2_dt * dgprime_dx2(I23) + dz_bh2_dt * dgprime_dx3(I23) );
  dg_dt(I33) = -(dx_bh2_dt * dgprime_dx1(I33) + dy_bh2_dt * dgprime_dx2(I33) + dz_bh2_dt * dgprime_dx3(I33) );


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